import torch
import numpy as np
from ase.io import read, write
from ase.neighborlist import NeighborList
from nequip.ase import NequIPCalculator
import ase.data  # Added for atomic number lookup
import os
import subprocess  # For triggering final training
import sys
from tqdm import tqdm  # Added for progress bars in loops
import gc  # Added for memory cleanup
import argparse
import glob
from ase import Atoms


class CustomNequIPCalculator(NequIPCalculator):
    def __init__(self, model, device, chemical_symbols, cutoff, periodic):
        super().__init__(model=model, device=device)
        self.chemical_symbols = chemical_symbols
        self.cutoff = cutoff
        self.periodic = periodic
        # Create lookup table for atom types (from type_mapper.py)
        self.lookup_table = torch.full(
            (max(ase.data.atomic_numbers.values()) + 1,), -1, dtype=torch.long
        )
        for idx, sym in enumerate(self.chemical_symbols):
            atomic_number = ase.data.atomic_numbers[sym]
            self.lookup_table[atomic_number] = idx

    def calculate(self, atoms, properties, system_changes):
        # Construct data dictionary
        data = {}
        data["pos"] = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        data["atom_types"] = self.lookup_table[atomic_numbers]
        if data["atom_types"].min() < 0:
            raise ValueError("Encountered an atomic number not in chemical_symbols")
        data["num_atoms"] = torch.tensor([len(atoms)], dtype=torch.long)
        if self.periodic and atoms.pbc.any():
            data["cell"] = torch.tensor(atoms.get_cell().array, dtype=torch.float32)
        else:
            data["cell"] = None
        # Build neighbor list (aligned with neighborlist.py)
        cutoffs = [self.cutoff] * len(atoms)
        neighbor_list = NeighborList(
            cutoffs, self_interaction=False, bothways=True, skin=0.0
        )
        neighbor_list.update(atoms)
        indices = []
        edge_shifts = []
        cell = atoms.get_cell() if self.periodic and atoms.pbc.any() else None
        for i in range(len(atoms)):
            neighbors, offsets = neighbor_list.get_neighbors(i)
            for j, offset in zip(neighbors, offsets):
                indices.append([i, j])  # Include all directed edges
                if cell is not None:
                    edge_shifts.append(offset)  # Use offset (not -offset)
        data["edge_index"] = torch.tensor(indices, dtype=torch.long).t()
        if edge_shifts:
            data["edge_cell_shift"] = torch.tensor(
                np.array(edge_shifts), dtype=torch.float32
            )
        else:
            data["edge_cell_shift"] = None
        # Move to device
        for key in data:
            if data[key] is not None and isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)
        # Predict
        out = self.model(data)
        self.results = {}
        if "energy" in properties and "total_energy" in out:
            self.results["energy"] = out["total_energy"].item()
        if "forces" in properties and "forces" in out:
            self.results["forces"] = out["forces"].detach().cpu().numpy()
        # Add other properties if needed (e.g., stress, virial)
        if "stress" in properties and "stress" in out:
            self.results["stress"] = out["stress"].detach().cpu().numpy()
        if "virial" in properties and "virial" in out:
            self.results["virial"] = out["virial"].detach().cpu().numpy()
        return self.results


class ConformalPrediction:
    """
    Performs quantile regression on score functions to obtain the estimated qhat
        on calibration data and apply to test data during prediction.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, residuals_calib, heurestic_uncertainty_calib) -> None:
        # score function
        scores = np.abs(residuals_calib / heurestic_uncertainty_calib)
        n = len(residuals_calib)
        if n == 0:
            self.qhat = 1.0  # Default if no calibration data
            return
        qhat = np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = qhat

    def predict(self, heurestic_uncertainty_test):
        cp_uncertainty_test = heurestic_uncertainty_test * self.qhat
        return cp_uncertainty_test, self.qhat


def split_test_calib(full_test_X, per_calib, seed=0):
    """
    Uniformly sample the test data at random to split into test and calibration.
    """
    np.random.seed(seed)
    num_total = len(full_test_X)
    num_calib = int(per_calib * num_total)
    rand_idx = np.random.permutation(num_total)
    calib_idx = rand_idx[:num_calib]
    test_idx = rand_idx[num_calib:]
    calib_X = [full_test_X[i] for i in calib_idx]
    test_X = [full_test_X[i] for i in test_idx]
    return test_X, calib_X, test_idx, calib_idx


# Configuration
CHEMICAL_SYMBOLS = ["Li", "F", "S", "O", "C", "N", "H"]
CUTOFF = 6.0
NUM_ENSEMBLES = 5
SUBSAMPLE_RATE = 1
FINAL_TEST_FILE = "final_test.extxyz"
PRIMARY_TRAIN_VAL_FILE = "aimd_trajectory_primary_train_val.extxyz"
TRUE_CALIB_FILE = "true_calib.extxyz"
MD_RESULTS_DIR = "../results"
ALPHA = 0.1  # Corresponds to 90% confidence
PRIMARY_MODEL_DIR = "../results/allegro_model_output_primary"


def parse_lammps_thermo(thermo_file):
    """Parses a LAMMPS thermo log file created by 'fix print' or standard thermo."""
    energy_map = {}
    if not os.path.exists(thermo_file):
        print(f"Warning: Thermo file not found: {thermo_file}")
        return energy_map
    pe_idx = None  # To be determined from header
    with open(thermo_file, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if line.strip().startswith("Step"):
                # Header line, find PotEng column
                if "PotEng" in parts:
                    pe_idx = parts.index("PotEng")
                continue
            if pe_idx is None:
                # Default to 1 if no header found
                pe_idx = 1
            try:
                step = int(parts[0])
                energy = float(parts[pe_idx])
                energy_map[step] = energy
            except (ValueError, IndexError):
                continue
    return energy_map


def parse_forces_dump(frc_file):
    """Manually parse forces from LAMMPS dump file (since it lacks positions)."""
    frames = []
    with open(frc_file, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            i += 1
            # Skip timestep value
            i += 1
        if "ITEM: NUMBER OF ATOMS" in lines[i]:
            i += 1
            natoms = int(lines[i].strip())
            i += 1
        if "ITEM: BOX BOUNDS" in lines[i]:
            i += 1
            # Skip 3 box lines
            i += 3
        if "ITEM: ATOMS" in lines[i]:
            # Assume 'element fx fy fz'
            i += 1
            forces = []
            for j in range(natoms):
                parts = lines[i + j].strip().split()
                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid force line in {frc_file}: {lines[i + j]}"
                    )
                fx = float(parts[1])
                fy = float(parts[2])
                fz = float(parts[3])
                forces.append([fx, fy, fz])
            frames.append(np.array(forces))
            i += natoms
    # print(f"Parsed {len(frames)} force frames from {frc_file}")
    return frames


def process_lammps_traj(pos_file, frc_file, thermo_file):
    """
    Process LAMMPS pos, frc, and thermo files into a list of ASE Atoms objects.
    This version uses the robust, built-in ASE reader for LAMMPS dump files.
    """
    try:
        # Use ASE's built-in reader for LAMMPS text dump files.
        # This automatically handles headers, cell, PBC, positions, and symbols.
        pos_atoms_list = read(pos_file, index=":", format="lammps-dump-text")
        # print(f"Read {len(pos_atoms_list)} position frames from {pos_file}")
    except Exception as e:
        print("ERROR: Failed to parse position LAMMPS dump file using ASE.")
        print(f"  File: {pos_file}")
        print(f"  ASE Error: {e}")
        return []
    # Manually parse forces since dump may lack positions
    frc_forces = parse_forces_dump(frc_file)
    energy_map = parse_lammps_thermo(thermo_file)
    # print(f"Parsed {len(energy_map)} energies from {thermo_file}")
    if len(pos_atoms_list) != len(frc_forces):
        print(
            f"Mismatch in number of frames between position ({len(pos_atoms_list)}) and forces ({len(frc_forces)})."
        )
        return []
    dump_freq = 100  # Assuming dump frequency is 100 steps, starting at step 0.
    processed_frames = []
    for i, frame in enumerate(pos_atoms_list):
        # Set forces from manual parse
        frame.arrays["forces"] = frc_forces[i]
        # Infer step and assign energy from the thermo log.
        current_step = i * dump_freq
        energy = energy_map.get(current_step)
        frame.info["step"] = current_step
        if energy is not None:
            frame.info["energy"] = energy
        else:
            frame.info["energy"] = 0.0  # Placeholder to include frame
            print(
                f"Warning: Using placeholder energy 0.0 for step {current_step} in {thermo_file}."
            )
        processed_frames.append(frame)
    print(f"Processed {len(processed_frames)} frames for this trajectory.")
    return processed_frames


def get_ensemble_uncertainties(
    chemical_symbols, cutoff, data_file, traj_files, iter_num
):
    if os.path.exists(data_file):
        try:
            print(f"Loading data from {data_file}...")
            return np.load(data_file, allow_pickle=True).item()
        except (EOFError, ValueError, OSError) as e:
            print(f"Error loading {data_file}: {e}. Recomputing UQ data.")
            os.remove(data_file) if os.path.exists(data_file) else None
        except Exception as e:
            print(f"Unexpected error loading {data_file}: {e}. Recomputing UQ data.")
            os.remove(data_file) if os.path.exists(data_file) else None
    temp_file = f"temp_mlpmd_iter_{iter_num}.extxyz"
    if os.path.exists(temp_file):
        print(f"Loading combined trajectory from {temp_file}...")
        all_frames = read(temp_file, index=":")
    else:
        print(f"Combining trajectories into {temp_file}...")
        all_frames = []
        for pos_file, frc_file, thermo_file in traj_files:
            frames = process_lammps_traj(pos_file, frc_file, thermo_file)
            all_frames.extend(frames)
            print(f"Processed {len(frames)} frames from {pos_file}")
        if all_frames:
            write(temp_file, all_frames)
            print(f"Saved combined trajectory to {temp_file}")
        else:
            raise ValueError(
                "No frames processed from trajectories. Check file formats and contents."
            )
    # Load ensemble models
    calculators = []
    for i in range(NUM_ENSEMBLES):
        model_path = f"../results/allegro_model_output_{i}/deployed.nequip.pth"
        model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        calc = CustomNequIPCalculator(
            model=model,
            device="cpu",
            chemical_symbols=chemical_symbols,
            cutoff=cutoff,
            periodic=True,
        )
        calculators.append(calc)
    print("Ensemble models loaded.")
    # Read subsampled frames from combined trajectory
    subsampled = all_frames[::SUBSAMPLE_RATE]
    print(
        f"Subsampled {len(subsampled)} frames from MLIP trajectories for uncertainty detection."
    )
    # Split into calib and test
    per_calib = 0.1
    test_X, calib_X, test_subidx, calib_subidx = split_test_calib(
        subsampled, per_calib=per_calib
    )
    print(f"Split into {len(calib_X)} calib and {len(test_X)} test frames for UQ.")
    # Load true calibration labels
    true_calib_atoms = read(TRUE_CALIB_FILE, index=":")
    if len(true_calib_atoms) != len(calib_X):
        raise ValueError("Mismatch between MLIP calib frames and true_calib frames.")
    # For calib: collect per-atom residuals and heuristics
    calib_abs_err_per_atom_list = []
    calib_std_per_atom_list = []
    for i, atoms in enumerate(tqdm(calib_X, desc="Processing calib frames")):
        pred_forces_list = []
        for calc in calculators:
            atoms.calc = calc
            try:
                pred_forces = atoms.get_forces()
            except Exception as e:
                print(f"Error for calib frame: {e}")
                continue
            pred_forces_list.append(pred_forces)
        if len(pred_forces_list) < NUM_ENSEMBLES:
            continue
        forces_array = np.stack(pred_forces_list)  # (models, natoms, 3)
        mean_forces = np.mean(forces_array, axis=0)
        true_forces = true_calib_atoms[
            i
        ].get_forces()  # DFT labels for corresponding frame
        abs_err_per_atom = np.mean(np.abs(true_forces - mean_forces), axis=1)
        calib_abs_err_per_atom_list.append(abs_err_per_atom)
        sqdev_per_model_per_atom = np.mean(
            (forces_array - mean_forces[None, :, :]) ** 2, axis=2
        )
        var_per_atom = np.mean(sqdev_per_model_per_atom, axis=0)
        std_per_atom = np.sqrt(var_per_atom)
        calib_std_per_atom_list.append(std_per_atom)
    calib_residual_flat = np.concatenate(calib_abs_err_per_atom_list)
    calib_heuristic_flat = np.concatenate(calib_std_per_atom_list)
    # Fit CP
    cp = ConformalPrediction(alpha=ALPHA)
    cp.fit(calib_residual_flat, calib_heuristic_flat)
    print(f"Calibration q_hat: {cp.qhat:.4f}")
    # For test: collect per-atom heuristics, natoms, avg heuristic, rmse
    test_std_per_atom_list = []
    test_heuristic_avg = []
    test_rmse = []
    test_natoms = []
    for atoms in tqdm(test_X, desc="Processing test frames"):
        pred_forces_list = []
        for calc in calculators:
            atoms.calc = calc
            try:
                pred_forces = atoms.get_forces()
            except Exception as e:
                print(f"Error for test frame: {e}")
                continue
            pred_forces_list.append(pred_forces)
        if len(pred_forces_list) < NUM_ENSEMBLES:
            continue
        forces_array = np.stack(pred_forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        true_forces = atoms.arrays["forces"]  # Original MLIP forces (for RMSE, not CP)
        rmse = np.sqrt(np.mean((true_forces - mean_forces) ** 2))
        test_rmse.append(rmse)
        sqdev_per_model_per_atom = np.mean(
            (forces_array - mean_forces[None, :, :]) ** 2, axis=2
        )
        var_per_atom = np.mean(sqdev_per_model_per_atom, axis=0)
        std_per_atom = np.sqrt(var_per_atom)
        test_std_per_atom_list.append(std_per_atom)
        test_heuristic_avg.append(np.mean(std_per_atom))
        test_natoms.append(len(atoms))
    test_heuristic_flat = np.concatenate(test_std_per_atom_list)
    test_unc_flat, qhat = cp.predict(test_heuristic_flat)
    # Compute max cal_unc per frame for selection
    cum = 0
    max_cal_unc = []
    for nat in test_natoms:
        unc_atoms = test_unc_flat[cum : cum + nat]
        max_unc = np.max(unc_atoms)
        max_cal_unc.append(max_unc)
        cum += nat
    results = {
        "unc_scores": np.array(test_heuristic_avg),
        "cal_unc_scores": np.array(test_heuristic_avg) * qhat,
        "max_cal_unc_scores": np.array(max_cal_unc),
        "rmse_scores": np.array(test_rmse),
        "frame_indices": np.arange(len(test_X)),
        "test_frames": test_X,
    }
    # Detach calculators to make results serializable
    for frame in results["test_frames"]:
        frame.calc = None
    np.save(data_file, results)
    print(f"Saved uncertainty data to {data_file}.")
    del calculators
    gc.collect()
    return results


MAX_ALLOWED_ENERGY_KCAL_MOL_per_atom = -7890.0
num_atoms = 244
MAX_ALLOWED_FORCE_KCAL_MOL_A = 800.0
MIN_DISTANCE_CUTOFF = 0.8


def is_physical(frame):
    distances = frame.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < MIN_DISTANCE_CUTOFF:
        return False
    return True


def compile_augmented_dataset(unc_results, added_file, model_dir, iter_num):
    if iter_num > 0:
        base_file = f"augmented_primary_train_val_iter_{iter_num - 1}.extxyz"
        if os.path.exists(base_file):
            base_frames = read(base_file, index=":")
        else:
            base_frames = read(PRIMARY_TRAIN_VAL_FILE, index=":")
    else:
        base_frames = read(PRIMARY_TRAIN_VAL_FILE, index=":")
    low = 3.0
    upp = 8.0
    print(
        f"Calibrated uncertainty thresholds: low={low:.1f}, upp={upp:.1f} (e.g., kcal/mol·Å)"
    )
    high_unc_indices = np.where(
        (unc_results["max_cal_unc_scores"] >= low)
        & (unc_results["max_cal_unc_scores"] < upp)
    )[0]
    sorted_high_unc_idx = high_unc_indices[
        np.argsort(unc_results["max_cal_unc_scores"][high_unc_indices])[::-1]
    ]
    # Load DFT frames for RMSE and uncertainty calculation (not for labeling)
    true_test_atoms = read(FINAL_TEST_FILE, index=":")
    # Load ensemble models for computing errors and uncertainties
    calculators = []
    for i in range(NUM_ENSEMBLES):
        model_path = f"../results/allegro_model_output_{i}/deployed.nequip.pth"
        model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        calc = CustomNequIPCalculator(
            model=model,
            device="cpu",
            chemical_symbols=CHEMICAL_SYMBOLS,
            cutoff=CUTOFF,
            periodic=True,
        )
        calculators.append(calc)
    # Compute qhat from unc_results
    qhat_index = np.argmax(unc_results["unc_scores"])
    if unc_results["unc_scores"][qhat_index] > 0:
        qhat = (
            unc_results["cal_unc_scores"][qhat_index]
            / unc_results["unc_scores"][qhat_index]
        )
    else:
        qhat = 1.0
    # Collect data for plotting
    per_atom_err = []
    per_atom_unc = []
    high_unc_frames = []
    physical_frames_added = 0
    unphysical_frames_rejected = 0
    print(
        f"Screening {len(sorted_high_unc_idx)} high-uncertainty candidates for physical realism..."
    )
    pos_tolerance = 1e-5  # Tolerance for position RMSE in Å
    for idx in sorted_high_unc_idx:
        frame = unc_results["test_frames"][idx]
        if not is_physical(frame):
            unphysical_frames_rejected += 1
            continue
        # Find best matching DFT frame for RMSE/unc calculation (no labeling)
        best_dft_idx = None
        min_rmse = float("inf")
        for dft_idx, dft_frame in enumerate(true_test_atoms):
            dft_pos = dft_frame.get_positions()
            pos_rmse = np.sqrt(np.mean((frame.get_positions() - dft_pos) ** 2))
            if pos_rmse < pos_tolerance and pos_rmse < min_rmse:
                min_rmse = pos_rmse
                best_dft_idx = dft_idx
        if best_dft_idx is None:
            print(f"No matching DFT frame for MLIP idx {idx}; skipping.")
            continue
        dft_frame = true_test_atoms[best_dft_idx]
        dft_energy_per_atom = dft_frame.info.get("energy", 0) / num_atoms
        # Check DFT energy
        if dft_energy_per_atom > MAX_ALLOWED_ENERGY_KCAL_MOL_per_atom:
            unphysical_frames_rejected += 1
            continue
        dft_forces = dft_frame.get_forces()
        # Check max DFT force
        if np.any(dft_forces):
            max_force = np.max(np.linalg.norm(dft_forces, axis=1))
            if max_force > MAX_ALLOWED_FORCE_KCAL_MOL_A:
                unphysical_frames_rejected += 1
                continue
        # Compute ensemble predictions for plot data
        pred_forces_list = []
        for calc in calculators:
            frame.calc = calc
            try:
                pred_forces = frame.get_forces()
            except Exception as e:
                print(f"Error computing forces for frame idx {idx}: {e}")
                continue
            pred_forces_list.append(pred_forces)
        if len(pred_forces_list) < NUM_ENSEMBLES:
            continue
        forces_array = np.stack(pred_forces_list)
        mean_forces = np.mean(forces_array, axis=0)
        # Compute per-atom error (mean abs over components)
        abs_err_per_atom = np.mean(np.abs(dft_forces - mean_forces), axis=1)
        # Compute per-atom heuristic unc
        sqdev_per_model_per_atom = np.mean(
            (forces_array - mean_forces[None, :, :]) ** 2, axis=2
        )
        var_per_atom = np.mean(sqdev_per_model_per_atom, axis=0)
        std_per_atom = np.sqrt(var_per_atom)
        # Apply CP
        cp_unc_per_atom = std_per_atom * qhat
        # Collect data
        per_atom_err.extend(abs_err_per_atom)
        per_atom_unc.extend(cp_unc_per_atom)
        # No DFT labeling; add original frame if passes position tolerance (already checked)
        high_unc_frames.append(frame)
        physical_frames_added += 1
    print(f"Screening complete. Added {physical_frames_added} new physical frames.")
    print(f"Rejected {unphysical_frames_rejected} unphysical frames.")
    failed_indices = np.where(unc_results["max_cal_unc_scores"] >= upp)[0]
    print(f"Skipped {len(failed_indices)} frames with cal_unc >= {upp:.1f}")
    # Save data for plotting
    plot_data = {
        "atomic_force_uncertainty": np.array(per_atom_unc),
        "atomic_force_rmse": np.array(per_atom_err),
    }
    plot_file = f"unc_rmse_plot_iter_{iter_num}.npy"
    np.save(plot_file, plot_data)
    print(f"Saved uncertainty vs RMSE data to {plot_file}.")
    # Save only the new high_unc_frames (without base, without DFT labels)
    augmented_file = f"augmented_dataset_iter{iter_num}.extxyz"
    if high_unc_frames:
        write(augmented_file, high_unc_frames)
        print(
            f"Augmented dataset (new frames only) size: {len(high_unc_frames)} saved to {augmented_file}"
        )
    else:
        print("No frames added; no augmented file saved.")
    del calculators
    gc.collect()
    return len(high_unc_frames)


def get_predictions(model_path, test_file, chemical_symbols, cutoff, data_file=None):
    if data_file and os.path.exists(data_file):
        print(f"Loading data from {data_file}...")
        return np.load(data_file, allow_pickle=True).item()
    print("Loading model...")
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    calculator = CustomNequIPCalculator(
        model=model,
        device="cpu",
        chemical_symbols=chemical_symbols,
        cutoff=cutoff,
        periodic=True,
    )
    print("Model loaded successfully.")
    print(f"Reading test data from {test_file}...")
    test_frames = read(test_file, index=":")
    print(f"Found {len(test_frames)} frames in the test file.")
    true_energies_per_atom = []
    pred_energies_per_atom = []
    true_forces_flat = []
    pred_forces_flat = []
    for i, atoms in enumerate(tqdm(test_frames, desc="Getting predictions")):
        true_total_energy = atoms.get_potential_energy()
        true_forces = atoms.get_forces()
        num_atoms = len(atoms)
        true_energies_per_atom.append(true_total_energy / num_atoms)
        true_forces_flat.extend(true_forces.flatten())
        atoms.calc = calculator
        try:
            pred_total_energy = atoms.get_potential_energy()
            pred_forces = atoms.get_forces()
        except Exception as e:
            print(f"Error computing predictions for frame {i}: {e}")
            continue
        pred_energies_per_atom.append(pred_total_energy / num_atoms)
        pred_forces_flat.extend(pred_forces.flatten())
    results = {
        "true_energy": np.array(true_energies_per_atom),
        "pred_energy": np.array(pred_energies_per_atom),
        "true_force": np.array(true_forces_flat),
        "pred_force": np.array(pred_forces_flat),
    }
    if data_file:
        np.save(data_file, results)
        print(f"Saved data to {data_file}.")
    return results


def get_rmse(model_path, test_file, chemical_symbols, cutoff, parity_file=None):
    results = get_predictions(
        model_path, test_file, chemical_symbols, cutoff, data_file=parity_file
    )
    force_rmse = np.sqrt(np.mean((results["true_force"] - results["pred_force"]) ** 2))
    return force_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CP uncertainty for AL augmentation"
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
        default=MD_RESULTS_DIR,
        help="Directory with MLIP trajectories",
    )
    parser.add_argument(
        "--save_parity",
        action="store_true",
        help="Save parity data for the final model",
    )
    parser.add_argument(
        "--compute_rmse",
        action="store_true",
        help="Compute and print RMSE only",
    )
    parser.add_argument(
        "--compute_rmse_and_parity",
        action="store_true",
        help="Compute RMSE and save parity data",
    )
    parser.add_argument(
        "--augment_only",
        action="store_true",
        help="Perform augmentation only (assumes UQ done)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Iteration number for per-iter files",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=PRIMARY_MODEL_DIR,
        help="Directory of the primary model to use for RMSE",
    )
    args = parser.parse_args()
    model_path = os.path.join(args.model_dir, "deployed.nequip.pth")
    if args.compute_rmse:
        # Compute and print RMSE only
        rmse = get_rmse(model_path, FINAL_TEST_FILE, CHEMICAL_SYMBOLS, CUTOFF)
        print(f"Force RMSE on test set: {rmse:.4f} kcal/mol/Å")
        sys.exit(0)
    if args.compute_rmse_and_parity:
        if args.iter is None:
            print("Error: --iter required for --compute_rmse_and_parity.")
            sys.exit(1)
        parity_file = f"temp_parity_iter_{args.iter}.npy"
        rmse = get_rmse(
            model_path,
            FINAL_TEST_FILE,
            CHEMICAL_SYMBOLS,
            CUTOFF,
            parity_file=parity_file,
        )
        print(f"Force RMSE on test set: {rmse:.4f} kcal/mol/Å")
        sys.exit(0)
    if args.save_parity:
        get_predictions(
            model_path,
            FINAL_TEST_FILE,
            CHEMICAL_SYMBOLS,
            CUTOFF,
            "final_parity.npy",
        )
        sys.exit(0)
    if args.iter is None:
        print("Error: --iter required for UQ/augmentation.")
        sys.exit(1)
    unc_file = f"unc_data_iter_{args.iter}.npy"
    added_file = f"added_frames_iter_{args.iter}.extxyz"
    pos_files = sorted(
        glob.glob(os.path.join(args.traj_dir, f"iter{args.iter}_*-md-pos.xyz"))
    )
    frc_files = sorted(
        glob.glob(os.path.join(args.traj_dir, f"iter{args.iter}_*-md-frc.xyz"))
    )
    thermo_files = sorted(
        glob.glob(os.path.join(args.traj_dir, f"iter{args.iter}_*-md-thermo.log"))
    )
    if not (len(pos_files) == len(frc_files) == len(thermo_files)):
        raise FileNotFoundError(
            "Mismatch between pos, frc, and thermo files for this iteration."
        )
    traj_files = list(zip(pos_files, frc_files, thermo_files))
    if args.augment_only:
        try:
            unc_results = np.load(unc_file, allow_pickle=True).item()
        except (EOFError, ValueError, OSError) as e:
            print(
                f"Error loading {unc_file} for augmentation: {e}. Recomputing UQ data."
            )
            unc_results = get_ensemble_uncertainties(
                CHEMICAL_SYMBOLS, CUTOFF, unc_file, traj_files, args.iter
            )
    else:
        unc_results = get_ensemble_uncertainties(
            CHEMICAL_SYMBOLS, CUTOFF, unc_file, traj_files, args.iter
        )
    augmented_size = compile_augmented_dataset(
        unc_results, added_file, args.model_dir, args.iter
    )
    rmse = get_rmse(
        model_path,
        FINAL_TEST_FILE,
        CHEMICAL_SYMBOLS,
        CUTOFF,
    )
    print(f"Force RMSE on test set: {rmse:.4f} kcal/mol/Å")
