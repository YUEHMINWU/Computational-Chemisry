import numpy as np
import os
from ase.io import read, write as ase_write
from ase import Atoms
from io import StringIO
import sys
import matplotlib.pyplot as plt

# Constants for LiTFSI water system
HARTREE_TO_KCAL_MOL = 627.50960803
BOHR_TO_ANGSTROM = 0.5291772109
FORCE_CONVERSION_FACTOR = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM
CELL_EDGE_LENGTH = 14.936
CUBIC_CELL = [CELL_EDGE_LENGTH] * 3
SPECIES = ["Li", "F", "S", "O", "C", "N", "H"]  # LiTFSI water system elements
N_ATOMS = 244  # Total number of atoms in the system
RUNS = [
    {
        "pos": "../results/litfsi_h2o_relax-pos.xyz",
        "frc": "../results/litfsi_h2o_relax-frc.xyz",
        "ener": "litfsi_h2o_relax-1.ener",
        "start_step": 0,
        "end_step": 7500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re7500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re7500-frc.xyz",
        "ener": "litfsi_h2o_prod_re7500-1.ener",
        "start_step": 7501,
        "end_step": 20500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re20500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re20500-frc.xyz",
        "ener": "litfsi_h2o_prod_re20500-1.ener",
        "start_step": 20501,
        "end_step": 39999,
    },
]
PRIMARY_TRAIN_VAL_FRAMES = 5000  # Total frames for primary model (train + val)
ENSEMBLE_TRAIN_VAL_FRAMES = 500  # Total frames per ensemble model (train + val)
COMBINED_TEST_FRAMES = 500  # Absolute for combined test from last 5 ps
NUM_ENSEMBLES = 5  # Number of ensemble models
NUM_PRIMARY = 1  # Primary model
NUM_INIT_STRUCTS = 5  # Number of diverse initial structures per iteration
FINAL_TEST_FILE = "final_test.extxyz"  # Combined test for final model evaluation
PRIMARY_TRAIN_VAL_FILE = (
    "aimd_trajectory_primary_train_val.extxyz"  # For primary model  # For primary model
)
TRUE_CALIB_FILE = "true_calib.extxyz"
TRUE_CALIB_FRAMES = 500
MIN_DISTANCE_THRESHOLD = 0.8  # Angstrom, below this considered unphysical
MAX_FORCE_THRESHOLD = 150.0  # kcal/mol/Å, above this eliminate from test


def read_xyz_frames(file_path):
    """Read XYZ file and return a dictionary mapping step numbers to frame strings."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        frames = {}
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            n_atoms = int(lines[i].strip())
            comment_line = lines[i + 1].strip()
            try:
                step_str = comment_line.split("i =")[1].split(",")[0].strip()
                step = int(step_str)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse step from comment: {comment_line}")
                i += n_atoms + 2
                continue
            frame_lines = lines[i : i + n_atoms + 2]
            frames[step] = "".join(frame_lines)
            i += n_atoms + 2
        return frames
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        raise


def process_trajectory_chunk(frame_data):
    """Process a chunk of frame data into Atoms objects with per-atom energy and max force magnitude."""
    processed_atoms = []
    for pos_string, force_string, energy_hartree, step in frame_data:
        try:
            pos_atoms = read(StringIO(pos_string), format="xyz")
            energy_kcal_mol = float(energy_hartree) * HARTREE_TO_KCAL_MOL
            energy_per_atom = energy_kcal_mol / N_ATOMS  # Calculate per-atom energy
            force_lines = force_string.strip().split("\n")
            n_atoms = int(force_lines[0].strip())
            if len(pos_atoms) != n_atoms or n_atoms != N_ATOMS:
                raise ValueError(
                    f"Atom count mismatch or unexpected count {n_atoms} at step {step}"
                )
            force_data = []
            for line in force_lines[2 : 2 + n_atoms]:
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"Line has {len(parts)} columns, expected at least 4: {line}"
                    )
                force_components = [float(x) for x in parts[-3:]]
                force_data.append(force_components)
            forces_hartree_bohr = np.array(force_data, dtype=np.float32)
            forces_kcal_mol_angstrom = forces_hartree_bohr * FORCE_CONVERSION_FACTOR
            max_force = np.max(
                np.linalg.norm(forces_kcal_mol_angstrom, axis=1)
            )  # Calculate max force magnitude
            atoms = Atoms(
                symbols=pos_atoms.symbols,
                positions=pos_atoms.positions.astype(np.float32),
            )
            atoms.set_cell(CUBIC_CELL)
            atoms.arrays["forces"] = forces_kcal_mol_angstrom
            atoms.info["energy"] = float(energy_kcal_mol)
            atoms.info["energy_per_atom"] = float(energy_per_atom)
            atoms.info["max_force"] = float(max_force)
            atoms.info["step"] = int(step)
            atoms.set_pbc(True)
            processed_atoms.append(atoms)
        except Exception as e:
            print(f"Error processing frame at step {step}: {e}")
            continue
    return processed_atoms


def sample_dataset(
    valid_atoms,
    output_traj_file,
    frames_to_use,
    random_seed,
    excluded_steps,
):
    # Filter to remaining frames (exclude previously selected steps)
    remaining_indices = [
        i
        for i in range(len(valid_atoms))
        if valid_atoms[i].info["step"] not in excluded_steps
    ]
    if len(remaining_indices) < frames_to_use:
        print(
            f"Warning: Only {len(remaining_indices)} frames remaining, cannot sample {frames_to_use}. Using all remaining."
        )
        selected_indices = remaining_indices
    else:
        remaining_steps = np.array(
            [valid_atoms[i].info["step"] for i in remaining_indices]
        )
        num_intervals = frames_to_use - 1
        min_lag = 1
        step_size = 1
        lags = [min_lag + k * step_size for k in range(num_intervals)]
        total_lag_needed = remaining_steps[-1] - remaining_steps[0]
        current_sum_lag = sum(lags)
        if current_sum_lag > total_lag_needed:
            scale = total_lag_needed / current_sum_lag
            lags = [max(1, int(lag * scale)) for lag in lags]
        selected_steps_list = [remaining_steps[0]]
        current_step = remaining_steps[0]
        for lag in lags:
            next_step = current_step + lag
            idx = np.searchsorted(remaining_steps, next_step)
            if idx < len(remaining_steps):
                current_step = remaining_steps[idx]
                selected_steps_list.append(current_step)
            else:
                break
        while len(selected_steps_list) < frames_to_use and len(remaining_steps) > len(
            selected_steps_list
        ):
            selected_steps_list.append(remaining_steps[-1])
        selected_steps_list = list(set(selected_steps_list))[:frames_to_use]
        step_to_index = {
            remaining_steps[j]: remaining_indices[j]
            for j in range(len(remaining_indices))
        }
        selected_indices = [step_to_index[s] for s in selected_steps_list]
    sampled_atoms = [valid_atoms[i] for i in selected_indices]
    # Save dataset
    ase_write(output_traj_file, sampled_atoms)
    print(f"Successfully wrote {len(sampled_atoms)} frames to {output_traj_file}")
    # Return selected steps for exclusion
    selected_steps = {valid_atoms[i].info["step"] for i in selected_indices}
    return selected_steps


def sample_init_structs(
    valid_atoms,
    output_dir,
    num_structs,
    random_seed,
):
    """Sample random initial structures for MLIP-MD."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_seed)
    selected_indices = rng.choice(len(valid_atoms), num_structs, replace=False)
    selected_atoms = [valid_atoms[i] for i in selected_indices]
    for i, atoms in enumerate(selected_atoms):
        output_file = os.path.join(output_dir, f"init_struct_{i}.extxyz")
        ase_write(output_file, atoms)
        print(f"Saved initial structure {i} to {output_file}")
    return [
        os.path.join(output_dir, f"init_struct_{i}.extxyz") for i in range(num_structs)
    ]


def is_unphysical_structure(atoms, threshold=MIN_DISTANCE_THRESHOLD):
    """Check if the structure is unphysical by checking minimum interatomic distance."""
    distances = atoms.get_all_distances(mic=True)
    min_dist = np.min(distances[np.triu_indices(len(atoms), k=1)])
    return min_dist < threshold, min_dist


if __name__ == "__main__":
    full_output = "aimd_trajectory_full.extxyz"
    if os.path.exists(full_output):
        print(f"Loading existing full trajectory from {full_output}")
        valid_atoms = read(full_output, index=":")
    else:
        # Process full trajectory once
        print("--- Processing full trajectory ---")
        combined_data = {}
        for i, run_info in enumerate(RUNS):
            pos_file, force_file, energy_file = (
                run_info["pos"],
                run_info["frc"],
                run_info["ener"],
            )
            start_step, end_step = run_info["start_step"], run_info["end_step"]
            print(f" > Processing run {i + 1}: {os.path.basename(pos_file)}")
            try:
                pos_frames = read_xyz_frames(pos_file)
                force_frames = read_xyz_frames(force_file)
                energies_data = np.loadtxt(energy_file, skiprows=1, dtype=np.float32)
                energies = {int(row[0]): float(row[4]) for row in energies_data}
                for step in energies:
                    if start_step <= step <= end_step:
                        if step in pos_frames and step in force_frames:
                            combined_data[step] = (
                                pos_frames[step],
                                force_frames[step],
                                energies[step],
                                step,
                            )
                        else:
                            print(f"Warning: Step {step} missing in pos or force files")
            except Exception as e:
                print(f"Error processing run {i + 1}: {e}", file=sys.stderr)
                sys.exit(1)
        frame_data = list(combined_data.values())
        all_atoms = process_trajectory_chunk(frame_data)
        # Filter unphysical
        energies = [atom.info["energy"] for atom in all_atoms]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        valid_atoms = [
            atom
            for atom in all_atoms
            # if mean_energy - 5 * std_energy
            # < atom.info["energy"]
            # < mean_energy + 5 * std_energy
        ]
        # Save full
        ase_write(full_output, valid_atoms)
        print(f"Saved full trajectory ({len(valid_atoms)} frames) to {full_output}")
    # Split into temporal parts
    n_early = 30000
    n_late = 10000
    if len(valid_atoms) != n_early + n_late:
        print(f"Warning: Total frames {len(valid_atoms)}, expected {n_early + n_late}")
    early_atoms = valid_atoms[:n_early]
    late_atoms = valid_atoms[n_early:]
    # Global excluded steps for early
    excluded_early = set()
    # Generate disjoint datasets for primary + ensembles (train/val from early)
    # Primary first
    print("\nGenerating dataset for primary model")
    primary_selected_steps = sample_dataset(
        early_atoms,
        PRIMARY_TRAIN_VAL_FILE,
        PRIMARY_TRAIN_VAL_FRAMES,
        0,  # Seed for primary
        excluded_early,
    )
    excluded_early.update(primary_selected_steps)
    # Then ensembles
    for ensemble_id in range(NUM_ENSEMBLES):
        print(f"\nGenerating dataset for ensemble model {ensemble_id}")
        ensemble_selected_steps = sample_dataset(
            early_atoms,
            f"aimd_trajectory_ensemble_{ensemble_id}_train_val.extxyz",
            ENSEMBLE_TRAIN_VAL_FRAMES,
            ensemble_id + 1,  # Unique seeds
            excluded_early,
        )
        excluded_early.update(ensemble_selected_steps)
    # Sample for true calib
    print("\nGenerating true calibration dataset")
    true_calib_selected_steps = sample_dataset(
        early_atoms,
        TRUE_CALIB_FILE,
        TRUE_CALIB_FRAMES,
        999,  # some seed
        excluded_early,
    )
    # Pre-filter late_atoms for low max_force
    low_force_late = sorted(
        [atom for atom in late_atoms if atom.info["max_force"] <= MAX_FORCE_THRESHOLD],
        key=lambda a: a.info["step"],
    )
    num_low_force = len(low_force_late)
    print(
        f"\nFiltered {num_low_force} frames with max_force <= {MAX_FORCE_THRESHOLD} from last 5 ps."
    )
    if num_low_force < COMBINED_TEST_FRAMES:
        print(
            f"Warning: Only {num_low_force} low-force frames available, using all for test."
        )
        test_atoms_to_use = low_force_late
        test_frames_to_use = num_low_force
    else:
        test_atoms_to_use = low_force_late
        test_frames_to_use = COMBINED_TEST_FRAMES
    # Sample combined test from filtered low-force late
    print("\nGenerating combined test dataset from last 5 ps (low-force frames)")
    test_excluded = set()  # No exclusions for test
    test_selected_steps = sample_dataset(
        test_atoms_to_use,
        FINAL_TEST_FILE,
        test_frames_to_use,
        42,  # Fixed seed for reproducibility
        test_excluded,
    )
    # Sample initial structures for MLIP-MD from early (random)
    print("\nGenerating random initial structures for MLIP-MD")
    init_structs_dir = "init_structs_4test"
    init_struct_files = sample_init_structs(
        early_atoms,
        init_structs_dir,
        NUM_INIT_STRUCTS,
        123,  # Fixed seed
    )
    all_selected_steps = []
    all_selected_max_forces = []
    all_selected_energies_per_atom = []
    # Primary
    primary_train_val = read(PRIMARY_TRAIN_VAL_FILE, index=":")
    primary_steps = [frame.info["step"] for frame in primary_train_val]
    all_selected_steps.extend(primary_steps)
    all_selected_max_forces.extend(
        [frame.info["max_force"] for frame in primary_train_val]
    )
    all_selected_energies_per_atom.extend(
        [frame.info["energy_per_atom"] for frame in primary_train_val]
    )
    # Ensembles
    ensemble_sets = []
    for i in range(NUM_ENSEMBLES):
        train_val = read(f"aimd_trajectory_ensemble_{i}_train_val.extxyz", index=":")
        steps = [frame.info["step"] for frame in train_val]
        all_selected_steps.extend(steps)
        ensemble_sets.append(set(steps))
        all_selected_max_forces.extend([frame.info["max_force"] for frame in train_val])
        all_selected_energies_per_atom.extend(
            [frame.info["energy_per_atom"] for frame in train_val]
        )
    # Add combined test to metrics
    combined_test = read(FINAL_TEST_FILE, index=":")
    test_steps = [frame.info["step"] for frame in combined_test]
    all_selected_steps.extend(test_steps)
    all_selected_max_forces.extend([frame.info["max_force"] for frame in combined_test])
    all_selected_energies_per_atom.extend(
        [frame.info["energy_per_atom"] for frame in combined_test]
    )
    # Print steps where max_force > 150 in test dataset and check if unphysical
    print("\nAnalysis of high max_force (>150) frames in test dataset:")
    high_force_info = []
    for frame in combined_test:
        if frame.info["max_force"] > 150:
            is_unphysical, min_dist = is_unphysical_structure(frame)
            print(
                f"Step: {frame.info['step']}, Max Force: {frame.info['max_force']:.2f}, Min Distance: {min_dist:.2f} Å, Unphysical: {is_unphysical}"
            )
    # Collect separate distributions
    primary_energies = [frame.info["energy_per_atom"] for frame in primary_train_val]
    primary_forces = [frame.info["max_force"] for frame in primary_train_val]
    ensemble_energies = []
    ensemble_forces = []
    for i in range(NUM_ENSEMBLES):
        train_val = read(f"aimd_trajectory_ensemble_{i}_train_val.extxyz", index=":")
        ensemble_energies.extend([frame.info["energy_per_atom"] for frame in train_val])
        ensemble_forces.extend([frame.info["max_force"] for frame in train_val])
    test_energies = [frame.info["energy_per_atom"] for frame in combined_test]
    test_forces = [frame.info["max_force"] for frame in combined_test]
    # Create six plots in one figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # Per-atom energy distributions
    axs[0, 0].hist(primary_energies, bins=50, alpha=0.5, label="Primary Selected")
    axs[0, 0].set_title("Per-Atom Energy - Primary")
    axs[0, 0].set_xlabel("Energy per Atom (kcal/mol/atom)")
    axs[0, 0].set_ylabel("Frame Count")
    axs[0, 0].legend()
    axs[0, 1].hist(ensemble_energies, bins=50, alpha=0.5, label="Ensembles Selected")
    axs[0, 1].set_title("Per-Atom Energy - All Ensembles")
    axs[0, 1].set_xlabel("Energy per Atom (kcal/mol/atom)")
    axs[0, 1].set_ylabel("Frame Count")
    axs[0, 1].legend()
    axs[0, 2].hist(test_energies, bins=50, alpha=0.5, label="Test Selected")
    axs[0, 2].set_title("Per-Atom Energy - Test")
    axs[0, 2].set_xlabel("Energy per Atom (kcal/mol/atom)")
    axs[0, 2].set_ylabel("Frame Count")
    axs[0, 2].legend()
    # Max force distributions
    axs[1, 0].hist(primary_forces, bins=50, alpha=0.5, label="Primary Selected")
    axs[1, 0].set_title("Max Force - Primary")
    axs[1, 0].set_xlabel("Max Force (kcal/mol/Å)")
    axs[1, 0].set_ylabel("Frame Count")
    axs[1, 0].legend()
    axs[1, 1].hist(ensemble_forces, bins=50, alpha=0.5, label="Ensembles Selected")
    axs[1, 1].set_title("Max Force - All Ensembles")
    axs[1, 1].set_xlabel("Max Force (kcal/mol/Å)")
    axs[1, 1].set_ylabel("Frame Count")
    axs[1, 1].legend()
    axs[1, 2].hist(test_forces, bins=50, alpha=0.5, label="Test Selected")
    axs[1, 2].set_title("Max Force - Test")
    axs[1, 2].set_xlabel("Max Force (kcal/mol/Å)")
    axs[1, 2].set_ylabel("Frame Count")
    axs[1, 2].legend()
    plt.tight_layout()
    plt.show()
    # Metrics
    total_unique = len(set(all_selected_steps))
    total_selections = len(all_selected_steps)
    coverage = total_unique / len(valid_atoms) * 100
    jaccards = []
    for i in range(NUM_ENSEMBLES):
        for j in range(i + 1, NUM_ENSEMBLES):
            intersection = len(ensemble_sets[i] & ensemble_sets[j])
            union = len(ensemble_sets[i] | ensemble_sets[j])
            jaccards.append(intersection / union if union > 0 else 0)
    avg_jaccard = np.mean(jaccards) if jaccards else 0
    print(f"Total unique frames across models: {total_unique}")
    print(f"Coverage of full trajectory: {coverage:.2f}%")
    print(f"Average Jaccard similarity between ensemble pairs: {avg_jaccard:.4f}")
