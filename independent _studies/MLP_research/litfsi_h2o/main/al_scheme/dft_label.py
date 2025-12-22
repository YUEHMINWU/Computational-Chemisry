import os
import subprocess
import shutil
import argparse
import sys
from ase.io import read, write
from ase import Atoms
from tqdm import tqdm
import numpy as np

PRIMARY_TRAIN_VAL_FILE = "aimd_trajectory_primary_train_val.extxyz"

HARTREE_TO_KCAL_MOL = 627.50960803
BOHR_TO_ANGSTROM = 0.5291772109
FORCE_CONVERSION_FACTOR = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM


def run_cp2k_single_point(
    frame,
    template_inp="dft_label.inp",
    label="frame",
    n_cores=4,
    work_dir=".",
    mode="augment",
    iter_num=None,
):
    temp_dir = os.path.join(work_dir, "dft_label_cp2k")
    os.makedirs(temp_dir, exist_ok=True)
    # Write coord to XYZ
    coord_file = os.path.join(temp_dir, "input.xyz")
    write(coord_file, frame, format="xyz")
    # Copy and update template inp
    shutil.copy(template_inp, temp_dir)
    inp_file = os.path.join(temp_dir, "dft_label.inp")
    with open(inp_file, "r") as f:
        lines = f.readlines()
    updated_lines = []
    for line in lines:
        if "COORD_FILE_NAME" in line:
            line = f" COORD_FILE_NAME {os.path.abspath(coord_file)}\n"
        updated_lines.append(line)
    with open(inp_file, "w") as f:
        f.write("".join(updated_lines))
    # Copy DFT-D3 file
    dftd3_src = "/opt/homebrew/share/cp2k/data/dftd3.dat"
    dftd3_dst = os.path.join("../logs/", "dftd3.dat")
    if os.path.exists(dftd3_src):
        shutil.copyfile(dftd3_src, dftd3_dst)
    else:
        raise FileNotFoundError(f"DFT-D3 parameter file not found: {dftd3_src}")
    # Run CP2K
    cp2k_ssmp = "/opt/homebrew/bin/cp2k.ssmp"
    if mode == "augment":
        out_dir = f"../logs/augment_label_{iter_num}"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{label}.out")
    elif mode == "calib":
        out_dir = f"../logs/calib_label_{iter_num}"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{label}.out")
    else:
        out_file = os.path.join("../logs/", f"{label}.out")
    cmd = [
        cp2k_ssmp,
        "-i",
        inp_file,
        "-o",
        out_file,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"CP2K failed for frame {label}: {e}"
        raise RuntimeError(error_msg)
    # Parse energy from .out
    energy = None
    with open(out_file, "r") as f:
        for line in f:
            if "ENERGY| Total FORCE_EVAL" in line:
                energy = float(line.split()[-1])
                break
    if energy is None:
        raise ValueError(f"Energy not found in {out_file}")
    # Parse forces from .out
    forces = parse_forces_from_out(out_file)
    # Convert units
    energy_kcal = energy * HARTREE_TO_KCAL_MOL
    forces_kcal_ang = forces * FORCE_CONVERSION_FACTOR
    return energy_kcal, forces_kcal_ang


def parse_forces_from_out(out_file):
    with open(out_file, "r") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "FORCES| Atomic forces [hartree/bohr]" in line.strip():
            start = i + 2  # Skip header line "FORCES| Atom x y z |f|"
            break
    if start is None:
        raise ValueError("No atomic forces section found in output")
    forces = []
    while (
        start < len(lines)
        and lines[start].strip().startswith("FORCES|")
        and not lines[start].strip().startswith("FORCES| Sum")
        and not lines[start].strip().startswith("FORCES| Total atomic force")
    ):
        parts = lines[start].split()
        if len(parts) >= 6:  # FORCES| atom_num fx fy fz norm
            fx = float(parts[2])
            fy = float(parts[3])
            fz = float(parts[4])
            forces.append([fx, fy, fz])
        start += 1
    if not forces:
        raise ValueError("No forces parsed from output")
    return np.array(forces)


def try_parse_existing_out(out_file):
    if not os.path.exists(out_file):
        return None
    try:
        # Parse energy
        energy = None
        with open(out_file, "r") as f:
            for line in f:
                if "ENERGY| Total FORCE_EVAL" in line:
                    energy = float(line.split()[-1])
                    break
        if energy is None:
            return None
        # Parse forces
        forces = parse_forces_from_out(out_file)
        # Convert units
        energy_kcal = energy * HARTREE_TO_KCAL_MOL
        forces_kcal_ang = forces * FORCE_CONVERSION_FACTOR
        return energy_kcal, forces_kcal_ang
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Perform DFT labeling with CP2K")
    parser.add_argument(
        "--iter",
        type=int,
        required=True,
        help="Iteration number for input/output files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="augment",
        choices=["augment", "calib"],
        help="Mode: 'augment' for augmented dataset, 'calib' for calibration frames",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="dft_label.inp",
        help="CP2K input template file",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=4,
        help="Number of MPI processes for CP2K",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Working directory for runs",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="../results",
        help="Directory for base frames",
    )
    args = parser.parse_args()
    iter_num = args.iter
    mode = args.mode
    if mode == "calib":
        input_file = f"calib_frames_iter_{iter_num}.extxyz"
        output_file = f"calib_labeled_iter_{iter_num}.extxyz"
        combined_file = None
        out_dir = f"../logs/calib_label_{iter_num}"
    else:
        input_file = f"augmented_dataset_iter{iter_num}.extxyz"
        output_file = f"augmented_dataset_iter{iter_num}_dft.extxyz"
        combined_file = f"augmented_primary_train_val_iter_{iter_num}.extxyz"
        out_dir = f"../logs/augment_label_{iter_num}"
    os.makedirs(out_dir, exist_ok=True)
    # Read frames
    frames = read(input_file, index=":")
    # Load partial labeled frames if output exists
    labeled_frames = []
    start_idx = 0
    if os.path.exists(output_file):
        labeled_frames = read(output_file, index=":")
        start_idx = len(labeled_frames)
        print(f"Resuming from {start_idx} labeled frames in {output_file}")
    # Label remaining frames
    for i in range(start_idx, len(frames)):
        frame = frames[i]
        label = f"frame_{i}"
        out_file = os.path.join(out_dir, f"{label}.out")
        parsed = try_parse_existing_out(out_file)
        if parsed is not None:
            energy, forces = parsed
            print(f"Using existing complete output for {label}")
        else:
            print(f"Running CP2K for {label} (no/incomplete output)")
            if os.path.exists(out_file):
                os.remove(out_file)  # Remove incomplete file
            energy, forces = run_cp2k_single_point(
                frame,
                args.template,
                label,
                args.n_cores,
                args.work_dir,
                mode=mode,
                iter_num=iter_num,
            )
        new_frame = frame.copy()
        new_frame.info["energy"] = energy
        new_frame.arrays["forces"] = forces
        labeled_frames.append(new_frame)
        # Append to output file incrementally
        write(output_file, [new_frame], append=True if i > start_idx else False)
    print(f"Labeled DFT file saved to {output_file}")
    if combined_file:
        # Load base frames (from previous or primary)
        if iter_num > 0:
            base_file = f"augmented_primary_train_val_iter_{iter_num - 1}.extxyz"
        else:
            base_file = PRIMARY_TRAIN_VAL_FILE
        if os.path.exists(base_file):
            base_frames = read(base_file, index=":")
        else:
            base_frames = []
        # Combine and save
        combined_frames = base_frames + labeled_frames
        write(combined_file, combined_frames)
        print(f"Combined augmented dataset saved to {combined_file}")


if __name__ == "__main__":
    main()
