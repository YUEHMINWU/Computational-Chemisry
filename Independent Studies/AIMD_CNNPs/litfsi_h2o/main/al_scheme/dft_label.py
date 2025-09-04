import os
import subprocess
import shutil
import argparse
import sys
from ase.io import read, write
from ase import Atoms
from tqdm import tqdm
import numpy as np

HARTREE_TO_KCAL_MOL = 627.50960803
BOHR_TO_ANGSTROM = 0.5291772109
FORCE_CONVERSION_FACTOR = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM
PRIMARY_TRAIN_VAL_FILE = "aimd_trajectory_primary_train_val.extxyz"

def run_cp2k_single_point(frame, template_inp="dft_label.inp", label="frame", n_cores=1, work_dir="."):
    temp_dir = os.path.join(work_dir, f"cp2k_run_{label}")
    os.makedirs(temp_dir, exist_ok=True)
    # Write coord to XYZ
    coord_file = os.path.join(temp_dir, "input.xyz")
    write(coord_file, frame, format='xyz')
    # Copy and update template inp
    shutil.copy(template_inp, temp_dir)
    inp_file = os.path.join(temp_dir, "dft_label.inp")
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    updated_lines = []
    for line in lines:
        if 'COORD_FILE_NAME' in line:
            line = f"      COORD_FILE_NAME {os.path.abspath(coord_file)}\n"
        updated_lines.append(line)
    with open(inp_file, 'w') as f:
        f.write(''.join(updated_lines))
    # Copy DFT-D3 file
    dftd3_src = "/opt/homebrew/share/cp2k/data/dftd3.dat"
    dftd3_dst = os.path.join(temp_dir, "dftd3.dat")
    if os.path.exists(dftd3_src):
        shutil.copyfile(dftd3_src, dftd3_dst)
    else:
        raise FileNotFoundError(f"DFT-D3 parameter file not found: {dftd3_src}")
    # Run CP2K
    cp2k_psmp = "/opt/homebrew/bin/cp2k.psmp"
    out_file = os.path.join(temp_dir, f"{label}.out")
    cmd = [
        "mpirun",
        "-np",
        str(n_cores),
        cp2k_psmp,
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
    with open(out_file, 'r') as f:
        for line in f:
            if "ENERGY| Total FORCE_EVAL" in line:
                energy = float(line.split()[-1])
                break
    if energy is None:
        raise ValueError(f"Energy not found in {out_file}")
    # Convert energy to kcal/mol
    energy *= HARTREE_TO_KCAL_MOL
    # Parse forces from dft_label-frc-1.xyz (CP2K default for forces print)
    forces_file = os.path.join(temp_dir, "dft_label-frc-1.xyz")
    if not os.path.exists(forces_file):
        raise FileNotFoundError(f"Forces file not found: {forces_file}")
    forces_atoms = read(forces_file)
    forces = forces_atoms.get_forces()  # ASE parses forces if in extended XYZ format; adjust if needed
    # Convert forces to kcal/mol/Å
    forces *= FORCE_CONVERSION_FACTOR
    # Clean temp dir (optional)
    # shutil.rmtree(temp_dir)
    return energy, forces

def main():
    parser = argparse.ArgumentParser(description="Perform DFT labeling with CP2K")
    parser.add_argument(
        "--iter",
        type=int,
        required=True,
        help="Iteration number for input/output files",
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
        default=1,
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
    input_file = f"augmented_dataset_iter{iter_num}.extxyz"
    output_file = f"augmented_dataset_iter{iter_num}_dft.extxyz"
    combined_file = f"augmented_primary_train_val_iter_{iter_num}.extxyz"
    # Read high-unc frames
    frames = read(input_file, index=":")
    # Label each frame
    labeled_frames = []
    for i, frame in enumerate(tqdm(frames, desc="DFT labeling frames")):
        energy, forces = run_cp2k_single_point(frame, args.template, f"frame_{i}", args.n_cores, args.work_dir)
        new_frame = frame.copy()
        new_frame.info['energy'] = energy
        new_frame.arrays['forces'] = forces
        labeled_frames.append(new_frame)
    # Write labeled file
    write(output_file, labeled_frames)
    print(f"Labeled DFT file saved to {output_file}")
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
