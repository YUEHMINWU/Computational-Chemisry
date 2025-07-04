import numpy as np
import os
from ase.io import read, write as ase_write
from ase import Atoms
from io import StringIO
import sys
from concurrent.futures import ProcessPoolExecutor

# Constants
HARTREE_TO_EV = 27.2113862459
BOHR_TO_ANGSTROM = 0.5291772109
FORCE_CONVERSION_FACTOR = 51.4220861906
CELL_EDGE_LENGTH = 14.936
CUBIC_CELL = [CELL_EDGE_LENGTH] * 3


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
            # Extract step number from comment, e.g., "i = 1734, time = ..."
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
    """Process a chunk of frame data into Atoms objects, skipping invalid frames."""
    processed_atoms = []
    for pos_string, force_string, energy_hartree, step in frame_data:
        try:
            # Parse position data
            pos_atoms = read(StringIO(pos_string), format="xyz")
            energy_ev = float(energy_hartree) * HARTREE_TO_EV

            # Parse force data
            force_lines = force_string.strip().split("\n")
            n_atoms = int(force_lines[0].strip())
            if len(pos_atoms) != n_atoms:
                raise ValueError(
                    f"Atom count mismatch: pos={len(pos_atoms)}, force={n_atoms}"
                )
            if n_atoms != 244:  # Adjust based on your system
                raise ValueError(f"Unexpected atom count {n_atoms} at step {step}")

            force_data = []
            for line in force_lines[2 : 2 + n_atoms]:
                parts = line.split()
                # ⬇️ *** CORRECTED LINE *** ⬇️
                # Changed the column check from 7 to 4 to match the input file format.
                if len(parts) < 4:
                    raise ValueError(
                        f"Line has {len(parts)} columns, expected at least 4: {line}"
                    )
                try:
                    # The logic to grab the last 3 elements is correct and needs no change.
                    force_components = [float(x) for x in parts[-3:]]
                except ValueError as e:
                    raise ValueError(
                        f"Non-numeric force components: {parts[-3:]} - {e}"
                    )
                force_data.append(force_components)

            forces_hartree_bohr = np.array(force_data, dtype=np.float32)
            forces_ev_angstrom = forces_hartree_bohr * FORCE_CONVERSION_FACTOR

            # Create Atoms object
            atoms = Atoms(
                symbols=pos_atoms.symbols,
                positions=pos_atoms.positions.astype(np.float32),
            )
            atoms.set_cell(CUBIC_CELL)
            atoms.arrays["forces"] = forces_ev_angstrom
            atoms.info["energy"] = float(energy_ev)
            atoms.info["step"] = int(step)
            atoms.set_pbc(True)
            processed_atoms.append(atoms)
        except Exception as e:
            print(f"Error processing frame at step {step}: {e}")
            continue
    return processed_atoms


def convert_cp2k_to_traj(run_data_list, output_traj_file, total_frames_to_use):
    """Combine CP2K runs into a single trajectory file."""
    print(f"--- Step 1: Combining {len(run_data_list)} CP2K AIMD runs ---")

    combined_data = {}
    for i, run_info in enumerate(run_data_list):
        pos_file, force_file, energy_file = (
            run_info["pos"],
            run_info["frc"],
            run_info["ener"],
        )
        # Corrected typo from "start seeming" to "start_step"
        start_step, end_step = run_info.get("start_step"), run_info.get("end_step")

        print(f" > Processing run {i + 1}: {os.path.basename(pos_file)}")

        try:
            pos_frames = read_xyz_frames(pos_file)
            force_frames = read_xyz_frames(force_file)
            energies_data = np.loadtxt(energy_file, skiprows=1, dtype=np.float32)
            energies = {int(row[0]): float(row[4]) for row in energies_data}

            for step in energies:
                if (start_step is None or step >= start_step) and (
                    end_step is None or step <= end_step
                ):
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

    if not combined_data:
        print("Error: No valid frames found.", file=sys.stderr)
        sys.exit(1)

    # Sample frames
    frame_data = list(combined_data.values())
    np.random.seed(123)
    indices = np.random.choice(
        len(frame_data), min(len(frame_data), total_frames_to_use), replace=False
    )
    sampled_frame_data = [frame_data[i] for i in sorted(indices)]

    # Process in parallel
    # Replaced np.array_split with a standard list comprehension for chunking
    num_procs = min(len(sampled_frame_data), 9)
    if num_procs > 0:
        chunk_size = (
            len(sampled_frame_data) + num_procs - 1
        ) // num_procs  # Ensure all frames are processed
        chunks = [
            sampled_frame_data[i : i + chunk_size]
            for i in range(0, len(sampled_frame_data), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = executor.map(process_trajectory_chunk, chunks)
        all_atoms = [atom for sublist in results for atom in sublist]
    else:
        all_atoms = []

    if not all_atoms:
        print("Error: No frames were processed successfully.", file=sys.stderr)
        sys.exit(1)

    ase_write(output_traj_file, all_atoms, format="extxyz")
    print(f"Successfully wrote {len(all_atoms)} frames to {output_traj_file}")
