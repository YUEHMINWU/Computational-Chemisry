import os
import subprocess
import argparse
import sys
from ase.io import read
import re  # For parsing rmse from output
import glob
import time
import signal

# Configuration
NUM_ITERATIONS = 10
RMSE_THRESHOLD = 0.8  # kcal/mol/Å
NUM_INIT_STRUCTS = 3
INIT_STRUCT_DIR = "init_structs"
MD_RESULTS_DIR = "../results"
FINAL_TEST_FILE = "final_test.extxyz"
ALLEGRO_TRAINED_MODEL_DIR_BASE = "../results/allegro_model_output"
CHEMICAL_SYMBOLS = ["Li", "F", "S", "O", "C", "N", "H"]
CUTOFF = 6.0
NUM_ENSEMBLES = 3
TARGET_INITIAL_TEMP_STR = "297.99999999999994316"


def parse_rmse_from_output(output):
    # Parse the printed RMSE from stdout
    match = re.search(r"Force RMSE on test set: (\d+\.\d+)", output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Could not parse RMSE from output.")


def models_already_trained():
    # Check if primary model for iter 0 exists
    primary_path = os.path.join(
        "../results/allegro_model_output_primary_iter_0", "deployed.nequip.pth"
    )
    if not os.path.exists(primary_path):
        return False

    # Check if all ensemble models exist
    for i in range(NUM_ENSEMBLES):
        ensemble_path = os.path.join(
            f"{ALLEGRO_TRAINED_MODEL_DIR_BASE}_{i}", "deployed.nequip.pth"
        )
        if not os.path.exists(ensemble_path):
            return False

    return True


def all_mlpmd_done(current_iter):
    for i in range(NUM_INIT_STRUCTS):
        label = f"iter{current_iter}_struct{i}"
        pos_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-pos.xyz")
        thermo_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-thermo.log")
        if not os.path.exists(pos_file) or not check_initial_temp(thermo_file):
            return False
    return True


def check_initial_temp(thermo_file):
    if not os.path.exists(thermo_file):
        return None
    with open(thermo_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("0 "):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == TARGET_INITIAL_TEMP_STR:
                return True
            else:
                return False
    return None  # Step 0 not yet written


def delete_simulation_files(label):
    pos_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-pos.xyz")
    frc_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-frc.xyz")
    thermo_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-thermo.log")
    out_file = os.path.join("../logs", f"{label}.out")
    in_file = os.path.join("../logs", f"{label}.in")
    # data_file = os.path.join("../data", f"{label}.data")  # Can keep, as it's regenerated
    files = [pos_file, frc_file, thermo_file, out_file, in_file]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")


def main(start_iter=0):
    current_iter = start_iter
    while current_iter < NUM_ITERATIONS:
        print(f"\n--- Iteration {current_iter + 1}/{NUM_ITERATIONS} ---")

        primary_model_dir = (
            f"../results/allegro_model_output_primary_iter_{current_iter}"
        )

        # Step 1: Train models
        if current_iter == 0:
            if models_already_trained():
                print(
                    "Initial primary and ensemble models already trained. Skipping training."
                )
            else:
                # Initial: Train primary + ensembles
                train_cmd = ["python", "train_allegro_model.py", "--ensemble_mode"]
                subprocess.run(
                    train_cmd, check=True, capture_output=False
                )  # Set to False for live debug output
        else:
            if os.path.exists(os.path.join(primary_model_dir, "deployed.nequip.pth")):
                print("Augmented primary model already trained. Skipping training.")
            else:
                # Later: Train only primary on augmented
                augmented_file = (
                    f"augmented_primary_train_val_iter_{current_iter - 1}.extxyz"
                )
                augmented_size = len(read(augmented_file, index=":"))
                val_frames = int(0.1 * augmented_size)
                train_frames = augmented_size - val_frames
                train_cmd = [
                    "python",
                    "train_allegro_model.py",
                    "--train_val_file",
                    augmented_file,
                    "--output_dir",
                    primary_model_dir,
                    "--train_frames",
                    str(train_frames),
                    "--val_frames",
                    str(val_frames),
                ]
                subprocess.run(
                    train_cmd, check=True, capture_output=False
                )  # Set to False for live debug output

        # Step 2: Run MLIP-MD with primary on 3 init structs
        init_files = [
            os.path.join(INIT_STRUCT_DIR, f"init_struct_{i}.extxyz")
            for i in range(NUM_INIT_STRUCTS)
        ]
        all_structs_done = all_mlpmd_done(current_iter)
        if all_structs_done:
            print(
                "All MLIP-MD simulations for this iteration already done. Skipping step 2."
            )
        else:
            for i, init_file in enumerate(init_files):
                label = f"iter{current_iter}_struct{i}"
                pos_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-pos.xyz")
                thermo_file = os.path.join(MD_RESULTS_DIR, f"{label}-md-thermo.log")
                if os.path.exists(pos_file):
                    if check_initial_temp(thermo_file):
                        print(
                            f"MLIP-MD for {label} already run with correct initial temp. Skipping this struct."
                        )
                        continue
                    else:
                        print(
                            f"MLIP-MD for {label} exists but initial temp incorrect. Deleting and rerunning."
                        )
                        delete_simulation_files(label)

                # Run in a loop until success
                success = False
                while not success:
                    mlpmd_cmd = [
                        "conda",
                        "run",
                        "-n",
                        "lammps_mlp",
                        "python",
                        "run_mlpmd.py",
                        "--xyz_file",
                        init_file,
                        "--label",
                        label,
                        "--md_time",
                        "50.0",
                        "--model_file",
                        f"{primary_model_dir}/deployed.nequip.pth",
                    ]

                    # Start the process non-blocking, in a new session for group killing
                    process = subprocess.Popen(
                        mlpmd_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid
                    )

                    # Monitor thermo_file for initial temp
                    initial_temp_correct = False
                    start_time = time.time()
                    while (
                        time.time() - start_time < 120
                    ):  # Increased timeout to 120 seconds
                        if os.path.exists(thermo_file):
                            time.sleep(5)  # Give time for writing
                            status = check_initial_temp(thermo_file)
                            if status is True:
                                initial_temp_correct = True
                                break
                            elif status is False:
                                print(
                                    f"Initial temp for {label} not matching. Killing process and rerunning."
                                )
                                os.killpg(process.pid, signal.SIGTERM)
                                time.sleep(1)  # Give time for termination
                                try:
                                    process.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    os.killpg(process.pid, signal.SIGKILL)
                                delete_simulation_files(label)
                                break
                        time.sleep(1)  # Check every second

                    if not initial_temp_correct:
                        # If timed out without file or status False already handled
                        if process.poll() is None:
                            print(f"Timeout waiting for thermo file for {label}. Killing process and rerunning.")
                            os.killpg(process.pid, signal.SIGTERM)
                            time.sleep(1)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                        delete_simulation_files(label)
                        continue  # Rerun with new seed

                    # Wait for completion
                    try:
                        stdout, stderr = process.communicate()
                        if process.returncode != 0:
                            raise subprocess.CalledProcessError(
                                process.returncode, mlpmd_cmd
                            )
                        print(
                            f"\n✅ LAMMPS simulation for {label} completed successfully."
                        )
                        success = True
                    except (
                        subprocess.CalledProcessError,
                    ) as e:
                        print(
                            f"Simulation for {label} failed (possibly explosion). Killing and rerunning."
                        )
                        os.killpg(process.pid, signal.SIGTERM)
                        time.sleep(1)  # Give time for termination
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        delete_simulation_files(label)
                        # Continue to rerun

        # Step 3: UQ + Augment + Check RMSE with run_final_model.py
        unc_file = f"unc_data_iter_{current_iter}.npy"
        augmented_file = f"augmented_primary_train_val_iter_{current_iter}.extxyz"
        if os.path.exists(augmented_file):
            print("UQ and augmentation already done. Computing RMSE only.")
            rmse_cmd = [
                "python",
                "run_final_model.py",
                "--compute_rmse",
                "--model_dir",
                primary_model_dir,
            ]
            result = subprocess.run(
                rmse_cmd, capture_output=True, text=True, check=True
            )
            rmse = parse_rmse_from_output(result.stdout)
        elif os.path.exists(unc_file):
            print("UQ done. Performing augmentation and computing RMSE.")
            final_cmd = [
                "python",
                "run_final_model.py",
                "--augment_only",
                "--iter",
                str(current_iter),
                "--model_dir",
                primary_model_dir,
            ]
            result = subprocess.run(
                final_cmd, capture_output=True, text=True, check=True
            )
            rmse = parse_rmse_from_output(result.stdout)
        else:
            print("Performing UQ, augmentation, and computing RMSE.")
            final_cmd = [
                "python",
                "run_final_model.py",
                "--traj_dir",
                MD_RESULTS_DIR,
                "--iter",
                str(current_iter),
                "--model_dir",
                primary_model_dir,
            ]
            result = subprocess.run(
                final_cmd, capture_output=True, text=True, check=True
            )
            rmse = parse_rmse_from_output(result.stdout)
        print(f"Current RMSE: {rmse:.4f}")
        if rmse < RMSE_THRESHOLD:
            print("RMSE threshold met. Stopping iterations.")
            break

        current_iter += 1

    print("\nAL loop complete.")

    # After loop: Generate final_parity.npy using the final primary model
    print("\nGenerating final parity data...")
    final_primary_dir = (
        f"../results/allegro_model_output_primary_iter_{current_iter - 1}"
    )
    parity_cmd = [
        "python",
        "run_final_model.py",
        "--save_parity",
        "--model_dir",
        final_primary_dir,
    ]
    subprocess.run(parity_cmd, check=True, capture_output=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AL iteration script")
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
        help="Starting iteration number",
    )
    args = parser.parse_args()
    main(args.start_iter)
