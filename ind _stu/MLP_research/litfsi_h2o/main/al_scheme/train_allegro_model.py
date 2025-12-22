import os
import subprocess
import glob
import yaml
import sys
import argparse  # Added for command-line arguments
from ase.io import read

# ==============================================================================
# Configuration Parameters
# ==============================================================================
ALLEGRO_TRAINED_MODEL_DIR_BASE = "../results/allegro_model_output"
ALLEGRO_DEPLOYED_MODEL_NAME = "deployed.nequip.pth"

# --- Training Hyperparameters ---
SYSTEM_ELEMENTS = ["Li", "F", "S", "O", "C", "N", "H"]  # LiTFSI water system
CUTOFF_RADIUS = 6.0
MAX_EPOCHS = 200
FORCE_COEFF = 1.0
BATCH_SIZE = 5
NUM_WORKERS = 5
NUM_ENSEMBLES = 5  # Number of ensemble models to train


def prepare_allegro_config(
    output_dir,
    data_file,
    elements,
    cutoff_radius,
    max_epochs,
    force_coeff,
    batch_size,
    train_frames,
    val_frames,
):
    """
    Generates the YAML configuration for Allegro training.
    """
    print("--- Step 1: Preparing Allegro config ---")
    chemical_symbols_yaml = "[" + ", ".join(f'"{e}"' for e in elements) + "]"
    data_file_path = os.path.abspath(data_file)

    yaml_content = f"""
run: [train]

cutoff_radius: {cutoff_radius}
chemical_symbols: {chemical_symbols_yaml}
model_type_names: ${{chemical_symbols}}

data:
  _target_: nequip.data.datamodule.ASEDataModule
  seed: 855105
  split_dataset:
    file_path: {data_file_path}
    train: {train_frames}
    val: {val_frames}
  transforms:
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      chemical_symbols: ${{chemical_symbols}}
    - _target_: nequip.data.transforms.NeighborListTransform
      r_max: ${{cutoff_radius}}
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 1
    shuffle: true
    num_workers: {NUM_WORKERS}
  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 2
    num_workers: {NUM_WORKERS}
    persistent_workers: true
  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${{model_type_names}}

monitored_metric: val0_epoch/weighted_sum

trainer:
  _target_: lightning.Trainer
  accelerator: cpu
  devices: auto
  max_epochs: {max_epochs}
  max_time: 07:00:00:00
  check_val_every_n_epoch: 1
  log_every_n_steps: 500
  enable_progress_bar: false
  callbacks:
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: ${{monitored_metric}}
      patience: 20
      min_delta: 1e-4
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: ${{monitored_metric}}
      filename: best
      save_last: true

num_scalar_features: 64

training_module:
  _target_: nequip.train.EMALightningModule
  ema_decay: 0.99
  loss:
    _target_: nequip.train.EnergyForceLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 1.0
  val_metrics:
    _target_: nequip.train.EnergyForceMetrics
    coeffs:
      total_energy_rmse: 1.0
      forces_rmse: 1.0
      total_energy_mae: 1.0
      forces_mae: 1.0
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  lr_scheduler:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      factor: 0.8
      patience: 10
      min_lr: 1e-5
    monitor: ${{monitored_metric}}
    interval: epoch
    frequency: 1
  model:
    _target_: allegro.model.AllegroModel
    seed: 152621
    model_dtype: float32
    type_names: ${{model_type_names}}
    r_max: ${{cutoff_radius}}
    radial_chemical_embed:
      _target_: allegro.nn.TwoBodyBesselScalarEmbed
      num_bessels: 8
      bessel_trainable: true
      polynomial_cutoff_p: 6
    radial_chemical_embed_dim: ${{num_scalar_features}}
    scalar_embed_mlp_hidden_layers_depth: 4
    scalar_embed_mlp_hidden_layers_width: 256
    scalar_embed_mlp_nonlinearity: silu
    l_max: 2
    num_layers: 2
    num_scalar_features: 64
    num_tensor_features: 16
    allegro_mlp_hidden_layers_depth: 3
    allegro_mlp_hidden_layers_width: 256
    allegro_mlp_nonlinearity: silu
    parity: true
    tp_path_channel_coupling: true
    readout_mlp_hidden_layers_depth: 1
    readout_mlp_hidden_layers_width: 128
    readout_mlp_nonlinearity: null
    avg_num_neighbors: ${{training_data_stats:num_neighbors_mean}}
    per_type_energy_shifts: ${{training_data_stats:per_atom_energy_mean}}
    per_type_energy_scales: ${{training_data_stats:forces_rms}}
    per_type_energy_scales_trainable: false
    per_type_energy_shifts_trainable: false
    pair_potential:
      _target_: nequip.nn.pair_potential.ZBL
      units: real
      chemical_species: ${{chemical_symbols}}

global_options:
  allow_tf32: false
"""

    output_config_path = os.path.join(output_dir, "allegro_config.yaml")
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)

    with open(output_config_path, "w") as f:
        f.write(yaml_content)
    print(f"Config saved to {os.path.abspath(output_config_path)}")

    with open(output_config_path, "r") as f:
        try:
            yaml.safe_load(f)
            print("YAML file is valid.")
        except yaml.YAMLError as e:
            print(f"YAML error: {e}")
            sys.exit(1)

    return output_config_path


def train_allegro_model(config_path, output_dir):
    """
    Trains the Allegro model using the generated config and deploys it.
    """
    print("--- Step 2: Training and Deploying Allegro model ---")
    try:
        abs_config_path = os.path.abspath(config_path)

        if not os.path.exists(abs_config_path):
            print(f"Error: Config file '{abs_config_path}' not found.")
            sys.exit(1)

        config_dir = os.path.dirname(abs_config_path)
        config_name = os.path.basename(abs_config_path)

        command = [
            "nequip-train",
            "--config-path",
            config_dir,
            "--config-name",
            config_name,
        ]

        print(f"Executing: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Training STDOUT:", result.stdout)
        if result.stderr:
            print("Training STDERR:", result.stderr)

        output_dir_logs = os.path.join(
            os.getcwd(), "lightning_logs", "*", "checkpoints"
        )
        checkpoint_pattern = os.path.join(
            os.getcwd(), "lightning_logs", "*", "checkpoints", "best.ckpt"
        )
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)

        if not checkpoint_files:
            print(
                f"Error: No 'best.ckpt' file found in the '{output_dir_logs}' directory."
            )
            sys.exit(1)

        best_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Found most recent checkpoint: {best_checkpoint}")

        os.makedirs(output_dir, exist_ok=True)
        deployed_model_path = os.path.join(output_dir, ALLEGRO_DEPLOYED_MODEL_NAME)

        deploy_command = [
            "nequip-compile",
            "--mode",
            "torchscript",
            "--device",
            "cpu",
            best_checkpoint,
            deployed_model_path,
        ]
        print(f"Executing: {' '.join(deploy_command)}")
        deploy_result = subprocess.run(
            deploy_command, capture_output=True, text=True, check=True
        )
        print("Deployment STDOUT:", deploy_result.stdout)
        if deploy_result.stderr:
            print("Deployment STDERR:", deploy_result.stderr)

        print(f"\nModel successfully trained and deployed to: {deployed_model_path}")
        return deployed_model_path

    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{' '.join(e.cmd)}' failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(
            f"Error: Command not found: '{e.filename}'. Ensure NequIP is installed and in your system's PATH."
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Allegro model with custom parameters."
    )
    parser.add_argument(
        "--train_val_file", type=str, help="Path to train_val data file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for trained model"
    )
    parser.add_argument("--train_frames", type=int, help="Number of training frames")
    parser.add_argument("--val_frames", type=int, help="Number of validation frames")
    parser.add_argument("--ensemble_mode", action="store_true", help="Train ensembles")
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        help="Path to augmentation file for iterative training",
    )

    args = parser.parse_args()

    if args.ensemble_mode:
        # Ensemble mode: Train ensembles (primary handled separately)
        # Ensembles (always use their fixed files and 450/50)
        for ensemble_id in range(NUM_ENSEMBLES):
            ensemble_dir = f"{ALLEGRO_TRAINED_MODEL_DIR_BASE}_{ensemble_id}"
            ensemble_deployed = os.path.join(ensemble_dir, "deployed.nequip.pth")
            if os.path.exists(ensemble_deployed):
                print(f"Ensemble model {ensemble_id} already trained. Skipping.")
            else:
                print(f"\nTraining ensemble model {ensemble_id}")
                ensemble_file = (
                    f"aimd_trajectory_ensemble_{ensemble_id}_train_val.extxyz"
                )
                if not os.path.exists(ensemble_file):
                    print(f"Ensemble file {ensemble_file} not found.")
                    continue
                size = len(read(ensemble_file, index=":"))
                train_frames_ensemble = int(0.9 * size)
                val_frames_ensemble = size - train_frames_ensemble
                ensemble_config_path = prepare_allegro_config(
                    ensemble_dir,
                    ensemble_file,
                    SYSTEM_ELEMENTS,
                    CUTOFF_RADIUS,
                    MAX_EPOCHS,
                    FORCE_COEFF,
                    BATCH_SIZE,
                    train_frames_ensemble,
                    val_frames_ensemble,
                )
                ensemble_model_path = train_allegro_model(
                    ensemble_config_path, ensemble_dir
                )

        print("\n" + "=" * 50)
        print("✅ All ensemble trainings complete.")
        print("=" * 50)
    else:
        # Single model mode (for primary models)
        if (
            args.train_val_file
            and args.output_dir
            and args.train_frames is not None
            and args.val_frames is not None
        ):
            os.makedirs(args.output_dir, exist_ok=True)

            # Prepare config
            allegro_config_path = prepare_allegro_config(
                args.output_dir,
                args.train_val_file,
                SYSTEM_ELEMENTS,
                CUTOFF_RADIUS,
                MAX_EPOCHS,
                FORCE_COEFF,
                BATCH_SIZE,
                args.train_frames,
                args.val_frames,
            )

            # Train and deploy
            final_model_path = train_allegro_model(allegro_config_path, args.output_dir)

            print("\n" + "=" * 50)
            print("✅ Training and deployment complete.")
            print(f"  -> Deployed Model: {final_model_path}")
            print("=" * 50)
        else:
            print("Error: Missing required arguments for single model training.")
            sys.exit(1)
