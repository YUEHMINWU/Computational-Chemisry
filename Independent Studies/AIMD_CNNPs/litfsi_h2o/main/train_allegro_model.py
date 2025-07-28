import os
import subprocess
import glob
import yaml
import sys
from trajectory_utils import convert_cp2k_to_traj

# Configuration Parameters
CP2K_RUNS = [
    {
        "pos": "../results/litfsi_h2o_relax-pos.xyz",
        "frc": "../results/litfsi_h2o_relax-frc.xyz",
        "ener": "../main/litfsi_h2o_relax-1.ener",
        "start_step": 0,
        "end_step": 7500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re7500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re7500-frc.xyz",
        "ener": "../main/litfsi_h2o_prod_re7500-1.ener",
        "start_step": 7501,
        "end_step": 20500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re20500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re20500-frc.xyz",
        "ener": "../main/litfsi_h2o_prod_re20500-1.ener",
        "start_step": 20501,
        "end_step": 60000,
    },
]
TRAJ_DATA_FILE = "aimd_trajectory.extxyz"
ALLEGRO_TRAINED_MODEL_DIR = "../results/allegro_model_output"
ALLEGRO_DEPLOYED_MODEL_NAME = "deployed.nequip.pth"
TOTAL_FRAMES_TO_USE = 10000
SYSTEM_ELEMENTS = ["Li", "F", "S", "O", "C", "N", "H"]
CUTOFF_RADIUS = 6.0
MAX_EPOCHS = 1000
FORCE_COEFF = 1.0
BATCH_SIZE = 5
NUM_WORKERS = 9


def prepare_allegro_config(
    output_dir,
    data_file,
    elements,
    cutoff_radius,
    max_epochs,
    force_coeff,
    batch_size,
):
    print("--- Step 2: Preparing Allegro config ---")
    chemical_symbols_yaml = "[" + ", ".join(f'"{e}"' for e in elements) + "]"
    data_file_path = os.path.abspath(data_file)

    yaml_content = f"""
run: [train, test]

cutoff_radius: {cutoff_radius}
chemical_symbols: {chemical_symbols_yaml}
model_type_names: ${{chemical_symbols}}

data:
  _target_: nequip.data.datamodule.ASEDataModule
  seed: 123
  split_dataset:
    file_path: {data_file_path}
    train: 10000
    val: 1000
  transforms:
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      chemical_symbols: ${{chemical_symbols}}
    - _target_: nequip.data.transforms.NeighborListTransform
      r_max: ${{cutoff_radius}}
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 1
    shuffle: true
    num_workers: 5
  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 5
    num_workers: 5
    persistent_workers: true
  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${{model_type_names}}

trainer:
  _target_: lightning.Trainer
  accelerator: cpu
  devices: auto
  max_epochs: {max_epochs}
  check_val_every_n_epoch: 1
  log_every_n_steps: 5
  enable_progress_bar: true
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: ${{hydra:runtime.output_dir}}
      save_last: true

num_scalar_features: 64

training_module:
  _target_: nequip.train.EMALightningModule
  loss:
    _target_: nequip.train.EnergyForceLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: {force_coeff}
  val_metrics:
    _target_: nequip.train.EnergyForceMetrics
    coeffs:
      per_atom_energy_mae: 1.0
      forces_mae: 1.0
  test_metrics: ${{training_module.val_metrics}}
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  model:
    _target_: allegro.model.AllegroModel
    seed: 456
    model_dtype: float32
    type_names: ${{model_type_names}}
    r_max: ${{cutoff_radius}}
    radial_chemical_embed:
      _target_: allegro.nn.TwoBodyBesselScalarEmbed
      num_bessels: 8
      bessel_trainable: false
      polynomial_cutoff_p: 6
    radial_chemical_embed_dim: ${{num_scalar_features}}
    scalar_embed_mlp_hidden_layers_depth: 2
    scalar_embed_mlp_hidden_layers_width: 64
    scalar_embed_mlp_nonlinearity: silu
    l_max: 1
    num_layers: 2
    num_scalar_features: 64
    num_tensor_features: 32
    allegro_mlp_hidden_layers_depth: 2
    allegro_mlp_hidden_layers_width: 64
    allegro_mlp_nonlinearity: silu
    parity: true
    tp_path_channel_coupling: true
    readout_mlp_hidden_layers_depth: 1
    readout_mlp_hidden_layers_width: 32
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
    output_config_path = os.path.abspath(output_config_path)

    with open(output_config_path, "w") as f:
        f.write(yaml_content)
    print(f"Config saved to {output_config_path}")

    with open(output_config_path, "r") as f:
        try:
            yaml.safe_load(f)
            print("YAML file is valid.")
        except yaml.YAMLError as e:
            print(f"YAML error: {e}")
            sys.exit(1)

    return output_config_path


def train_allegro_model(config_path):
    print(f"--- Step 3: Training Allegro model with config: {config_path} ---")
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        if not os.path.exists(config_path):
            print(f"Error: Config file {config_path} not found.")
            sys.exit(1)

        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path)
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

        # Locate the last.ckpt file
        checkpoint_pattern = os.path.join(os.getcwd(), "outputs", "*", "*", "last.ckpt")
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)

        if not checkpoint_files:
            print(f"Error: No last.ckpt file found in {checkpoint_pattern}.")
            sys.exit(1)

        # Use the most recent last.ckpt file
        last_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        deployed_model_path = os.path.join(
            ALLEGRO_TRAINED_MODEL_DIR, ALLEGRO_DEPLOYED_MODEL_NAME
        )

        print(f"Using checkpoint: {last_checkpoint}")

        deploy_command = [
            "nequip-compile",
            "--mode",
            "torchscript",
            "--device",
            "cpu",
            last_checkpoint,
            deployed_model_path,
        ]
        print(f"Executing: {' '.join(deploy_command)}")
        deploy_result = subprocess.run(
            deploy_command, capture_output=True, text=True, check=True
        )
        print("Deployment STDOUT:", deploy_result.stdout)
        if deploy_result.stderr:
            print("Deployment STDERR:", deploy_result.stderr)
        print(f"Model deployed to: {deployed_model_path}")

        return deployed_model_path

    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{' '.join(e.cmd)}' failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: 'nequip-train' or 'nequip-compile' not found. Ensure NequIP is installed."
        )
        sys.exit(1)


def deploy_allegro_model(trained_model_dir, deployed_model_name):
    print(f"--- Step 4: Deployed model at {deployed_model_name} ---")
    return os.path.join(trained_model_dir, deployed_model_name)


if __name__ == "__main__":
    os.makedirs(ALLEGRO_TRAINED_MODEL_DIR, exist_ok=True)

    convert_cp2k_to_traj(
        CP2K_RUNS,
        TRAJ_DATA_FILE,
        TOTAL_FRAMES_TO_USE,
    )

    allegro_config_path = prepare_allegro_config(
        ALLEGRO_TRAINED_MODEL_DIR,
        TRAJ_DATA_FILE,
        SYSTEM_ELEMENTS,
        CUTOFF_RADIUS,
        MAX_EPOCHS,
        FORCE_COEFF,
        BATCH_SIZE,
    )

    final_model_path = train_allegro_model(allegro_config_path)

    final_model_path = deploy_allegro_model(
        ALLEGRO_TRAINED_MODEL_DIR, ALLEGRO_DEPLOYED_MODEL_NAME
    )

    print(f"\nTraining and deployment complete. Model saved to: {final_model_path}")
