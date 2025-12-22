#!/usr/bin/env python

from pathlib import Path
import numpy as np
import aml
import aml.acsf as acsf
from ase.io import Trajectory

def read_ase_trajectory(traj_file, stride=1):
    """Read ASE trajectory and convert to AML Structures."""
    print(f'Reading trajectory: {traj_file}')
    traj = Trajectory(traj_file)
    frames = []
    for frame in traj[::stride]:
        # Extract positions, forces, and energy
        positions = frame.get_positions()
        forces = frame.get_forces()
        energy = frame.get_potential_energy()
        cell = frame.get_cell()
        pbc = frame.get_pbc()
        frames.append(aml.Frame(
            positions=positions,
            forces=forces,
            energy=energy,
            cell=cell,
            pbc=pbc,
            numbers=frame.get_atomic_numbers()
        ))
    structures = aml.Structures.from_frames(frames, probability=1.0)
    print(f'{len(structures)} structures loaded with stride {stride}')
    return structures

def run_qbc(structures, dir_output='qbc', n_train_initial=20, n_add=20, n_iterations=10):
    """Run Query by Committee to select training structures."""
    dir_output = Path(dir_output)
    dir_output.mkdir(exist_ok=True)
    
    kwargs_model = dict(
        elements=('C', 'H', 'O', 'N'),
        n=8,  # Number of committee members
        fn_template=str(dir_output / 'input.nn'),
        exclude_triples=[['O', 'O', 'O'], ['N', 'N', 'N']],  # Avoid redundant triples
        n_tasks=4,  # Parallel tasks for NUC's 16 cores
        n_core_task=4,
        remove_output=True
    )
    
    # Generate default ACSFs
    radials, angulars = acsf.generate_radial_angular_default()
    
    # Format ACSFs for input.nn
    acsf_str = acsf.format_combine_ACSFs(
        radials,
        angulars,
        elements=['C', 'H', 'O', 'N'],
        exclude_triples=[['O', 'O', 'O'], ['N', 'N', 'N']]
    )
    
    # Write input.nn file
    input_nn_content = f"""
    number_of_elements 4
    elements C H O N
    cutoff_type 2
    scale_symmetry_functions
    scale_min_short 0.0
    scale_max_short 1.0
    center_symmetry_functions
    global_hidden_layers_short 2
    global_nodes_short 20 20
    global_activation_short t t l
    use_short_forces
    random_seed 42
    epochs 15
    updater_type 1
    parallel_mode 1
    jacobian_mode 2
    update_strategy 0
    selection_mode 2
    task_batch_size_energy 1
    task_batch_size_force 1
    memorize_symfunc_results
    test_fraction 0.1
    force_weight 2.0
    short_energy_fraction 1.0
    short_force_fraction 0.05
    short_energy_error_threshold 0.8
    short_force_error_threshold 1.0
    rmse_threshold_trials 3
    weights_min -1.0
    weights_max 1.0
    nguyen_widrow_weights_short
    kalman_type 0
    kalman_epsilon 1.0E-2
    kalman_q0 0.01
    kalman_qtau 2.302
    kalman_qmin 1.0E-6
    kalman_eta 0.01
    kalman_etatau 2.302
    kalman_etamax 1.0
    # Symmetry functions
    {acsf_str}
    """
    with open(dir_output / 'input.nn', 'w') as f:
        f.write(input_nn_content)
    
    qbc = aml.QbC(
        structures=structures,
        cls_model=aml.N2P2,
        kwargs_model=kwargs_model,
        n_train_initial=n_train_initial,
        n_add=n_add,
        n_epoch=15,
        n_iterations=n_iterations,
        n_candidate=1000,  # Reduced for liquid system
        fn_results=str(dir_output / 'results.shelf'),
        fn_restart=None
    )
    
    print('Running QbC...')
    qbc.run()
    
    # Load final training set
    final_train_dir = dir_output / f'iteration-{n_iterations:03d}' / 'train-000'
    structures_train = aml.Structures.from_file(str(final_train_dir / 'input.data'))
    print(f'Loaded {len(structures_train)} training structures from QbC')
    return structures_train, dir_output

def train_cnnp(structures_train, dir_output='qbc', dir_training='final-training'):
    """Train C-NNP model using selected structures."""
    dir_training = Path(dir_training)
    dir_training.mkdir(exist_ok=True)
    
    kwargs_model = dict(
        elements=('C', 'H', 'O', 'N'),
        n=8,
        fn_template=str(dir_output / 'input.nn'),
        exclude_triples=[['O', 'O', 'O'], ['N', 'N', 'N']],
        n_tasks=4,
        n_core_task=4
    )
    
    n2p2 = aml.N2P2(dir_run=dir_training, **kwargs_model)
    print('Training C-NNP...')
    n2p2.train(structures_train, n_epoch=50)
    
    print('Saving C-NNP model...')
    n2p2.save_model()
    print(f'Model saved in {dir_training}')
    return n2p2

def main(traj_file='dopamine_aimd.traj'):
    """Main function to train C-NNP model."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Read AIMD trajectory
    structures = read_ase_trajectory(traj_file, stride=10)
    
    # Run QbC to select training structures
    structures_train, dir_output = run_qbc(structures)
    
    # Train and save C-NNP
    n2p2 = train_cnnp(structures_train, dir_output)
    
    return n2p2

if __name__ == "__main__":
    main()