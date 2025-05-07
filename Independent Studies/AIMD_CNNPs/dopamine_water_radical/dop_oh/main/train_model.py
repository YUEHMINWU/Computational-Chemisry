import numpy as np
from aml import Structures, QbC, acsf
from ase.io import Trajectory
import os

def read_ase_trajectory(traj_file):
    """Read ASE trajectory and convert to AML Structures."""
    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"Trajectory file {traj_file} not found")
    traj = Trajectory(traj_file)
    structures = Structures()
    for frame in traj:
        structures.add(frame.get_positions(), frame.get_forces(), frame.get_potential_energy(), 
                       frame.get_cell(), frame.get_pbc(), frame.get_atomic_numbers())
    return structures

def run_qbc(structures, dir_output, n_train_initial, n_add, n_iterations, fn_results, fn_restart):
    """Run Query by Committee (QbC) to select training structures."""
    qbc = QbC(n_train_initial=n_train_initial, n_add=n_add, n_iterations=n_iterations)
    qbc.run(structures, dir_output=dir_output, fn_results=fn_results, fn_restart=fn_restart)
    return qbc

def train_cnnp(structures_train, dir_training, elements, n, fn_template, exclude_triples, n_tasks, n_core_task):
    """Train C-NNP model using N2P2."""
    from aml import N2P2
    kwargs_model = dict(
        elements=elements,
        n=n,
        fn_template=fn_template,
        exclude_triples=exclude_triples,
        n_tasks=n_tasks,
        n_core_task=n_core_task,
        remove_output=True
    )
    n2p2 = N2P2(**kwargs_model)
    n2p2.train(structures_train, dir_training=dir_training)
    return n2p2

def main(traj_file='dopamine_water_hydroxyl_aimd.traj'):
    """Main function to orchestrate the workflow."""
    np.random.seed(42)  # For reproducibility
    
    # Step 1: Read trajectory
    structures = read_ase_trajectory(traj_file)
    
    # Step 2: Generate input.nn for ACSFs
    elements = ('C', 'H', 'O', 'N')
    input_nn_content = acsf.generate_radial_angular_default(elements=elements)
    with open('input.nn', 'w') as f:
        f.write(input_nn_content)
    
    # Step 3: Run QbC
    dir_output = 'dopamine_water_hydroxyl_qbc'
    fn_results = 'dopamine_water_hydroxyl_qbc_results.json'
    fn_restart = 'dopamine_water_hydroxyl_qbc_restart.json'
    n_train_initial = 20
    n_add = 20
    n_iterations = 10
    qbc = run_qbc(structures, dir_output, n_train_initial, n_add, n_iterations, fn_results, fn_restart)
    structures_train = qbc.get_structures_train()
    
    # Step 4: Train C-NNP
    dir_training = 'dopamine_water_hydroxyl_final-training'
    n = 8
    fn_template = 'input.nn'
    exclude_triples = [['O', 'O', 'O'], ['N', 'N', 'N']]
    n_tasks = 4
    n_core_task = 4
    train_cnnp(structures_train, dir_training, elements, n, fn_template, exclude_triples, n_tasks, n_core_task)

if __name__ == '__main__':
    main()