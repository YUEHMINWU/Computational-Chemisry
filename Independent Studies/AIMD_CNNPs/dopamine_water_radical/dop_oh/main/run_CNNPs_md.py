import numpy as np
from ase.io import read, Trajectory
from ase.calculators.cp2k import CP2K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase import units
import os
from pathlib import Path

def read_system(pdb_file, cell=[11.08, 11.99, 13.04]):
    """Read and prepare the molecular system from a PDB file."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file {pdb_file} not found")
    system = read(pdb_file, format='proteins')
    expected_atoms = 180  # Dopamine (22) + 52 waters (156) + OH• (2)
    if len(system) != expected_atoms:
        raise ValueError(f"Expected {expected_atoms} atoms but found {len(system)}")
    system.set_cell(cell)
    system.center()
    system.set_pbc(True)
    return system

def run_cnnp_md(pdb_file, model_dir='final-training', label='dopamine_oh', steps=200000,
                atom1=15, atom2=179, x0_list=np.arange(1.5, 5.1, 0.5), k_spring=0.015):
    """Run C-NNP MD simulation with umbrella sampling."""
    system = read_system(pdb_file)
    temperature = 298
    
    trajectories = []
    for i, x0 in enumerate(x0_list):
        dir_name = f"umbrella_{x0:.1f}"
        os.makedirs(dir_name, exist_ok=True)
        
        cp2k_input = f"""
        &GLOBAL
          PROJECT {label}_umbrella_{x0:.1f}
          RUN_TYPE MD
        &END GLOBAL
        
        &FORCE_EVAL
          METHOD FIST
          &MM
            &FORCEFIELD
              PAR_TYPE_MIXING GEOMETRIC
              NONBONDED {model_dir}/scaling.data
              &NNP
                NNPS {model_dir}/nnp-000.nn
                NNPS {model_dir}/nnp-001.nn
                NNPS {model_dir}/nnp-002.nn
                NNPS {model_dir}/nnp-003.nn
                NNPS {model_dir}/nnp-004.nn
                NNPS {model_dir}/nnp-005.nn
                NNPS {model_dir}/nnp-006.nn
                NNPS {model_dir}/nnp-007.nn
              &END NNP
            &END FORCEFIELD
          &END MM
          &SUBSYS
            &CELL
              ABC {system.cell[0,0]} {system.cell[1,1]} {system.cell[2,2]}
            &END CELL
            &TOPOLOGY
              COORD_FILE_NAME {pdb_file}
              COORD_FILE_FORMAT PDB
            &END TOPOLOGY
          &END SUBSYS
        &END FORCE_EVAL
        
        &MOTION
          &MD
            ENSEMBLE NVT
            TIMESTEP 0.5
            STEPS {steps}
            TEMPERATURE {temperature}
            &THERMOSTAT
              TYPE CSVR
              TIME_CON 200
            &END THERMOSTAT
            &PRINT
              &TRAJECTORY
                EACH 10
                FILENAME {label}_cnnp.traj
                FORMAT DCD
              &END TRAJECTORY
              &VELOCITIES
                EACH 10
                FILENAME {label}_velocities.xyz
              &END VELOCITIES
              &FORCES
                EACH 10
                FILENAME {label}_forces.xyz
              &END FORCES
            &END PRINT
          &END MD
          &CONSTRAINT
            &COLLECTIVE
              COLVAR 1
              TARGET {x0}
              &RESTRAINT
                K {k_spring}
              &END RESTRAINT
            &END COLLECTIVE
            &COLVAR
              &DISTANCE
                ATOMS {atom1} {atom2}
              &END DISTANCE
            &END COLVAR
          &END CONSTRAINT
        &END MOTION
        """
        
        calc = CP2K(label=f"{label}_umbrella_{x0:.1f}", cp2k_input=cp2k_input,
                    directory=dir_name)
        system.set_calculator(calc)
        
        MaxwellBoltzmannDistribution(system, temperature_K=temperature)
        
        dyn = VelocityVerlet(system, timestep=0.5 * units.fs)
        
        traj_file = os.path.join(dir_name, f"{label}_cnnp.traj")
        traj = Trajectory(traj_file, 'w', system)
        dyn.attach(traj.write, interval=10)
        
        dyn.run(steps)
        trajectories.append(traj)
        
        with open(os.path.join(dir_name, "center.txt"), "w") as f:
            f.write(str(x0))
    
    return trajectories

def main():
    pdb_file = 'dopamine_water_oh.pdb'
    atom1 = 15  # Dopamine hydroxyl hydrogen (example)
    atom2 = 179  # OH• oxygen (example)
    x0_list = np.arange(1.5, 5.1, 0.5)
    k_spring = 0.015  # Hartree/Å² (~9.5 kcal/mol/Å²)
    trajs = run_cnnp_md(pdb_file, atom1=atom1, atom2=atom2, x0_list=x0_list, k_spring=k_spring, steps=20000)
    return trajs

if __name__ == "__main__":
    main()