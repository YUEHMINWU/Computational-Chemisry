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
    expected_atoms = 178  # Dopamine (22) + 52 waters (156)
    if len(system) != expected_atoms:
        raise ValueError(f"Expected {expected_atoms} atoms but found {len(system)}")
    system.set_cell(cell)
    system.center()
    system.set_pbc(True)
    return system

def run_cnnp_md(pdb_file, model_dir='final-training', label='dopamine', steps=200000):
    """Run C-NNP MD simulation."""
    system = read_system(pdb_file)
    temperature = 298
    
    cp2k_input = f"""
    &GLOBAL
      PROJECT {label}
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
    &END MOTION
    """
    
    calc = CP2K(label=label, cp2k_input=cp2k_input)
    system.set_calculator(calc)
    
    MaxwellBoltzmannDistribution(system, temperature_K=temperature)
    
    dyn = VelocityVerlet(system, timestep=0.5 * units.fs)
    
    traj = Trajectory(f"{label}_cnnp.traj", 'w', system)
    dyn.attach(traj.write, interval=10)
    
    dyn.run(steps)
    
    return traj

def main():
    pdb_file = 'dopamine_water.pdb'
    traj = run_cnnp_md(pdb_file)
    return traj

if __name__ == "__main__":
    main()