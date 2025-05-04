#!/usr/bin/env python

import numpy as np
from ase.io import read, Trajectory
from ase.calculators.cp2k import CP2K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase import units
from pathlib import Path

def read_system(pdb_file, cell=[11.08, 11.99, 13.04]):
    """Read dopamine + water system from Packmol PDB file and set cell."""
    system = read(pdb_file, format='proteins')
    # Verify system: 1 dopamine (22 atoms, C8H12NO2+) + 52 waters (156 atoms)
    if len(system) != 178:
        raise ValueError(f"Expected 178 atoms (1 dopamine + 52 waters), got {len(system)}")
    # Set orthorhombic cell from paper
    system.set_cell(cell)
    system.center()
    system.pbc = True  # Periodic boundary conditions
    return system

def run_cnnp_md(pdb_file, model_dir='final-training', label='dopamine', steps=200000):
    """Run C-NNP-based MD simulation with CP2K."""
    system = read_system(pdb_file)
    model_dir = Path(model_dir)
    
    # CP2K settings for C-NNP MD, adapted from zundel.inp
    calc = CP2K(
        label=label,
        command='cp2k_shell.psmp',
        print_level='LOW',
        inp=f"""
        &GLOBAL
          PROJECT {label}
          RUN_TYPE MD
        &END GLOBAL
        &FORCE_EVAL
          METHOD NNP
          &NNP
            NNP_INPUT_FILE_NAME {model_dir}/nnp-000/input.nn
            SCALE_FILE_NAME {model_dir}/nnp-000/scaling.data
            &MODEL
              WEIGHTS {model_dir}/nnp-000/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-001/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-002/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-003/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-004/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-005/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-006/weights
            &END MODEL
            &MODEL
              WEIGHTS {model_dir}/nnp-007/weights
            &END MODEL
            &PRINT
              &ENERGIES SILENT
                &EACH
                  MD 1
                &END EACH
              &END ENERGIES
            &END PRINT
          &END NNP
          &SUBSYS
            &CELL
              ABC 11.08 11.99 13.04
              PERIODIC XYZ
            &END CELL
          &END SUBSYS
        &END FORCE_EVAL
        &MOTION
          &MD
            ENSEMBLE NVT
            STEPS {steps}
            TIMESTEP 0.5
            TEMPERATURE 298
            &THERMOSTAT
              TYPE CSVR
              REGION MASSIVE
              &CSVR
                TIMECON 200.0
              &END CSVR
            &END THERMOSTAT
          &END MD
          &PRINT
            &TRAJECTORY
              &EACH
                MD 10
              &END EACH
            &END TRAJECTORY
            &VELOCITIES
              &EACH
                MD 10
              &END EACH
            &END VELOCITIES
            &FORCES
              &EACH
                MD 10
              &END EACH
            &END FORCES
            &RESTART OFF
            &RESTART_HISTORY OFF
          &END PRINT
        &END MOTION
        """
    )

    system.set_calculator(calc)
    MaxwellBoltzmannDistribution(system, temperature_K=298)

    # Run MD simulation
    dyn = VelocityVerlet(system, timestep=0.5 * units.fs)
    traj = Trajectory(f'{label}_cnnp.traj', 'w', system)
    dyn.attach(traj.write, interval=10)
    dyn.run(steps)

    return traj

def main():
    """Main function to run C-NNP MD simulation."""
    # Replace with your Packmol-generated PDB file path
    pdb_file = 'dopamine_water.pdb'
    traj = run_cnnp_md(pdb_file)
    return traj

if __name__ == "__main__":
    main()