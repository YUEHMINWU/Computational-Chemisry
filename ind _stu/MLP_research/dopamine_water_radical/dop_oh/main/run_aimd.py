import numpy as np
from ase.io import read
from ase.calculators.cp2k import CP2K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase import units
import os

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

def run_aimd(pdb_file, label='dopamine_oh', equil_steps=2000, prod_steps=10000, time_step=0.2):
    """Run AIMD simulation with CP2K for reactive system."""
    system = read_system(pdb_file)
    
    cp2k_input = f"""
    &GLOBAL
      PROJECT {label}
      RUN_TYPE MD
    &END GLOBAL
    
    &FORCE_EVAL
      METHOD Quickstep
      &DFT
        BASIS_SET_FILE_NAME BASIS_MOLOPT
        BASIS_SET DZVP-MOLOPT-SR-GTH
        POTENTIAL_FILE_NAME POTENTIAL
        &POISSON
          PERIODIC XYZ
          PSOLVER PERIODIC
        &END POISSON
        &XC
          &XC_FUNCTIONAL BLYP
          &END XC_FUNCTIONAL
          &VDW_POTENTIAL
            DISPERSION_FUNCTIONAL PAIR_POTENTIAL
            &PAIR_POTENTIAL
              TYPE DFTD3
              PARAMETER_FILE_NAME dftd3.dat
              REFERENCE_FUNCTIONAL BLYP
            &END PAIR_POTENTIAL
          &END VDW_POTENTIAL
        &END XC
        &SCF
          MAX_SCF 30
          EPS_SCF 5.0E-7
        &END SCF
        CHARGE 1
        UKS T
        MULTIPLICITY 2
      &END DFT
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
        TIMESTEP {time_step}
        STEPS {equil_steps + prod_steps}
        TEMPERATURE 298
        &THERMOSTAT
          TYPE CSVR
          TIME_CON 200
        &END THERMOSTAT
      &END MD
    &END MOTION
    """
    
    calc = CP2K(label=label, cp2k_input=cp2k_input)
    system.set_calculator(calc)
    
    MaxwellBoltzmannDistribution(system, temperature_K=298)
    
    dyn = VelocityVerlet(system, timestep=time_step * units.fs)
    
    from ase.io import Trajectory
    full_traj = Trajectory(f'{label}_full.traj', 'w', system)
    prod_traj = Trajectory(f'{label}_aimd.traj', 'w', system)
    
    def write_traj(step):
        full_traj.write(system)
        if step >= equil_steps:
            prod_traj.write(system)
    
    dyn.attach(write_traj, interval=10)
    dyn.run(equil_steps + prod_steps)
    
    return prod_traj

def main():
    pdb_file = 'dopamine_water_oh.pdb'
    traj = run_aimd(pdb_file, time_step=0.2)
    return traj

if __name__ == "__main__":
    main()