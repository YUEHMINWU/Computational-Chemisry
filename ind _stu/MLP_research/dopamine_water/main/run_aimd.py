import numpy as np
from ase.io import read, Trajectory
from ase.calculators.cp2k import CP2K
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase import units

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

def run_aimd(pdb_file, label='dopamine', equil_steps=4000, prod_steps=20000):
    """Run BLYP-D3 AIMD with CP2K, continuous equilibration + production."""
    system = read_system(pdb_file)

    # CP2K settings matching paper
    calc = CP2K(
        label=label,
        command='cp2k_shell.psmp',
        xc='BLYP',
        basis_set='DZVP-MOLOPT-SR-GTH',
        pseudo_potential='GTH-BLYP',
        cutoff=330 * units.Ry,
        rel_cutoff=60 * units.Ry,
        max_scf=30,
        eps_scf=5.0e-7,
        charge=1,  # Protonated dopamine
        uks=False,
        print_level='LOW',
        inp="""
        &FORCE_EVAL
          METHOD QS
          &DFT
            &QS
              EXTRAPOLATION_ORDER 3
            &END QS
            &SCF
              SCF_GUESS ATOMIC
              &MIXING
                METHOD BROYDEN_MIXING
              &END
            &END SCF
            &XC
              &XC_FUNCTIONAL BLYP
              &END XC_FUNCTIONAL
              &VDW_POTENTIAL
                DISPERSION_FUNCTIONAL PAIR_POTENTIAL
                &PAIR_POTENTIAL
                  TYPE DFTD3
                  REFERENCE_FUNCTIONAL BLYP
                  PARAMETER_FILE_NAME dftd3.dat
                &END PAIR_POTENTIAL
              &END VDW_POTENTIAL
            &END XC
            &POISSON
              PERIODIC XYZ
              POISSON_SOLVER PERIODIC
            &END POISSON
          &END DFT
          &SUBSYS
            &CELL
              PERIODIC XYZ
            &END CELL
          &END SUBSYS
        &END FORCE_EVAL
        &MOTION
          &MD
            ENSEMBLE NVT
            STEPS {total_steps}
            TIMESTEP 0.5
            TEMPERATURE 298
            &THERMOSTAT
              TYPE CSVR
              REGION MASSIVE
              &CSVR
                TIMECON 100.0
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

    # Continuous simulation: 2 ps equil (4000 steps) + 10 ps prod (20000 steps)
    total_steps = equil_steps + prod_steps
    dyn = VelocityVerlet(system, timestep=0.5 * units.fs)
    
    # Save full trajectory
    full_traj = Trajectory(f'{label}_full.traj', 'w', system)
    dyn.attach(full_traj.write, interval=10)
    
    # Save production-only trajectory for AML
    prod_traj = Trajectory(f'{label}_aimd.traj', 'w', system)
    def save_prod(step, equil_steps=equil_steps):
        if step * 10 >= equil_steps:  # Save after equilibration
            prod_traj.write()
    dyn.attach(save_prod, interval=10)
    
    # Run simulation
    dyn.run(total_steps)

    return prod_traj

def main():
    """Main function to run AIMD simulation."""
    # Replace with your Packmol-generated PDB file path
    pdb_file = 'dopamine_water.pdb'
    traj = run_aimd(pdb_file)
    return traj

if __name__ == "__main__":
    main()