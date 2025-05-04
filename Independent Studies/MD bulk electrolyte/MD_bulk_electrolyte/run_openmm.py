from ionic_liquid_simulator import IonicLiquidSimulator

# Simulation parameters
pdb_file = "choline_cl.pdb"  # Replace with your PDB file
forcefield_file = "clpol.xml"  # Replace with your XML force field file
simulation_time_ns = 10  # Simulation time in nanoseconds
num_trajectories = 10  # Number of parallel trajectories

# Fixed parameters
timestep = 0.002  # Timestep in picoseconds (fixed as requested)
npt_steps = 1000000  # 2 ns for NPT equilibration (fixed)
report_interval = 10000  # Reporting interval (fixed)

# Calculate NVT steps from simulation time
steps_per_ns = 1000000 / timestep  # Steps per nanosecond
nvt_steps = int(simulation_time_ns * steps_per_ns)

# Initialize simulator
simulator = IonicLiquidSimulator(pdb_file, forcefield_file, report_interval=report_interval)

# Run NPT equilibration
print("Running NPT equilibration...")
avg_density, avg_volume = simulator.run_npt_equilibration(npt_steps)

# Run NVT production simulations
print(f"Running {num_trajectories} NVT simulations for {simulation_time_ns} ns...")
simulator.run_multiple_nvt_simulations("equilibrated_state.xml", nvt_steps, num_trajectories)

# Save average volume for analysis
with open("avg_volume.txt", "w") as f:
    f.write(str(avg_volume))

print("Simulations completed.")