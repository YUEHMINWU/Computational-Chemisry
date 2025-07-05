import os
import subprocess
import argparse
import sys
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers  # MODIFIED LINE: Import ASE data

# Define simulation box and atom types
CELL_SIZE = 14.936  # Angstrom
ATOM_TYPES = ["H", "O", "Li", "F", "S", "C", "N"]


def is_jupyter():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


def read_xyz_manual(xyz_file):
    """Read XYZ file to create ASE Atoms object."""
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    if natoms != 244:
        raise ValueError(f"Expected 244 atoms, but found {natoms} in {xyz_file}")

    symbols = []
    positions = []
    for i, line in enumerate(lines[2 : 2 + natoms], 1):
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ line {i}: {line.strip()}")
        symbol = parts[0]
        if symbol not in ATOM_TYPES:
            raise ValueError(
                f"Atom {i} has unexpected symbol {symbol}. Expected: {ATOM_TYPES}"
            )
        symbols.append(symbol)
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def write_lammps_data_manual(fd, atoms, specorder, cell_size=CELL_SIZE):
    """Write LAMMPS data file."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    symbol_to_type = {symbol: idx + 1 for idx, symbol in enumerate(specorder)}

    with open(fd, "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(specorder)} atom types\n\n")
        f.write(f"0.0 {cell_size} xlo xhi\n")
        f.write(f"0.0 {cell_size} ylo yhi\n")
        f.write(f"0.0 {cell_size} zlo zhi\n\n")
        f.write("Atoms\n\n")
        for i, (symbol, pos) in enumerate(zip(symbols, positions)):
            atom_type = symbol_to_type[symbol]
            f.write(f"{i + 1} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def convert_xyz_to_lammps_data(
    xyz_file, lammps_data_file, atom_types=ATOM_TYPES, cell_size=CELL_SIZE
):
    """Convert XYZ file to LAMMPS data file."""
    if not os.path.isfile(xyz_file):
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    atoms = read_xyz_manual(xyz_file)
    if len(atoms) != 244:
        raise ValueError(f"Expected 244 atoms, but read {len(atoms)} from {xyz_file}")

    atoms.set_cell([cell_size, cell_size, cell_size])
    atoms.set_pbc([True, True, True])
    write_lammps_data_manual(lammps_data_file, atoms, atom_types)
    print(f"Converted {xyz_file} to {lammps_data_file}")


def generate_lammps_input(
    label,
    data_dir,
    log_dir,
    results_dir,
    lammps_data_file,
    model_file,
    atom_types=ATOM_TYPES,
    cell_size=CELL_SIZE,
    steps=200000,
    timestep=0.0005, # unit ps
    temperature=298.0,
    thermostat_damping=100.0,
):
    """Generate LAMMPS input script for MLPMD simulation."""
    input_script = os.path.join(log_dir, f"{label}.in")
    output_pos = os.path.join(results_dir, f"{label}-pos.xyz")
    output_vel = os.path.join(results_dir, f"{label}-vel.xyz")
    output_frc = os.path.join(results_dir, f"{label}-frc.xyz")

    # MODIFIED BLOCK: Generate mass commands for the input script
    mass_commands = ""
    for i, symbol in enumerate(atom_types):
        atom_type_index = i + 1
        mass = atomic_masses[atomic_numbers[symbol]]
        mass_commands += f"mass {atom_type_index} {mass:.4f}\n"

    lammps_input_content = f"""
# LAMMPS input script for MLPMD with Allegro model
units metal
atom_style atomic
newton on
dimension 3
boundary p p p

# Read LAMMPS data file
read_data {lammps_data_file}

# if you want to run a larger system, simply replicate the system in space
# replicate 3 3 3

# MODIFIED BLOCK: Set masses for each atom type
{mass_commands}

# Pair style for Allegro
pair_style allegro
pair_coeff * * {model_file} {" ".join(atom_types)}

# Neighbor list settings
neighbor 2.0 bin
neigh_modify delay 5 every 1

# Set timestep
timestep {timestep}

# NVT ensemble with Nosé-Hoover
fix 1 all nvt temp {temperature} {temperature} {thermostat_damping}
thermo 100
thermo_style custom step temp pe ke etotal

# Output trajectory, velocities, and forces
dump pos all xyz 1 {output_pos}
dump_modify pos element {" ".join(atom_types)}
dump vel all custom 1 {output_vel} id type vx vy vz
dump frc all custom 1 {output_frc} id type fx fy fz

# Run simulation
run {steps}
"""
    with open(input_script, "w") as f:
        f.write(lammps_input_content)

    with open(os.path.join(log_dir, "lammps_input.log"), "w") as f:
        f.write(f"Generated LAMMPS input file: {input_script}\n")
        f.write(lammps_input_content)

    return input_script


def run_mlpmd(
    xyz_file,
    label="litfsi_h2o_allegro",
    sim_time=100.0,
    timestep=0.5,
    data_dir="../data",
    results_dir="../results",
    log_dir="../logs",
    np_processes=9,
    model_file="../results/allegro_model_output/deployed.nequip.pth",
    atom_types=ATOM_TYPES,
    cell_size=CELL_SIZE,
    temperature=298.0,
    thermostat_damping=100.0,
):
    """Run MLPMD simulation using LAMMPS."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    steps_per_ps = 1000 / timestep
    sim_steps = int(sim_time * steps_per_ps)
    print(f"Simulation steps: {sim_steps}")

    lammps_data_file = os.path.join(data_dir, f"{label}.data")
    convert_xyz_to_lammps_data(xyz_file, lammps_data_file, atom_types, cell_size)

    input_file = os.path.abspath(
        generate_lammps_input(
            label,
            data_dir,
            log_dir,
            results_dir,
            lammps_data_file,
            model_file,
            atom_types,
            cell_size,
            sim_steps,
            timestep,
            temperature,
            thermostat_damping,
        )
    )
    out_file = os.path.join(log_dir, f"{label}.out")

    # This points to the correct executable you built
    lammps_exec = "/Users/yue-minwu/Ind_Stud/AIMD_CNNP/lammps/build/lmp"
    if not os.path.isfile(lammps_exec):
        raise FileNotFoundError(f"LAMMPS executable not found: {lammps_exec}")

    # The 'mpirun' command from your conda environment will be used automatically
    cmd = [
        "mpirun",
        "-np",
        str(np_processes),
        lammps_exec,
        "-in",
        input_file,
        "-log",
        out_file,
    ]

    env = os.environ.copy()

    print(f"Executing LAMMPS command: {' '.join(cmd)}")
    try:
        working_dir = os.path.dirname(input_file)
        result = subprocess.run(
            cmd, env=env, check=True, capture_output=True, text=True, cwd=working_dir
        )
        with open(os.path.join(log_dir, "lammps_wrapper.log"), "w") as f:
            f.write(f"LAMMPS command: {' '.join(cmd)}\n")
            f.write(f"LAMMPS stdout:\n{result.stdout}\n")
            f.write(f"LAMMPS stderr:\n{result.stderr}\n")
            f.write(f"Using binary: {lammps_exec}\n")
    except subprocess.CalledProcessError as e:
        # Print the LAMMPS log file for easier debugging
        print(f"--- LAMMPS Log File ({out_file}) ---")
        if os.path.exists(out_file):
            with open(out_file, "r") as log:
                print(log.read())
        else:
            print("Log file not found.")
        print("--- End Log File ---")
        raise RuntimeError(
            f"LAMMPS simulation failed. See log for details. Error: {e.stderr}"
        )


def main():
    if is_jupyter():
        args = argparse.Namespace(
            xyz_file="../results/litfsi_h2o_relax-pos.xyz",
            sim_time=100.0,
            np_processes=9,
        )
    else:
        parser = argparse.ArgumentParser(description="Run MLPMD simulation with LAMMPS")
        parser.add_argument(
            "--xyz_file",
            default="../results/litfsi_h2o_relax-pos.xyz",
            help="Path to XYZ file",
        )
        parser.add_argument(
            "--sim_time", type=float, default=100.0, help="Simulation time in ps"
        )
        parser.add_argument(
            "--np_processes", type=int, default=9, help="Number of MPI processes"
        )
        args = parser.parse_args()

    run_mlpmd(
        args.xyz_file,
        sim_time=args.sim_time,
        np_processes=args.np_processes,
    )


if __name__ == "__main__":
    main()
