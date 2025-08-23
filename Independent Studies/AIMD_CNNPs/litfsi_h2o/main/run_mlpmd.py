import os
import subprocess
import argparse
import random
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

# Define simulation box and atom types
CELL_SIZE = 14.936  # Angstrom
ATOM_TYPES = ["H", "O", "Li", "F", "S", "C", "N"]


def read_xyz_manual(file):
    """Read XYZ file to create ASE Atoms object."""
    with open(file, "r") as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    if natoms != 244:
        raise ValueError(f"Expected 244 atoms, but found {natoms} in {file}")

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
        f.write("LAMMPS data file for LiTFSI-H2O system\n\n")
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
    atoms.set_cell([cell_size, cell_size, cell_size])
    atoms.set_pbc([True, True, True])
    write_lammps_data_manual(lammps_data_file, atoms, atom_types)
    print(f"Converted {xyz_file} to {lammps_data_file}")


def generate_lammps_input(
    label,
    results_dir,
    lammps_data_file,
    model_file,
    md_steps,
    atom_types=ATOM_TYPES,
    timestep=0.5,  # unit fs
    temperature=298.0,
    thermostat_damping=100.0,  # unit fs
):
    """Generate LAMMPS input script for a single-stage MD simulation."""
    input_script = os.path.join("../logs", f"{label}.in")  # Save .in file in data_dir

    # Output file paths
    md_pos_file = os.path.join(results_dir, f"{label}-md-pos.xyz")
    md_frc_file = os.path.join(results_dir, f"{label}-md-frc.xyz")
    md_thermo_file = os.path.join(results_dir, f"{label}-md-thermo.log")

    mass_commands = ""
    for i, symbol in enumerate(atom_types):
        atom_type_index = i + 1
        mass = atomic_masses[atomic_numbers[symbol]]
        mass_commands += f"mass {atom_type_index} {mass:.4f} # {symbol}\n"

    # Generate a random seed for velocity initialization
    seed = 656435

    lammps_input_content = f"""
# LAMMPS input script for MLPMD with Allegro model (Real Units)
# Workflow: Single MD run
# ----------------------------------------------------------------------------
# Variable Definitions
variable        TEMP equal {temperature}
variable        SEED equal {seed}

# General Setup
# Energy: kcal/mol, Distance: Angstrom, Time: fs
units           real
atom_style      atomic
boundary        p p p
newton          on

# Neighbor list settings
neighbor        2.0 bin
neigh_modify    every 5 delay 0 check no

# System Definition
read_data       {lammps_data_file}
{mass_commands}
# Potential Definition
pair_style      allegro
pair_coeff      * * {model_file} {" ".join(atom_types)}

# Initialize Velocities
velocity        all create ${{TEMP}} ${{SEED}}

# ----------------------------------------------------------------------------
# MD Run (NVT)
# ----------------------------------------------------------------------------
timestep        {timestep}
fix             1 all nvt temp ${{TEMP}} ${{TEMP}} {thermostat_damping}

thermo_style    custom step temp pe ke etotal econserve temp vol press
thermo          100

# Dump trajectory and forces (custom format with unwrapped positions)
dump            md_dump_pos all custom 100 {md_pos_file} element xu yu zu
dump_modify     md_dump_pos element {" ".join(atom_types)} sort id

dump            md_dump_frc all custom 100 {md_frc_file} element fx fy fz
dump_modify     md_dump_frc element {" ".join(atom_types)} sort id


# This file is easier to parse than the main log file.
shell rm -f {md_thermo_file}
fix             thermo_header all print 1 "Step PotEng Temp" file {md_thermo_file} screen no
run             0
unfix           thermo_header
fix             thermo_out all print 100 "$(step) $(pe) $(temp)" append {md_thermo_file} screen no

print "Starting {md_steps}-step NVT MD run..."
run             {md_steps}
print "MD run complete."

# ----------------------------------------------------------------------------
print "Simulation finished."
"""
    with open(input_script, "w") as f:
        f.write(lammps_input_content)

    return input_script


def run_mlpmd(
    xyz_file,
    label="litfsi_h2o_allegro",
    md_time=50.0,  # MD time in ps
    timestep=0.5,  # Timestep in fs
    data_dir="../data",
    results_dir="../results",
    log_dir="../logs",
    np_processes=9,  # OPENMPI = 9 for full usage of cpus
    model_file="../results/allegro_model_output_primary/deployed.nequip.pth",
    atom_types=ATOM_TYPES,
    temperature=298.0,
    thermostat_damping=100.0,  # Damping time in fs
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Convert simulation time from ps to steps
    steps_per_ps = 1000 / timestep
    md_steps = int(md_time * steps_per_ps)

    print("🚀 Starting MLPMD Simulation Workflow 🚀")
    print(f"MD:    {md_time} ps ({md_steps} steps)")
    print(f"Timestep:      {timestep} fs")
    print(f"Temperature:   {temperature} K")

    lammps_data_file = os.path.join(data_dir, f"{label}.data")
    convert_xyz_to_lammps_data(xyz_file, lammps_data_file, atom_types, CELL_SIZE)

    input_file = generate_lammps_input(
        label,
        os.path.abspath(results_dir),
        os.path.abspath(lammps_data_file),
        os.path.abspath(model_file),
        md_steps,
        atom_types,
        timestep,
        temperature,
        thermostat_damping,
    )
    out_file = os.path.join(log_dir, f"{label}.out")

    lammps_exec = "/Users/yue-minwu/Ind_Stud/AIMD_CNNP/lammps/build/lmp"
    if not os.path.isfile(lammps_exec):
        raise FileNotFoundError(f"LAMMPS executable not found: {lammps_exec}")

    cmd = [
        "mpirun",
        "-np",
        str(np_processes),
        lammps_exec,
        "-in",
        input_file,
        "-log",
        os.path.abspath(out_file),
    ]

    print(f"\nExecuting LAMMPS command: {' '.join(cmd)}")
    try:
        working_dir = log_dir  # Run from the log directory
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=os.path.abspath(working_dir),
        )
        print(f"\n✅ LAMMPS simulation completed successfully. Log file at: {out_file}")
    except subprocess.CalledProcessError as e:
        print("\n❌ ERROR: LAMMPS simulation failed.")
        print(f"--- LAMMPS Log File ({out_file}) ---")
        if os.path.exists(out_file):
            with open(out_file, "r") as log:
                print(log.read())
        else:
            print("Log file not found.")
        print("--- End Log File ---")
        raise RuntimeError("LAMMPS simulation failed. Check the log file for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single-stage MD simulation with LAMMPS for AL"
    )
    parser.add_argument(
        "--xyz_file",
        default="../results/litfsi_h2o_relax-pos.xyz",
        help="Path to the initial XYZ structure file.",
    )
    parser.add_argument(
        "--md_time",
        type=float,
        default=50.0,
        help="MD time in picoseconds (ps).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep in femtoseconds (fs).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Simulation temperature in Kelvin (K).",
    )
    parser.add_argument(
        "--thermostat_damping",
        type=float,
        default=100.0,
        help="Thermostat damping time in femtoseconds (fs).",
    )
    parser.add_argument(
        "--np_processes",
        type=int,
        default=9,
        help="Number of MPI processes for LAMMPS.",
    )
    parser.add_argument(
        "--model_file",
        default="../results/allegro_model_output_primary/deployed.nequip.pth",
        help="Path to the Allegro model file.",
    )
    parser.add_argument(
        "--data_dir",
        default="../data",
        help="Directory for LAMMPS data files.",
    )
    parser.add_argument(
        "--results_dir",
        default="../results",
        help="Directory for simulation output files.",
    )
    parser.add_argument(
        "--log_dir",
        default="../logs",
        help="Directory for LAMMPS log files.",
    )
    parser.add_argument(
        "--label",
        default="litfsi_h2o_allegro",
        help="Label for output files.",
    )
    args = parser.parse_args()

    run_mlpmd(
        xyz_file=args.xyz_file,
        label=args.label,
        md_time=args.md_time,
        timestep=args.timestep,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
        np_processes=args.np_processes,
        model_file=args.model_file,
        temperature=args.temperature,
        thermostat_damping=args.thermostat_damping,
    )
