import os
import subprocess
import shutil
import argparse
import sys


def is_jupyter():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False

def generate_cp2k_input(label, data_dir, log_dir, template_file="aimd.inp"):
    template_path = os.path.join(data_dir, template_file)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"CP2K input template not found: {template_path}")

    # Copy the template input file to results directory
    inp_file = os.path.join(log_dir, f"{label}.inp")
    shutil.copyfile(template_path, inp_file)

    with open(os.path.join(log_dir, "cp2k_input.log"), "w") as f:
        f.write(f"Copied CP2K input file: {inp_file}\n")
        f.write(f"Template used: {template_path}\n")
        with open(inp_file, "r") as inp_f:
            f.write(inp_f.read())

    dftd3_src = "/opt/homebrew/share/cp2k/data/dftd3.dat"
    dftd3_dst = os.path.join(log_dir, "dftd3.dat")
    if os.path.exists(dftd3_src):
        shutil.copyfile(dftd3_src, dftd3_dst)
    else:
        raise FileNotFoundError(f"DFT-D3 parameter file not found: {dftd3_src}")

    return inp_file


def run_aimd(
    pdb_file,
    label="pBQ_h2o",
    sim_time=10.0,
    timestep=0.5,
    data_dir="../data",
    results_dir="/Users/yue-minwu/Ind_Stud/AIMD_CNNP/Q_h2o/results/",
    log_dir="../logs/",
    np_processes=9,
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)



    # Calculate steps
    steps_per_ps = 1000 / timestep  # fs per ps / timestep (10000 steps/ps for 0.10 fs)
    sim_steps = int(sim_time * steps_per_ps)  # e.g., 1000 steps for 1 ps
    print(sim_steps)

    # Generate and run CP2K input
    inp_file = os.path.abspath(generate_cp2k_input(label, data_dir, log_dir))
    out_file = os.path.join(log_dir, f"{label}.out")

    cp2k_psmp = "/opt/homebrew/bin/cp2k.psmp"
    wrapper_script = os.path.join(
        os.path.dirname(__file__), "../../cp2k_shell_wrapper.sh"
    )

    if not os.path.isfile(cp2k_psmp):
        raise FileNotFoundError(f"CP2K binary not found: {cp2k_psmp}")
    if not os.path.isfile(wrapper_script):
        raise FileNotFoundError(f"CP2K wrapper script not found: {wrapper_script}")

    cmd = [
        "mpirun",
        "-np",
        str(np_processes),
        wrapper_script,
        "-i",
        inp_file,
        "-o",
        out_file,
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/lib:/opt/openmpi/lib:" + env.get(
        "DYLD_LIBRARY_PATH", ""
    )
    env["CP2K_DATA_DIR"] = "/opt/homebrew/share/cp2k/data"
    env["OMPI_MCA_btl_tcp_port_min_v4"] = "10000"
    env["OMPI_MCA_btl_tcp_port_range_v4"] = "1000"

    print(f"Executing CP2K command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, env=env, check=True, capture_output=True, text=True
        )
        with open(os.path.join(log_dir, "cp2k_wrapper.log"), "w") as f:
            f.write(f"CP2K command: {' '.join(cmd)}\n")
            f.write(f"CP2K stdout:\n{result.stdout}\n")
            f.write(f"CP2K stderr:\n{result.stderr}\n")
            f.write(f"Using binary: {cp2k_psmp}\n")
    except subprocess.CalledProcessError as e:
        error_msg = f"CP2K simulation failed: {e}\nCommand: {' '.join(cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        raise RuntimeError(error_msg)


def main():
    if is_jupyter():
        # Set default arguments for Jupyter
        args = argparse.Namespace(
            pdb_file="../data/pBQ_h2o.pdb",
            sim_time=10.0,
            np_processes=9,
        )
    else:
        # Parse command-line arguments, ignoring Jupyter-specific ones
        parser = argparse.ArgumentParser(
            description="Run AIMD simulation with CP2K", add_help=False
        )
        parser.add_argument(
            "--pdb_file", default="../data/li_ec_lio_bond.pdb", help="Path to PDB file"
        )
        parser.add_argument(
            "--sim_time", type=float, default=1.0, help="Simulation time in ps"
        )

        parser.add_argument(
            "--np_processes", type=int, default=8, help="Number of MPI processes"
        )
        parser.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit",
        )

        # Filter out Jupyter-specific arguments
        known_args = [arg for arg in sys.argv[1:] if not arg.startswith("--f=")]
        args, unknown = parser.parse_known_args(known_args)

    traj = run_aimd(
        args.pdb_file,
        sim_time=args.sim_time,
        np_processes=args.np_processes,
    )
    return traj


if __name__ == "__main__":
    main()
