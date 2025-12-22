import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import MDAnalysis as mda
from MDAnalysis.analysis import rdf, hbonds
from ase.io import Trajectory
import os
from aml import Structures
from aml.score import score
from aml.acsf import RadialSF, AngularSF
from pymbar import MBAR
from ase import units

def load_trajectories(pdb_file, aimd_traj_file, cnnp_traj_file):
    """Load AIMD and C-NNP trajectories."""
    # Load AIMD trajectory (ASE format)
    traj_aimd = Trajectory(aimd_traj_file)
    # Load C-NNP trajectory (DCD format)
    u_cnnp = mda.Universe(pdb_file, cnnp_traj_file, format='DCD')
    return traj_aimd, u_cnnp

def parse_input_nn(nn_file):
    """Parse input.nn file to extract ACSFs."""
    radials, angulars = [], []
    with open(nn_file, 'r') as f:
        for line in f:
            if line.startswith('symfunction_short'):
                parts = line.split()
                if parts[2] == '2':
                    radials.append(RadialSF(parts[1], float(parts[3]), float(parts[4])))
                elif parts[2] == '3':
                    angulars.append(AngularSF(parts[1], float(parts[3]), float(parts[4]), float(parts[5])))
    return radials, angulars

def visualize_acsf(radials, angulars, filename='acsf_plots.png'):
    """Visualize radial and angular symmetry functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    r = np.linspace(0, 6, 100)
    for sf in radials:
        ax1.plot(r, np.exp(-((r - sf.eta)**2) / (2 * sf.rs**2)), label=f'eta={sf.eta}')
    ax1.set_title('Radial ACSFs')
    ax1.set_xlabel('Distance (Å)')
    ax1.legend()
    
    theta = np.linspace(0, np.pi, 100)
    for sf in angulars:
        ax2.plot(theta, np.cos(theta)**sf.lambda0 * np.exp(-sf.eta * (theta - sf.theta_s)**2), label=f'eta={sf.eta}')
    ax2.set_title('Angular ACSFs')
    ax2.set_xlabel('Angle (rad)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_rdf(u_aimd_dcd, u_cnnp, selection1, selection2, nbins=100, rmax=6.0):
    """Compute radial distribution function."""
    rdf_aimd = rdf.InterRDF(u_aimd_dcd.select_atoms(selection1), u_aimd_dcd.select_atoms(selection2),
                            nbins=nbins, range=(0, rmax)).run()
    rdf_cnnp = rdf.InterRDF(u_cnnp.select_atoms(selection1), u_cnnp.select_atoms(selection2),
                            nbins=nbins, range=(0, rmax)).run()
    return rdf_aimd.results.bins, rdf_aimd.results.rdf, rdf_cnnp.results.rdf

def compute_vdos(u_aimd_dcd, u_cnnp, dt=0.005):
    """Compute vibrational density of states."""
    def get_velocities(u):
        velocities = []
        for ts in u.trajectory:
            velocities.append(ts.velocities.copy())
        return np.array(velocities)
    
    v_aimd = get_velocities(u_aimd_dcd)
    v_cnnp = get_velocities(u_cnnp)
    
    freq = fft.fftfreq(len(v_aimd), dt)[:len(v_aimd)//2]
    vdos_aimd = np.abs(fft.fft(v_aimd, axis=0))[:len(v_aimd)//2].mean(axis=(1,2))
    vdos_cnnp = np.abs(fft.fft(v_cnnp, axis=0))[:len(v_cnnp)//2].mean(axis=(1,2))
    
    return freq, vdos_aimd, vdos_cnnp

def compute_force_rmse(traj_aimd, model_dir):
    """Compute force RMSE between AIMD and C-NNP."""
    structures = Structures.from_ase_trajectory(traj_aimd)
    return score(structures, model_dir)

def compute_force_disagreement(traj_aimd, model_dir):
    """Compute force disagreement between AIMD and C-NNP."""
    structures = Structures.from_ase_trajectory(traj_aimd)
    forces_aimd = structures.get_forces()
    forces_cnnp = structures.get_forces(model_dir)
    disagreement = np.mean(np.linalg.norm(forces_aimd - forces_cnnp, axis=1))
    return disagreement

def compute_hbonds(u_cnnp):
    """Compute hydrogen bonds between dopamine and water."""
    hbond_analysis = hbonds.HydrogenBondAnalysis(
        u_cnnp, donors_sel='resname DOP and name H1 H2', acceptors_sel='resname SOL and name OW'
    )
    hbond_analysis.run()
    return hbond_analysis.times, hbond_analysis.count_by_time()

def compute_diffusion(u_cnnp):
    """Compute diffusion coefficient of dopamine."""
    from MDAnalysis.analysis.msd import EinsteinMSD
    msd = EinsteinMSD(u_cnnp, select='resname DOP', msd_type='com').run()
    time = msd.times
    msd_vals = msd.results.msd
    slope = np.polyfit(time[10:], msd_vals[10:], 1)[0]
    diffusion = slope / 6 * 1e-4  # Å²/ps to cm²/s
    return time, msd_vals, diffusion

def compute_pmf(pdb_file, umbrella_dirs, atom1, atom2, x0_list, k_spring):
    """Compute free energy profile using MBAR from umbrella sampling."""
    all_data = []
    beta = 1 / (units.kB * 298 / units.Hartree)  # Beta in Hartree^-1
    for dir in umbrella_dirs:
        traj_file = os.path.join(dir, "dopamine_oh_cnnp.traj")
        if not os.path.exists(traj_file):
            raise FileNotFoundError(f"Trajectory file {traj_file} not found")
        universe = mda.Universe(pdb_file, traj_file, format='DCD')
        ag1 = universe.select_atoms(f'index {atom1}')
        ag2 = universe.select_atoms(f'index {atom2}')
        distances = []
        for ts in universe.trajectory:
            dist = np.linalg.norm(ag1.positions[0] - ag2.positions[0])
            distances.append(dist)
        all_data.append(distances)
    
    all_x = np.concatenate(all_data)
    K = len(umbrella_dirs)
    N_total = len(all_x)
    U = np.zeros((K, N_total))
    N_k = [len(data) for data in all_data]
    
    for k in range(K):
        x0_k = x0_list[k]
        for n in range(N_total):
            x_n = all_x[n]
            U[k,n] = 0.5 * beta * k_spring * (x_n - x0_k)**2
    
    mbar = MBAR(U, N_k)
    mbar.run()
    
    x_min = min(all_x)
    x_max = max(all_x)
    x_grid = np.linspace(x_min, x_max, 100)
    PMF = []
    for x in x_grid:
        p_x = 0
        for k in range(K):
            U_k_x = 0.5 * beta * k_spring * (x - x0_list[k])**2
            p_x += np.exp(- (mbar.F[k] + U_k_x))
        pmf_x = -1 / beta * np.log(p_x + 1e-10)  # In Hartree
        PMF.append(pmf_x)
    PMF = np.array(PMF)
    PMF -= np.min(PMF)
    PMF_kcal = PMF * 627.5095  # Convert to kcal/mol
    return x_grid, PMF_kcal

def plot_results(rdf_data, vdos_data, force_rmse, force_disagreement, hbond_data, diffusion_data):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].plot(rdf_data[0], rdf_data[1], label='AIMD')
    axes[0, 0].plot(rdf_data[0], rdf_data[2], label='C-NNP')
    axes[0, 0].set_title('O_dop-H_wat RDF')
    axes[0, 0].set_xlabel('Distance (Å)')
    axes[0, 0].set_ylabel('g(r)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(vdos_data[0], vdos_data[1], label='AIMD')
    axes[0, 1].plot(vdos_data[0], vdos_data[2], label='C-NNP')
    axes[0, 1].set_title('Vibrational Density of States')
    axes[0, 1].set_xlabel('Frequency (1/ps)')
    axes[0, 1].legend()
    
    axes[0, 2].text(0.5, 0.5, f'Force RMSE: {force_rmse:.3f} eV/Å\nForce Disagreement: {force_disagreement:.3f} eV/Å', 
                    ha='center', va='center', fontsize=12)
    axes[0, 2].set_title('Force Metrics')
    axes[0, 2].axis('off')
    
    axes[1, 0].plot(hbond_data[0], hbond_data[1])
    axes[1, 0].set_title('Dopamine-Water H-bonds')
    axes[1, 0].set_xlabel('Time (ps)')
    axes[1, 0].set_ylabel('Number of H-bonds')
    
    axes[1, 1].plot(diffusion_data[0], diffusion_data[1])
    axes[1, 1].set_title(f'Dopamine MSD (D = {diffusion_data[2]:.2e} cm²/s)')
    axes[1, 1].set_xlabel('Time (ps)')
    axes[1, 1].set_ylabel('MSD (Å²)')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.close()

def main():
    aimd_traj_file = 'dopamine_oh_aimd.traj'
    cnnp_traj_file = 'dopamine_oh_cnnp.traj'
    pdb_file = 'dopamine_water_oh.pdb'
    nn_file = 'final-training/input.nn'
    model_dir = 'final-training'
    
    # Convert AIMD trajectory to DCD if not already done
    aimd_dcd_file = 'dopamine_oh_aimd.dcd'
    if not os.path.exists(aimd_dcd_file):
        from ase.io import read, write
        traj = read(aimd_traj_file, index=':')
        write(aimd_dcd_file, traj)
    
    # Load trajectories
    traj_aimd, u_cnnp = load_trajectories(pdb_file, aimd_traj_file, cnnp_traj_file)
    u_aimd_dcd = mda.Universe(pdb_file, aimd_dcd_file, format='DCD')
    
    # Parse ACSFs
    radials, angulars = parse_input_nn(nn_file)
    visualize_acsf(radials, angulars)
    
    # Compute RDF
    rdf_data = compute_rdf(u_aimd_dcd, u_cnnp, 'resname DOP and name O1 O2', 'resname SOL and name H1 H2')
    
    # Compute VDOS
    vdos_data = compute_vdos(u_aimd_dcd, u_cnnp)
    
    # Compute force RMSE
    force_rmse = compute_force_rmse(traj_aimd, model_dir)
    
    # Compute force disagreement
    force_disagreement = compute_force_disagreement(traj_aimd, model_dir)
    
    # Compute hydrogen bonds
    hbond_data = compute_hbonds(u_cnnp)
    
    # Compute diffusion
    diffusion_data = compute_diffusion(u_cnnp)
    
    # Plot results
    plot_results(rdf_data, vdos_data, force_rmse, force_disagreement, hbond_data, diffusion_data)
    
    # Free energy profile
    umbrella_dirs = [f"umbrella_{x:.1f}" for x in np.arange(1.5, 5.1, 0.5)]
    atom1 = 15  # Dopamine hydroxyl H (adjust based on PDB)
    atom2 = 179  # OH• O (adjust based on PDB)
    x0_list = np.arange(1.5, 5.1, 0.5)
    k_spring = 0.015  # Hartree/Å²
    x_grid, PMF = compute_pmf(pdb_file, umbrella_dirs, atom1, atom2, x0_list, k_spring)
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, PMF, 'b-', label='Free Energy Profile')
    plt.xlabel('Reaction Coordinate (Å)')
    plt.ylabel('Free Energy (kcal/mol)')
    plt.title('Free Energy Profile of Dopamine-OH• Interaction')
    plt.grid(True)
    plt.legend()
    plt.savefig('free_energy_profile.png')
    plt.close()
    
    # Print key metrics
    print(f'Force RMSE: {force_rmse:.3f} eV/Å')
    print(f'Force Disagreement: {force_disagreement:.3f} eV/Å')
    print(f'Diffusion Coefficient: {diffusion_data[2]:.2e} cm²/s')
    print(f'Average Number of H-bonds: {np.mean(hbond_data[1]):.2f}')
    print(f'Energy Barrier Height: {np.max(PMF) - np.min(PMF):.2f} kcal/mol')

if __name__ == "__main__":
    main()