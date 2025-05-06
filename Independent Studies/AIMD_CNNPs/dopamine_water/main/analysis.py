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

def load_ase_trajectory(traj_file, pdb_file):
    """Load ASE trajectory and create MDAnalysis Universe."""
    if not os.path.exists(traj_file) or not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Trajectory or PDB file not found")
    u = mda.Universe(pdb_file, traj_file, format='TRJ')
    traj = Trajectory(traj_file)
    return u, traj

def parse_input_nn(nn_file):
    """Parse input.nn file to extract ACSFs."""
    radials, angulars = [], []
    with open(nn_file, 'r') as f:
        for line in f:
            if line.startswith('symfunction_short'):
                parts = line.split()
                if parts[2] == '2':
                    radials.append(RadialSF(parts 0)
                    radials.append(RadialSF(parts[1], float(parts[3]), float(parts[4]))
                elif parts[2] == '3':
                    angulars.append(AngularSF(parts[1], float(parts[3]), float(parts[4]), float(parts[5]))
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

def compute_rdf(u_aimd, u_cnnp, selection1, selection2, nbins=100, rmax=6.0):
    """Compute radial distribution function."""
    rdf_aimd = rdf.InterRDF(u_aimd.select_atoms(selection1), u_aimd.select_atoms(selection2),
                            nbins=nbins, range=(0, rmax)).run()
    rdf_cnnp = rdf.InterRDF(u_cnnp.select_atoms(selection1), u_cnnp.select_atoms(selection2),
                            nbins=nbins, range=(0, rmax)).run()
    return rdf_aimd.results.bins, rdf_aimd.results.rdf, rdf_cnnp.results.rdf

def compute_vdos(traj_aimd, traj_cnnp, dt=0.005):
    """Compute vibrational density of states."""
    def get_velocities(traj):
        v = np.array([atoms.get_velocities() for atoms in traj])
        return v - v.mean(axis=0)
    
    v_aimd = get_velocities(traj_aimd)
    v_cnnp = get_velocities(traj_cnnp)
    
    freq = fft.fftfreq(len(v_aimd), dt)[:len(v_aimd)//2]
    vdos_aimd = np.abs(fft.fft(v_aimd, axis=0))[:len(v_aimd)//2].mean(axis=(1, 2))
    vdos_cnnp = np.abs(fft.fft(v_cnnp, axis=0))[:len(v_cnnp)//2].mean(axis=(1, 2))
    
    return freq, vdos_aimd, vdos_cnnp

def compute_force_rmse(traj_aimd, model_dir):
    """Compute force RMSE between AIMD and C-NNP."""
    structures = Structures.from_ase_trajectory(traj_aimd)
    return score(structures, model_dir)

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
    slope = np.polyfit(time, msd_vals, 1)[0]
    diffusion = slope / 6 * 1e-4  # Å²/ps to cm²/s
    return time, msd_vals, diffusion

def compute_dihedral_distribution(u_cnnp, dihedral_indices):
    """Compute distribution of a specified dihedral angle."""
    dihedrals = []
    for ts in u_cnnp.trajectory:
        dihedral = u_cnnp.atoms[dihedral_indices].dihedral.value()
        dihedrals.append(dihedral)
    return np.array(dihedrals)

def plot_results(rdf_data, vdos_data, force_rmse, hbond_data, diffusion_data, dihedral_data):
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
    
    axes[0, 2].text(0.5, 0.5, f'Force RMSE: {force_rmse:.3f} eV/Å', 
                    ha='center', va='center', fontsize=12)
    axes[0, 2].set_title('Force RMSE')
    axes[0, 2].axis('off')
    
    axes[1, 0].plot(hbond_data[0], hbond_data[1])
    axes[1, 0].set_title('Dopamine-Water H-bonds')
    axes[1, 0].set_xlabel('Time (ps)')
    axes[1, 0].set_ylabel('Number of H-bonds')
    
    axes[1, 1].plot(diffusion_data[0], diffusion_data[1])
    axes[1, 1].set_title(f'Dopamine MSD (D = {diffusion_data[2]:.2e} cm²/s)')
    axes[1, 1].set_xlabel('Time (ps)')
    axes[1, 1].set_ylabel('MSD (Å²)')
    
    axes[1, 2].hist(dihedral_data, bins=50, range=(-180, 180))
    axes[1, 2].set_title('C1-C7-C8-N Dihedral Distribution')
    axes[1, 2].set_xlabel('Dihedral Angle (degrees)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.close()

def main():
    aimd_traj_file = 'dopamine_aimd.traj'
    cnnp_traj_file = 'dopamine_cnnp.traj'
    pdb_file = 'dopamine_water.pdb'
    nn_file = 'final-training/input.nn'
    model_dir = 'final-training'
    
    u_aimd, traj_aimd = load_ase_trajectory(aimd_traj_file, pdb_file)
    u_cnnp, traj_cnnp = load_ase_trajectory(cnnp_traj_file, pdb_file)
    
    radials, angulars = parse_input_nn(nn_file)
    visualize_acsf(radials, angulars)
    
    rdf_data = compute_rdf(u_aimd, u_cnnp, 'resname DOP and name O1 O2', 'resname SOL and name H1 H2')
    vdos_data = compute_vdos(traj_aimd, traj_cnnp)
    force_rmse = compute_force_rmse(traj_aimd, model_dir)
    hbond_data = compute_hbonds(u_cnnp)
    diffusion_data = compute_diffusion(u_cnnp)
    
    # Example dihedral indices for C1-C7-C8-N (adjust based on PDB numbering)
    dihedral_indices = [0, 6, 7, 10]
    dihedral_data = compute_dihedral_distribution(u_cnnp, dihedral_indices)
    
    plot_results(rdf_data, vdos_data, force_rmse, hbond_data, diffusion_data, dihedral_data)
    
    print(f'Force RMSE: {force_rmse:.3f} eV/Å')
    print(f'Diffusion coefficient: {diffusion_data[2]:.2e} cm²/s (expected ~6.0e-6 cm²/s)')
    print(f'Average number of H-bonds: {np.mean(hbond_data[1]):.2f}')

if __name__ == "__main__":
    main()