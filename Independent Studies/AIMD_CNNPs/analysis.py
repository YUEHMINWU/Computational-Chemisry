#!/usr/bin/env python

import numpy as np
import aml
import aml.score as mlps
import aml.acsf as acsf
import MDAnalysis as mda
from MDAnalysis.analysis import rdf, diffusionmap
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
import matplotlib.pyplot as plt
from ase.io import Trajectory
from pathlib import Path

def load_ase_trajectory(traj_file, pdb_file):
    """Load ASE trajectory and create MDAnalysis Universe."""
    traj = Trajectory(traj_file)
    u = mda.Universe(pdb_file, traj_file, format='ASETRAJ')
    return u, traj

def parse_input_nn(nn_file):
    """Parse input.nn to extract radial and angular ACSFs."""
    radials = []
    angulars = []
    with open(nn_file, 'r') as f:
        for line in f:
            if line.startswith('symfunction_short'):
                parts = line.split()
                element1 = parts[1]
                sf_type = int(parts[2])
                if sf_type == 2:  # Radial (G2)
                    element2 = parts[3]
                    eta, mu, r_c = map(float, parts[4:7])
                    radials.append(acsf.RadialSF(eta=eta, mu=mu, r_c=r_c))
                elif sf_type == 3:  # Angular (G3)
                    element2, element3 = parts[3:5]
                    eta, lam, zeta, r_c = map(float, parts[5:9])
                    angulars.append(acsf.AngularSF(lam=lam, zeta=zeta, eta=eta, r_c=r_c, mu=0.0))
    return radials, angulars

def visualize_acsf(radials, angulars, filename='acsf_plots.png'):
    """Visualize radial and angular ACSFs."""
    plt.style.use('seaborn')
    f_cut = acsf.f_cut_cos
    
    plt.figure(figsize=(12, 4))
    
    # Radial ACSFs
    plt.subplot(121)
    r = np.linspace(0, max(radial.r_c for radial in radials), 1000)
    for radial in radials:
        f = acsf.f_radial(r, radial.r_c, radial.mu, radial.eta, f_cut)
        f /= max(f)  # Normalize
        plt.plot(r * aml.angstrom, f, lw=2)
    for r_c in {radial.r_c for radial in radials}:
        plt.plot(r * aml.angstrom, f_cut(r, r_c), 'k:', lw=1)
    plt.xlabel('Radial distance (Å)')
    plt.ylabel('ACSF value')
    plt.title('Radial Symmetry Functions')
    
    # Angular ACSFs
    plt.subplot(122)
    theta = np.linspace(0, 2 * np.pi, 1000)
    for angular in angulars:
        f = acsf.f_angular(theta, angular)
        plt.plot(theta, f, lw=2)
    plt.xlabel('θ (rad)')
    plt.ylabel('ACSF value')
    plt.xticks([0, np.pi, 2 * np.pi], ['0', 'π', '2π'])
    plt.title('Angular Symmetry Functions')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def compute_rdf(u_aimd, u_cnnp, selection1, selection2, nbins=100, rmax=6.0):
    """Compute RDF for AIMD and C-NNP trajectories."""
    rdf_aimd = rdf.InterRDF(u_aimd.select_atoms(selection1), u_aimd.select_atoms(selection2),
                            nbins=nbins, range=(0.0, rmax))
    rdf_aimd.run()
    
    rdf_cnnp = rdf.InterRDF(u_cnnp.select_atoms(selection1), u_cnnp.select_atoms(selection2),
                            nbins=nbins, range=(0.0, rmax))
    rdf_cnnp.run()
    
    return rdf_aimd.bins, rdf_aimd.rdf, rdf_cnnp.rdf

def compute_vdos(traj_aimd, traj_cnnp, dt=0.005):
    """Compute VDOS from velocities (dt in ps, assuming 0.5 fs timestep, output every 10 steps)."""
    vel_aimd = np.array([frame.get_velocities() for frame in traj_aimd])
    vel_cnnp = np.array([frame.get_velocities() for frame in traj_cnnp])
    
    # Compute velocity autocorrelation
    def autocorr(vel):
        result = np.correlate(vel.ravel(), vel.ravel(), mode='full')
        result = result[result.size//2:]
        return result[:len(vel)] / result[0]
    
    autocorr_aimd = autocorr(vel_aimd)
    autocorr_cnnp = autocorr(vel_cnnp)
    
    # FFT to get VDOS
    freq = np.fft.fftfreq(len(autocorr_aimd), d=dt)[:len(autocorr_aimd)//2]
    vdos_aimd = np.abs(np.fft.fft(autocorr_aimd))[:len(autocorr_aimd)//2]
    vdos_cnnp = np.abs(np.fft.fft(autocorr_cnnp))[:len(autocorr_cnnp)//2]
    
    return freq, vdos_aimd, vdos_cnnp

def compute_force_rmse(traj_aimd, model_dir):
    """Compute force RMSE between AIMD and C-NNP predictions."""
    frames = []
    for frame in traj_aimd:
        frames.append(aml.Frame(
            positions=frame.get_positions(),
            forces=frame.get_forces(),
            energy=frame.get_potential_energy(),
            cell=frame.get_cell(),
            pbc=frame.get_pbc(),
            numbers=frame.get_atomic_numbers()
        ))
    structures_ref = aml.Structures.from_frames(frames, stride=1, probability=1.0)
    
    rmse = mlps.run_rmse_test(str(Path(model_dir)), None, structures_ref)
    return rmse

def compute_hbonds(u_cnnp):
    """Compute hydrogen bonds between dopamine and water."""
    hb = HydrogenBondAnalysis(
        universe=u_cnnp,
        donors_sel="resname DOP and (name H1 or name H2 or name H10 or name H11 or name H12)",
        acceptors_sel="resname H2O and name O",
        d_a_cut=3.0,  # Donor-acceptor distance (Å)
        d_h_a_angle_cut=150  # Angle cutoff (degrees)
    )
    hb.run()
    return hb.count_by_time(), hb.times

def compute_diffusion(u_cnnp):
    """Compute diffusion coefficient of dopamine."""
    ag = u_cnnp.select_atoms("resname DOP")
    diff = diffusionmap.DiffusionMap(ag, verbose=True)
    diff.run()
    # Convert MSD slope to diffusion coefficient (Å²/ps to cm²/s)
    D = diff.results.D[0] / 6 * 1e-4  # Divide by 6 for 3D, convert Å²/ps to cm²/s
    return diff.times, diff.msd, D

def plot_results(rdf_data, vdos_data, force_rmse, hbond_data, diffusion_data):
    """Plot RDF, VDOS, force RMSE, H-bonds, and diffusion."""
    plt.figure(figsize=(15, 10))
    
    # RDF
    plt.subplot(231)
    bins, rdf_aimd, rdf_cnnp = rdf_data
    plt.plot(bins, rdf_aimd, label='AIMD')
    plt.plot(bins, rdf_cnnp, label='C-NNP')
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('O_dop-H_wat RDF')
    plt.legend()
    
    # VDOS
    plt.subplot(232)
    freq, vdos_aimd, vdos_cnnp = vdos_data
    plt.plot(freq, vdos_aimd, label='AIMD')
    plt.plot(freq, vdos_cnnp, label='C-NNP')
    plt.xlabel('Frequency (1/ps)')
    plt.ylabel('VDOS')
    plt.title('Vibrational Density of States')
    plt.legend()
    
    # Force RMSE
    plt.subplot(233)
    plt.text(0.5, 0.5, f'Force RMSE: {force_rmse:.3f} eV/Å', 
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.title('Force RMSE')
    
    # H-bonds
    plt.subplot(234)
    counts, times = hbond_data
    plt.plot(times, counts)
    plt.xlabel('Time (ps)')
    plt.ylabel('Number of H-bonds')
    plt.title('Dopamine-Water H-bonds')
    
    # Diffusion
    plt.subplot(235)
    times, msd, D = diffusion_data
    plt.plot(times, msd)
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title(f'Dopamine MSD (D = {D:.2e} cm²/s)')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300)
    plt.close()

def main(aimd_traj='dopamine_aimd.traj', cnnp_traj='dopamine_cnnp.traj', 
         pdb_file='dopamine_water.pdb', model_dir='final-training', nn_file='final-training/nnp-000/input.nn'):
    """Analyze C-NNP MD trajectory and validate against AIMD."""
    # Load trajectories
    u_aimd, traj_aimd = load_ase_trajectory(aimd_traj, pdb_file)
    u_cnnp, traj_cnnp = load_ase_trajectory(cnnp_traj, pdb_file)
    
    # Parse input.nn for ACSFs
    radials, angulars = parse_input_nn(nn_file)
    
    # Visualize ACSFs
    visualize_acsf(radials, angulars)
    
    # RDF: Dopamine oxygen (O1 or O2) to water hydrogen
    rdf_data = compute_rdf(
        u_aimd, u_cnnp,
        selection1="resname DOP and (name O1 or name O2)",
        selection2="resname H2O and name H"
    )
    
    # VDOS
    vdos_data = compute_vdos(traj_aimd, traj_cnnp)
    
    # Force RMSE
    force_rmse = compute_force_rmse(traj_aimd, model_dir)
    
    # H-bonds
    hbond_data = compute_hbonds(u_cnnp)
    
    # Diffusion
    diffusion_data = compute_diffusion(u_cnnp)
    
    # Plot results
    plot_results(rdf_data, vdos_data, force_rmse, hbond_data, diffusion_data)
    
    # Print key metrics
    print(f"Force RMSE: {force_rmse:.3f} eV/Å")
    print(f"Diffusion Coefficient: {diffusion_data[2]:.2e} cm²/s (Expected ~6.0e-6 cm²/s)")
    print(f"Average H-bonds: {np.mean(hbond_data[0]):.2f}")

if __name__ == "__main__":
    main()