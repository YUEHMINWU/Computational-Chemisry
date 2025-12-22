import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import msd
from scipy.stats import linregress
from scipy.signal import correlate
import matplotlib.pyplot as plt

class TransportAnalyzer:
    def __init__(self, trajectory_files, pressure_files, dipole_files, topology_file, avg_volume, dt, temperature=300):
        self.trajectory_files = trajectory_files
        self.pressure_files = pressure_files
        self.dipole_files = dipole_files
        self.topology_file = topology_file
        self.avg_volume = avg_volume  # in nm^3
        self.dt = dt  # in seconds
        self.temperature = temperature  # in Kelvin

    def calculate_diffusion(self, selection='all'):
        D_list = []
        msd_data = []
        times = []
        for traj in self.trajectory_files:
            u = mda.Universe(self.topology_file, traj)
            atoms = u.select_atoms(selection)
            MSD = msd.MSD(atoms)
            MSD.run()
            time = MSD.times * self.dt / 1e-12  # Convert to ps
            msd_values = MSD.results.timeseries * 1e-20  # Convert to m^2
            slope, _, _, _, _ = linregress(time, msd_values)
            D = slope / 6 * 1e9  # Convert to 10^-9 m^2/s
            D_list.append(D)
            msd_data.append(msd_values)
            times.append(time)
        D_avg = np.mean(D_list)
        D_std = np.std(D_list)

        # Save diffusion data to file
        avg_msd = np.mean(msd_data, axis=0)
        with open('diffusion_data.txt', 'w') as f:
            f.write('# Time (ps), MSD (m^2), D_avg (10^-9 m^2/s), D_std (10^-9 m^2/s)\n')
            for t, m in zip(times[0], avg_msd):
                f.write(f'{t},{m},{D_avg},{D_std}\n')

        # Save individual trajectory MSDs for reference
        with open('diffusion_trajectories.txt', 'w') as f:
            f.write('# Time (ps), MSD_traj_0 (m^2), MSD_traj_1 (m^2), ...\n')
            max_len = max(len(t) for t in times)
            for i in range(max_len):
                row = [times[0][i] if i < len(times[0]) else times[0][-1]]
                for msd_traj in msd_data:
                    row.append(msd_traj[i] if i < len(msd_traj) else msd_traj[-1])
                f.write(','.join(map(str, row)) + '\n')

        return D_avg, D_std

    def acf(self, x):
        x = x - np.mean(x)
        result = correlate(x, x, mode='full')
        result = result[result.size//2:]
        return result / len(x)

    def calculate_viscosity(self):
        acf_list = []
        times = []
        for file in self.pressure_files:
            data = np.loadtxt(file, delimiter=',', skiprows=1)
            P_off = data[:,1:] * 1e5  # Convert bar to Pa
            acf_avg = np.mean([self.acf(P_off[:,i]) for i in range(3)], axis=0)
            acf_list.append(acf_avg)
            time = np.arange(len(acf_avg)) * self.dt * 1e12  # Convert to ps
            times.append(time)
        
        acf_avg_all = np.mean(acf_list, axis=0)
        time = np.mean(times, axis=0)
        integral = np.cumsum(acf_avg_all) * self.dt
        kB = 1.380649e-23
        V_m3 = self.avg_volume * 1e-27  # Convert nm^3 to m^3
        eta = (V_m3 / (kB * self.temperature)) * integral * 1e3  # Convert Pa·s to mPa·s
        eta_avg = eta[-1]
        eta_std = np.std([np.cumsum(acf)[-1] * (V_m3 / (kB * self.temperature)) * 1e3 for acf in acf_list])

        # Save viscosity data to file
        with open('viscosity_data.txt', 'w') as f:
            f.write('# Time (ps), ACF (Pa^2), eta_avg (mPa·s), eta_std (mPa·s)\n')
            for t, acf in zip(time, acf_avg_all):
                f.write(f'{t},{acf},{eta_avg},{eta_std}\n')

        return eta_avg, eta_std

    def compute_msd(self, M):
        N = len(M)
        msd = []
        for lag in range(1, N//2):
            delta = M[lag:] - M[:-lag]
            msd.append(np.mean(np.sum(delta**2, axis=1)))
        return np.array(msd)

    def calculate_conductivity(self):
        slopes = []
        msd_data = []
        times = []
        for file in self.dipole_files:
            data = np.loadtxt(file, delimiter=',', skiprows=1)
            M = data[:,1:]  # in e*nm
            msd_M = self.compute_msd(M) * 1e-18  # Convert (e*nm)^2 to (e*m)^2
            time = np.arange(1, len(msd_M)+1) * self.dt * 1e12  # Convert to ps
            slope, _, _, _, _ = linregress(time, msd_M)
            slopes.append(slope)
            msd_data.append(msd_M)
            times.append(time)
        
        avg_slope = np.mean(slopes)
        kB = 1.380649e-23
        V_m3 = self.avg_volume * 1e-27
        e = 1.60217662e-19
        sigma = avg_slope * e**2 / (6 * V_m3 * kB * self.temperature)  # S/m
        sigma_std = np.std([slope * e**2 / (6 * V_m3 * kB * self.temperature) for slope in slopes])

        # Save conductivity data to file
        avg_msd = np.mean(msd_data, axis=0)
        with open('conductivity_data.txt', 'w') as f:
            f.write('# Time (ps), MSD (e^2·m^2), sigma_avg (S/m), sigma_std (S/m)\n')
            for t, m in zip(times[0], avg_msd):
                f.write(f'{t},{m},{sigma},{sigma_std}\n')

        # Save individual trajectory MSDs for reference
        with open('conductivity_trajectories.txt', 'w') as f:
            f.write('# Time (ps), MSD_traj_0 (e^2·m^2), MSD_traj_1 (e^2·m^2), ...\n')
            max_len = max(len(t) for t in times)
            for i in range(max_len):
                row = [times[0][i] if i < len(times[0]) else times[0][-1]]
                for msd_traj in msd_data:
                    row.append(msd_traj[i] if i < len(msd_traj) else msd_traj[-1])
                f.write(','.join(map(str, row)) + '\n')

        return sigma, sigma_std

    def plot_msd_diffusion(self, data_file='diffusion_data.txt', traj_file='diffusion_trajectories.txt'):
        # Load data
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        times, avg_msd, D_avg, D_std = data[:,0], data[:,1], data[0,2], data[0,3]

        traj_data = np.loadtxt(traj_file, delimiter=',', skiprows=1)
        traj_times = traj_data[:,0]
        traj_msds = traj_data[:,1:]

        plt.figure(figsize=(6, 4))
        for i in range(traj_msds.shape[1]):
            plt.plot(traj_times, traj_msds[:,i], lw=1, alpha=0.3, color='gray', label='Trajectories' if i == 0 else None)
        plt.plot(times, avg_msd, lw=2, color='blue', label='Average MSD')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('MSD (m²)', fontsize=12)
        plt.title(f'Diffusion MSD (D = {D_avg:.2f} ± {D_std:.2f} × 10⁻⁹ m²/s)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_acf_viscosity(self, data_file='viscosity_data.txt'):
        # Load data
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        times, acf_data, eta_avg, eta_std = data[:,0], data[:,1], data[0,2], data[0,3]

        plt.figure(figsize=(6, 4))
        plt.plot(times, acf_data, lw=2, color='green')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Pressure Tensor ACF (Pa²)', fontsize=12)
        plt.title(f'Viscosity ACF (η = {eta_avg:.2f} ± {eta_std:.2f} mPa·s)', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_msd_conductivity(self, data_file='conductivity_data.txt', traj_file='conductivity_trajectories.txt'):
        # Load data
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        times, avg_msd, sigma_avg, sigma_std = data[:,0], data[:,1], data[0,2], data[0,3]

        traj_data = np.loadtxt(traj_file, delimiter=',', skiprows=1)
        traj_times = traj_data[:,0]
        traj_msds = traj_data[:,1:]

        plt.figure(figsize=(6, 4))
        for i in range(traj_msds.shape[1]):
            plt.plot(traj_times, traj_msds[:,i], lw=1, alpha=0.3, color='gray', label='Trajectories' if i == 0 else None)
        plt.plot(times, avg_msd, lw=2, color='red', label='Average MSD')
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Dipole Moment MSD (e²·m²)', fontsize=12)
        plt.title(f'Conductivity MSD (σ = {sigma_avg:.2f} ± {sigma_std:.2f} S/m)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()