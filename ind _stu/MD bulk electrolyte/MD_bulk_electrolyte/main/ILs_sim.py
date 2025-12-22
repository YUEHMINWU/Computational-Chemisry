import openmm as mm
from openmm import app
from openmm.unit import *
import pandas as pd
import multiprocessing as mp

class IonicLiquidSimulator:
    def __init__(self, pdb_file, forcefield_file, temperature=300*kelvin, pressure=1*atmospheres, timestep=0.002*picoseconds, report_interval=10000):
        self.pdb_file = pdb_file
        self.forcefield_file = forcefield_file
        self.temperature = temperature
        self.pressure = pressure
        self.timestep = timestep
        self.report_interval = report_interval
        self.pdb = app.PDBFile(pdb_file)
        self.forcefield = app.ForceField(forcefield_file)

    def run_npt_equilibration(self, steps=1000000):
        system = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds)
        barostat = mm.MonteCarloBarostat(self.pressure, self.temperature, 25)
        system.addForce(barostat)
        integrator = mm.LangevinMiddleIntegrator(self.temperature, 1/picosecond, self.timestep)
        simulation = app.Simulation(self.pdb.topology, system, integrator)
        simulation.context.setPositions(self.pdb.positions)
        print('Minimizing energy...')
        simulation.minimizeEnergy()
        simulation.reporters.append(app.StateDataReporter('npt_data.csv', self.report_interval, step=True, volume=True, density=True))
        print('Equilibrating...')
        simulation.context.setVelocitiesToTemperature(self.temperature)
        simulation.step(steps)
        state = simulation.context.getState(getPositions=True, getVelocities=True, getPeriodicBoxVectors=True)
        with open('equilibrated_state.xml', 'w') as f:
            f.write(mm.XmlSerializer.serialize(state))
        data = pd.read_csv('npt_data.csv')
        avg_density = data['Density (g/mL)'].mean()
        avg_volume = data['Box Volume (nm^3)'].mean()
        print(f'Average density: {avg_density} g/cm^3')
        return avg_density, avg_volume

    class PressureAndDipoleReporter:
        def __init__(self, file_pressure, file_dipole, reportInterval, charges):
            self.file_pressure = open(file_pressure, 'w')
            self.file_dipole = open(file_dipole, 'w')
            self.reportInterval = reportInterval
            self.charges = charges
            self.file_pressure.write('# Time (ps), P_xy (bar), P_xz (bar), P_yz (bar)\n')
            self.file_dipole.write('# Time (ps), M_x (e*nm), M_y (e*nm), M_z (e*nm)\n')

        def describeNextReport(self, simulation):
            steps = self.reportInterval - simulation.currentStep % self.reportInterval
            return (steps, True, False, False, True, None)

        def report(self, simulation, state):
            time = state.getTime().value_in_unit(picosecond)
            positions = state.getPositions().value_in_unit(nanometer)
            velocities = state.getVelocities().value_in_unit(nanometer / picosecond)
            box_vectors = state.getPeriodicBoxVectors()
            volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2] * (nanometer**3)
            masses = [simulation.system.getParticleMass(i).value_in_unit(dalton) for i in range(simulation.system.getNumParticles())]

            K = np.zeros((3,3))
            for i in range(len(masses)):
                v = velocities[i]
                m = masses[i]
                for alpha in range(3):
                    for beta in range(3):
                        K[alpha, beta] += m * v[alpha] * v[beta]
            K *= (dalton * (nanometer / picosecond)**2 / 1000)

            W = state.getVirial().value_in_unit(kilojoule_per_mole)
            V = volume.value_in_unit(nanometer**3)
            P = (K + W) / V * 166.0539

            P_xy = P[0,1]
            P_xz = P[0,2]
            P_yz = P[1,2]
            self.file_pressure.write(f"{time},{P_xy},{P_xz},{P_yz}\n")

            M = np.zeros(3)
            for i in range(len(self.charges)):
                q = self.charges[i]
                r = positions[i]
                M += q * r
            self.file_dipole.write(f"{time},{M[0]},{M[1]},{M[2]}\n")

            self.file_pressure.flush()
            self.file_dipole.flush()

        def __del__(self):
            self.file_pressure.close()
            self.file_dipole.close()

    def run_nvt_simulation(self, equilibrated_state_file, steps, seed):
        system = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds)
        integrator = mm.LangevinMiddleIntegrator(self.temperature, 1/picosecond, self.timestep)
        integrator.setRandomNumberSeed(seed)
        simulation = app.Simulation(self.pdb.topology, system, integrator)

        with open(equilibrated_state_file, 'r') as f:
            state = mm.XmlSerializer.deserialize(f.read())
        simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocitiesToTemperature(self.temperature, seed)

        nonbonded_force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        charges = [nonbonded_force.getParticleParameters(i)[0].value_in_unit(elementary_charge) for i in range(system.getNumParticles())]

        simulation.reporters.append(app.DCDReporter(f'traj_{seed}.dcd', self.report_interval))
        simulation.reporters.append(self.PressureAndDipoleReporter(f'pressure_{seed}.dat', f'dipole_{seed}.dat', self.report_interval, charges))

        simulation.step(steps)

    def run_multiple_nvt_simulations(self, equilibrated_state_file, steps, num_simulations):
        with mp.Pool(processes=num_simulations) as pool:
            pool.starmap(self.run_nvt_simulation, [(equilibrated_state_file, steps, i) for i in range(num_simulations)])