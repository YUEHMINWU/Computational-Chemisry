import MDAnalysis as mda
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

############################ System Total Charge Class ##############################
class STC:
    def __init__(self, u, Startframe, Endframe, pdb, pdb_bd, forcefield, num_bins, solution_list, hist_range):
        self.u= u
        self.Startframe= Startframe
        self.Endframe = Endframe
        self.pdb= pdb
        self.pdb_bd= pdb_bd
        self.forcefield= forcefield
        self.num_bins= num_bins
        self.solution_list= solution_list
        self.hist_range= hist_range
        
    def cal_cd(self):
        pdb = PDBFile(self.pdb)
        pdb.topology.loadBondDefinitions(self.pdb_bd[0])
        pdb.topology.loadBondDefinitions(self.pdb_bd[1])
        pdb.topology.loadBondDefinitions(self.pdb_bd[2])
        pdb.topology.loadBondDefinitions(self.pdb_bd[3])
        pdb.topology.createStandardBonds()
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # forcefield = ForceField('ffdir/graph_c_freeze.xml','ffdir/graph_n_freeze.xml', 'ffdir/graph_s_freeze.xml','ffdir/sapt_noDB_2sheets.xml')
        # modeller.addExtraParticles(forcefield)
        
        # Load forcefield from multiple files
        forcefield = ForceField(*self.forcefield)

        # Add extra particles to the modeller
        modeller.addExtraParticles(forcefield)
        
        system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, constraints=None, rigidWater=True)
        nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
        # this is not important, this is just to create openmm object that we can use to access topology
        integ_md = DrudeLangevinIntegrator(300, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
        platform = Platform.getPlatformByName('CPU')
        simmd = Simulation(modeller.topology, system, integ_md, platform)
        
        vec_x = self.u.trajectory[self.Startframe].triclinic_dimensions[0]
        vec_y = self.u.trajectory[self.Startframe].triclinic_dimensions[1]
        area = LA.norm( np.cross(vec_x, vec_y) )

        solution_list= self.solution_list
        num_bins = self.num_bins
        accumulated_total_charge=np.zeros(num_bins)

        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            positions = []
            charge_data = []

            for solutions in solution_list:
                z_coord = solutions.positions[:, 2]
                positions.extend(z_coord)

                # Find the charge data corresponding to the current component
                component_charge_data = []
                for atom in solutions.atoms:
                    charge, _, _ = nbondedForce.getParticleParameters(atom.index)
                    component_charge_data.append(charge._value)

                charge_data.extend(component_charge_data)

            # Now, 'positions' contains all positions, and 'charge_data' contains corresponding charge data
            # hist_pos, bins_pos = np.histogram(positions, bins=num_bins)
            # You can use 'charge_data' as needed, for example, to accumulate charge in each bin
            total_charge_in_bins = np.histogram(positions, bins=num_bins, range=self.hist_range, weights=charge_data)[0]
            accumulated_total_charge += total_charge_in_bins
        sys_total_charge= accumulated_total_charge / (self.Endframe-self.Startframe)
        return sys_total_charge

############################## Charge density Plot Class ###################################
class NTC:
    def __init__(self, filename):
        self.filename = filename
    
    def NTCV_list(self):
        NTCV_list = []
        with open(self.filename, 'r') as NTC_data:
            for line in NTC_data:
                value = float(line.strip())
                NTCV_list.append(value)
        return NTCV_list

class CDP:
    def __init__(self, left_data, right_data, bin_width, x_0, x_l, y_0, y_l):
        self.left_data = left_data
        self.right_data = right_data
        self.bin_width = bin_width
        self.x_0 = x_0
        self.x_l = x_l
        self.y_0 = y_0
        self.y_l = y_l

    def plot(self, title, voltages, linestyles, colors):
        plt.figure(figsize=(12, 4))
        
        # Create the left subplot
        left_subplot = plt.subplot(1, 2, 1)
        left_subplot.set_xlabel("Distance from the cathode ($\AA$)")
        left_subplot.set_ylabel("Charge $ (e/(nm^{3}))$ ")
        left_subplot.axhline(0, color='black', linestyle='--', lw=0.5)
        left_subplot.set_title(title)
        left_subplot.spines['right'].set_visible(False)
        left_subplot.spines['top'].set_visible(False)
        left_subplot.set_ylim(self.y_0, self.y_l)
        left_subplot.set_xlim(self.x_0, self.x_l)

        # Create the right subplot
        right_subplot = plt.subplot(1, 2, 2)
        right_subplot.set_xlabel("Distance from the anode ($\AA$)")
        right_subplot.set_ylabel("Charge $ (e/(nm^{3}))$")
        right_subplot.axhline(0, color='black', linestyle='--', lw=0.5)
        right_subplot.set_title(title)
        right_subplot.spines['left'].set_visible(False)
        right_subplot.spines['top'].set_visible(False)
        right_subplot.set_ylim(self.y_0, self.y_l)
        right_subplot.set_xlim(self.x_0, self.x_l)
        
        for densities, subplot in [(self.left_data, left_subplot), (self.right_data, right_subplot)]:
            for i in range(len(voltages)):
                subplot.plot(
                    np.arange(0, len(densities[i])) * self.bin_width,
                    densities[i],
                    lw="1.0",
                    label=f'{voltages[i]/2}/-{voltages[i]/2} V',
                    linestyle=linestyles[i],
                    color=colors[i]
                )

        plt.gca().invert_xaxis()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

        # Add legends
        left_subplot.legend(fontsize=10, frameon=False, ncol=2)
        # right_subplot.legend(frameon=False)
        plt.text(0.5, 0.5, '//',fontsize=15, horizontalalignment='center', verticalalignment='center',transform=plt.gcf().transFigure)

        # Show the plots
        plt.tight_layout()
        plt.show()

class CDP_oneside:
    def __init__(self, bin_width, dmax, ylim):
        self.bin_width = bin_width
        self.dmax = dmax
        self.ylim = ylim


    def w_m_plots(self, NTCD_list, voltages, colors, linestyles, y_nticks):
        fig, ax = plt.subplots(figsize=(5.5,5)) 

        for k in range(len(voltages)):
            # Assign the labels based on voltages
            labels = f'0 / 0 V' if voltages[k] == 0 else f'-{int(voltages[k]/2)} / {int(voltages[k]/2)} V'
            
            # Plot the data for the corresponding voltage
            plt.plot(np.arange(0, len(NTCD_list[k])) * self.bin_width,
                     NTCD_list[k], label=labels, color=colors[k], linestyle=linestyles[k], lw=1.25)

    
        plt.xlim(0.0, self.dmax)
        plt.ylim(-self.ylim, self.ylim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.5))   
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.ylim/y_nticks))
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.axhline(y=0, color='gray', linewidth=1)
        plt.axvline(x=0, color='lightgray', linewidth=20)
        plt.xlabel("Distance from electrode ($\\AA$)", fontsize = 12)
        plt.ylabel("Charge $ (e/nm^{3})$", fontsize = 12)
        plt.legend(frameon=False, fontsize=12)
        plt.show()
        
        

################################# Electrode Total Charge ###################################33
class ETC:
    def __init__(self, electrode_total_charge_files, voltage_levels, Startframe, Endframe):
        self.electrode_total_charge_files = electrode_total_charge_files
        self.voltage_levels = voltage_levels
        self.Startframe = Startframe
        self.Endframe = Endframe

    def cal_ETC_2done(self): # For calulating the file simulation is completed.  
        for i, file_path in enumerate(self.electrode_total_charge_files):
            total_charge_sum_cat = 0.0
            line_count_cat = 0
            total_charge_sum_anod = 0.0
            line_count_anod = 0
            start_processing = False
            done_found = False
            with open(file_path, 'r') as file:
                for line in file:
                    start = f'{self.Startframe} iteration'
                    end = f'{self.Endframe} iteration'
                                            
                    if start in line:
                        start_processing = True
                        continue

                    if start_processing:
                        if 'cathode' in line:
                            parts = line.split()
                            if len(parts) >= 5 and parts[0] == "Q_numeric" and parts[2] == "Q_analytic":
                                Q_numeric = float(parts[-1])
                                total_charge_sum_cat += Q_numeric
                                line_count_cat += 1

                        if 'anode' in line:
                            parts = line.split()
                            if len(parts) >= 5 and parts[0] == "Q_numeric" and parts[2] == "Q_analytic":
                                Q_numeric = float(parts[-1])
                                total_charge_sum_anod += Q_numeric
                                line_count_anod += 1
                                
                        if 'done!' in line:
                            done_found = True
                            break
                            
                        
            if line_count_cat > 0:
                average_total_cathode_charge = total_charge_sum_cat / line_count_cat
                average_total_anode_charge = total_charge_sum_anod / line_count_anod
                return [average_total_cathode_charge, average_total_anode_charge]
            else:
                print(f"No valid lines found in the file: {file_path}")
                return None

    def cal_ETC_2endframe(self): # For calulating the file simulation is interupted. 
        for i, file_path in enumerate(self.electrode_total_charge_files):
            total_charge_sum_cat = 0.0
            line_count_cat = 0
            total_charge_sum_anod = 0.0
            line_count_anod = 0
            start_processing = False
            done_found = False
            with open(file_path, 'r') as file:
                for line in file:
                    start = f'{self.Startframe} iteration'
                    end = f'{self.Endframe} iteration'

                    if 'done!' in line:
                        done_found = True
                        print('done found')
                                            
                    if start in line:
                        start_processing = True
                        continue

                    if start_processing:
                        if 'cathode' in line:
                            parts = line.split()
                            if len(parts) >= 5 and parts[0] == "Q_numeric" and parts[2] == "Q_analytic":
                                Q_numeric = float(parts[-1])
                                total_charge_sum_cat += Q_numeric
                                line_count_cat += 1

                        if 'anode' in line:
                            parts = line.split()
                            if len(parts) >= 5 and parts[0] == "Q_numeric" and parts[2] == "Q_analytic":
                                Q_numeric = float(parts[-1])
                                total_charge_sum_anod += Q_numeric
                                line_count_anod += 1 
                                
                        # Stop if Endframe is reached, but only if done was not found
                        elif end in line:
                            if not done_found:
                                break  # Stop processing at the Endframe
                            
            if line_count_cat > 0:
                average_total_cathode_charge = total_charge_sum_cat / line_count_cat
                average_total_anode_charge = total_charge_sum_anod / line_count_anod
                return [average_total_cathode_charge, average_total_anode_charge]
            else:
                print(f"No valid lines found in the file: {file_path}")
                return None

            
###################### Possion Potential Plots Class ###########################
class NTC:
    def __init__(self, filename):
        self.filename = filename
    
    def NTCV_list(self):
        NTCV_list = []
        with open(self.filename, 'r') as NTC_data:
            for line in NTC_data:
                value = float(line.strip())
                NTCV_list.append(value)
        return NTCV_list

class pospot:
    def __init__(self, u, dz, L_all, V_app, Q_cat, V_init, NTC, electrode_z_positions, num_bins, bin_width):
        self.u = u
        self.dz = dz
        self.L_all = L_all
        self.V_app = V_app
        self.Q_cat = Q_cat
        self.V_init = V_init
        self.NTC = NTC
        self.electrode_z_positions = electrode_z_positions
        self.bin_width = bin_width
        self.num_bins = num_bins
        self.rho_sys = None
        self.E_sys = None
        self.V_sys = None

    def calculate_pospot(self):
        vec_x = self.u.trajectory[0].triclinic_dimensions[0]
        vec_y = self.u.trajectory[0].triclinic_dimensions[1]
        area = LA.norm(np.cross(vec_x, vec_y))
        Å_to_bohradi = 1 / 0.529177
        nm2bohr = 18.8973
        conv = 1/(nm2bohr**3)
        eV_to_hartree = 1 / 27.21138602
        dz_bohr = self.dz * Å_to_bohradi
        L_gap = self.L_all - (max(self.electrode_z_positions) - min(self.electrode_z_positions))
        E_surface = [4 * np.pi * Q_cat / ((area) * (Å_to_bohradi** 2) ) for Q_cat in self.Q_cat]
        E_gap = [-(V_app * eV_to_hartree / (L_gap * Å_to_bohradi)) for V_app in self.V_app]
        E_right = [float(E_surface[i] + E_gap[i]) for i in range(len(self.V_app))]
        
        rho_sys = [[] for _ in range(len(self.V_app))]
        for i in range(len(self.NTC)):
            NTC = self.NTC[i]
            rho_sys[i].extend(np.array(NTC) * 1000 / (area * self.bin_width))
        
        self.rho_sys = rho_sys

        E_sys = [[] for _ in range(len(self.V_app))]
        for i in range(len(E_right)):
            rho_temp = self.NTC[i]
            for j in range(len(self.NTC[0])):
                E_right[i] += 4 * np.pi * (rho_temp[j] / (area * self.bin_width * (Å_to_bohradi ** 3))) * dz_bohr
                E_sys[i].append(E_right[i])

        self.E_sys= E_sys

        V_sys = [[] for _ in range(len(self.V_init))]
        V_init = [(V_i) * eV_to_hartree for V_i in self.V_init]

        for i in range(len(self.V_init)):
            V_sys[i].append(self.V_init[i])
        for i in range(len(self.V_app)):
            V_init_temp = V_init[i]
            E_sys_temp = E_sys[i]
            for j in range(len(E_sys[0])):
                V_init_temp -= E_sys_temp[j] * dz_bohr
                V_sys[i].append(V_init_temp / eV_to_hartree)

        self.V_sys = V_sys        
        return rho_sys, E_sys, V_sys
    
    def cal_V_drop(self):
        if self.V_sys is None:
            self.calculate_pospot()
        delta_V_cat = [(V_sys[0] - (np.sum(V_sys[(self.num_bins // 2 - 100):(self.num_bins // 2 + 100)]) / 200)) for V_sys in self.V_sys]
        delta_V_anode = [(V_sys[-1] - (np.sum(V_sys[(self.num_bins // 2 - 100):(self.num_bins // 2 + 100)]) / 200)) for V_sys in self.V_sys]
        V_cat = [V_sys[0] for V_sys in self.V_sys]
        V_anode = [V_sys[-1] for V_sys in self.V_sys]
        V_mid = [V_sys[self.num_bins // 2] for V_sys in self.V_sys]
        V_drop_cat = delta_V_cat
        V_drop_anode = delta_V_anode
        return V_cat, V_anode, V_mid, V_drop_cat, V_drop_anode

    def write_pospot(self, V_app, ns):
        if self.V_sys is None:
            self.calculate_pospot()
            
        # Write rho, Efield, and pospot data for each voltage level
        for i, voltage in enumerate(V_app):
            out_path = f'../sim_output_v{voltage}_ns{ns}/AnaRes/'
            os.makedirs(out_path, exist_ok=True)
            
            # Write rho data - one value per line
            with open(os.path.join(out_path, 'rho.dat'), "w") as file:
                for bin_value in self.rho_sys[i]:
                    file.write(f"{bin_value}\n")
                
            # Write Efield data - one value per line
            with open(os.path.join(out_path, 'Efield.dat'), "w") as file:
                for bin_value in self.E_sys[i]:
                    file.write(f"{bin_value}\n")
                
            # Write pospot data - one value per line
            with open(os.path.join(out_path, 'pospot.dat'), "w") as file:
                for bin_value in self.V_sys[i]:
                    file.write(f"{bin_value}\n")

def read_data_file(filepath):
    """Read data from file and convert to numpy array"""
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

def plot_pospot(fp_cd, fp_Efield, fp_pospot, V_init, bin_width, colors, linestyles):
    # Read all data files
    cd_data = [read_data_file(fp) for fp in fp_cd]
    Efield_data = [read_data_file(fp) for fp in fp_Efield]
    pospot_data = [read_data_file(fp) for fp in fp_pospot]
    
    # Plot charge density
    plt.figure(figsize=(8, 5))
    for i in range(len(cd_data)):
        labels = f'0 / 0 V' if V_init[i] == 0 else f'-{V_init[i]} / {V_init[i]} V'
        plt.plot(np.arange(0, len(cd_data[i])) * bin_width, cd_data[i],
                label=labels, color=colors[i], linestyle=linestyles[i], lw=1.2)
    plt.xlabel("Distance from electrode ($\AA$)")
    plt.ylabel("Charge $ (e/nm^{3})$ ")
    plt.axhline(0, color='black', linestyle='--', lw=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, max(np.arange(0, len(cd_data[0])) * bin_width))
    plt.title('Charge Density Profile')
    plt.legend(fontsize=12, frameon=False)
    
    # Plot electric field
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(Efield_data)):
        labels = f'0 / 0 V' if V_init[i] == 0 else f'-{V_init[i]} / {V_init[i]} V'
        plt.plot(np.arange(0, len(Efield_data[i])) * bin_width, Efield_data[i],
                label=labels, color=colors[i], linestyle=linestyles[i], lw=1.2)
    plt.xlabel("Distance from electrode ($\AA$)")
    plt.ylabel('Electric Field ($e/a_0^2$)')
    plt.axhline(0, color='black', linestyle='--', lw=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, max(np.arange(0, len(Efield_data[0])) * bin_width))
    plt.title('Electric Field Profile')
    plt.legend(fontsize=12, frameon=False)
    
    # Plot Poisson potential
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(pospot_data)):
        labels = f'0 / 0 V' if V_init[i] == 0 else f'-{V_init[i]} / {V_init[i]} V'
        plt.plot(np.arange(0, len(pospot_data[i])) * bin_width, pospot_data[i],
                label=labels, color=colors[i], linestyle=linestyles[i], lw=1.5)
    plt.xlabel('Distance from electrode (Å)', fontsize=12)
    plt.ylabel('Poisson Potential (V)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(0, color='black', linestyle='--', lw=0.5)
    plt.xlim(0, max(np.arange(0, len(pospot_data[0])) * bin_width))
    plt.ylim(-V_init[-1], V_init[-1])
    plt.legend(fontsize=12, frameon=False, loc='lower center')
    plt.show()
############################### Differential Capacitances Plot Class #############################
class dCP:
    def __init__(self, area, V_app, voltages_drops_pos, voltages_drops_neg, Q_cathode, Q_anode):
        self.area = area
        self.V_app = V_app
        self.voltages_drops_pos = voltages_drops_pos
        self.voltages_drops_neg = voltages_drops_neg
        self.Q_cathode = Q_cathode
        self.Q_anode = Q_anode


    def cal_Q_surf(self):
        Vcat = self.voltages_drops_pos
        Vcat = [Vcat[i] - Vcat[0] for i in range(len(Vcat))]
        Van = self.voltages_drops_neg
        Van = [Van[i] - Van[0] for i in range(len(Van))]
        return [Vcat, Van]

    def cal_dCD(self, system_name, output_dir):
        e_to_Coulomb= 1.602176634e-19
        angstrom_to_cm= 1e-8
        conv = e_to_Coulomb/(self.area*angstrom_to_cm*angstrom_to_cm)/(1e-6)
        
        Vcat = self.voltages_drops_pos
        Vcat= [Vcat[i]-Vcat[0] for i in range(len(Vcat))]
        
        Van = self.voltages_drops_neg
        Van= [Van[i]-Van[0] for i in range(len(Van))]
        
        cEDLcat = [(self.Q_cathode[i] - self.Q_cathode[i-2])/(Vcat[i]-Vcat[i-2]) for i in range(2, len(Vcat))]
        cEDLcat_m = [(cEDLcat[i]+cEDLcat[i+1])*0.5 for i in range(0, len(cEDLcat), 2)]
        
        cEDLan = [(self.Q_anode[i] - self.Q_anode[i-2])/(Van[i]-Van[i-2]) for i in range(2, len(Van))]
        cEDLan_m = [(cEDLan[i]+cEDLan[i+1])*0.5 for i in range(0, len(cEDLan), 2)]
        
        cEDL0V = self.Q_cathode[1]-(self.Q_anode[1])/(Vcat[1]-Van[1])
        
        cEDLcat_SI = [(cEDLcat[i]*conv) for i in range(len(cEDLcat))]
        cEDL0V_SI = [float(cEDL0V*conv)]
        cEDLan_SI = [(cEDLan[i]*conv) for i in range(len(cEDLan))]
        cEDLcat_SI_m = [(cEDLcat_m[i]*conv) for i in range(len(cEDLcat_m))]
        cEDL0V_SI_m = [float(cEDL0V*conv)]
        cEDLan_SI_m = [(cEDLan_m[i]*conv) for i in range(len(cEDLan_m))]
        
        vEDL = [(Van[i-2]+Van[i])*0.5 for i in range(2,len(Van))][::-1]+[(Vcat[1]+Van[1])*0.5]+[(Vcat[i-2]+Vcat[i])*0.5 for i in range(2,len(Vcat))]
        
        vEDL_m = [(vEDL[i]+vEDL[i+1])*0.5 for i in range(0,len(vEDL),2) if i < int(len(vEDL)/2)]+[(Vcat[1]+Van[1])*0.5] + [(vEDL[i]+vEDL[i+1])*0.5 for i in range(5,len(vEDL),2) if i > int(len(vEDL)/2)]
        cEDL = cEDLan_SI[::-1] + cEDL0V_SI+ cEDLcat_SI
        cEDL_m = cEDLan_SI_m[::-1] + cEDL0V_SI_m+ cEDLcat_SI_m
        
        cEDLcat_std= [np.std([cEDLcat[i]*conv,cEDLcat[i+1]*conv]) for i in range(0,len(cEDLcat),2)]
        cEDL0V_std = [np.std(cEDL0V*conv)]
        cEDLan_std= [np.std([cEDLan[i]*conv,cEDLan[i+1]*conv]) for i in range(0,len(cEDLan),2)]
        cEDL_std = cEDLan_std[::-1] + cEDL0V_std + cEDLcat_std

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f'{system_name} C_D.log')
        with open(filename, 'w') as f:
            f.write(f"V_EDL: {vEDL}\n")
            f.write(f"C_D: {cEDL}\n")
        

        filename_std = os.path.join(output_dir, f'{system_name} C_D_std.log')
        with open(filename_std, 'w') as f:
            f.write(f"V_EDL: {vEDL_m}\n")
            f.write(f"C_D: {cEDL_m}\n")
            f.write(f"C_D_std: {cEDL_std}\n")

    
def plot_Q_surf(systems, system_names, colors):
    plt.figure(figsize=(5, 5))
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.ylabel("Q$_{electrode}$", fontsize=14)
    
    for i, system in enumerate(systems):
        Q_surfs = system.cal_Q_surf()
        plt.plot(Q_surfs[0], system.Q_cathode, marker='D', label=system_names[i], color= colors[i])
        plt.plot(Q_surfs[1], system.Q_anode, marker='v', color= colors[i])
    
    plt.legend(loc='upper left', fontsize=14, frameon=False)
    plt.show()
    

def plot_dCD(systems, filenames, system_names, sys_name, output_dir, markers, colors, ylim_l, ylim_u):
    for i, system in enumerate(systems):
        system.cal_dCD(sys_name[i], output_dir)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL = eval(lines[0].split(":")[1].strip())
            cEDL = eval(lines[1].split(":")[1].strip())

        plt.plot(vEDL, cEDL, color= colors[i], marker=markers[i], label=system_names[i])
        

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=14, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()
    
def plot_dCD_std(systems, filenames, system_names, sys_name, output_dir, fmts, colors, ylim_l, ylim_u):
    for i, system in enumerate(systems):
        dCDs = system.cal_dCD(sys_name[i], output_dir)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL_m = eval(lines[0].split(":")[1].strip())
            cEDL_m = eval(lines[1].split(":")[1].strip())
            cEDL_std = eval(lines[2].split(":")[1].strip())

        plt.errorbar(vEDL_m, cEDL_m, yerr=cEDL_std, fmt=fmts[i], ecolor=colors[i], capsize=5, label=system_names[i])

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=14, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()

def plot_dCD(systems, filenames, system_names, sys_name, output_dir, markers, colors, ylim_l, ylim_u):
    for i, system in enumerate(systems):
        system.cal_dCD(sys_name[i], output_dir)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL = eval(lines[0].split(":")[1].strip())
            cEDL = eval(lines[1].split(":")[1].strip())

        plt.plot(vEDL, cEDL, color= colors[i], marker=markers[i], label=system_names[i])
        

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=14, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()
    
def plot_dCD_std(systems, filenames, system_names, sys_name, output_dir, fmts, colors, ylim_l, ylim_u):
    for i, system in enumerate(systems):
        dCDs = system.cal_dCD(sys_name[i], output_dir)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL_m = eval(lines[0].split(":")[1].strip())
            cEDL_m = eval(lines[1].split(":")[1].strip())
            cEDL_std = eval(lines[2].split(":")[1].strip())

        plt.errorbar(vEDL_m, cEDL_m, yerr=cEDL_std, fmt=fmts[i], ecolor=colors[i], capsize=5, label=system_names[i])

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=14, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()

def plot_dCD_from_file(filenames, system_names, markers, colors, ylim_l, ylim_u):
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL = eval(lines[0].split(":")[1].strip())
            cEDL = eval(lines[1].split(":")[1].strip())

        plt.plot(vEDL, cEDL, color= colors[i], marker=markers[i], label=system_names[i])

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()

def plot_dCD_std_from_file(filenames, system_names, fmts, colors, ylim_l, ylim_u):
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
            vEDL_m = eval(lines[0].split(":")[1].strip())
            cEDL_m = eval(lines[1].split(":")[1].strip())
            cEDL_std = eval(lines[2].split(":")[1].strip())

        plt.errorbar(vEDL_m, cEDL_m, yerr=cEDL_std, fmt=fmts[i], ecolor=colors[i], capsize=5, label=system_names[i])

    plt.xticks(np.arange(-4, 4, 0.5), fontsize=14)
    plt.xlim([-2.5, 2.5])
    plt.ylim([ylim_l, ylim_u])
    plt.xlabel("$\Delta$V$_{electrode}$ (vs. PZC)", fontsize=14)
    plt.yticks(np.arange(ylim_l, ylim_u, (ylim_u - ylim_l) / 10), fontsize=14)
    plt.ylabel("C$_D$ ($\mu$F/cm$^2$)", fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.axvline(x=0, color='lightgray', linewidth=1)
    plt.show()
