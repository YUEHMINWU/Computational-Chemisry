from DiffCaptAnlys_dat import *

electrode_total_charge_files = [
#    '../sim_output_v0_ns500/energy_v0.log',
    '../sim_output_v0.5_ns1000/energy_v0.5.log',
    '../sim_output_v1.5_ns1000/energy_v1.5.log',
    '../sim_output_v2_ns1000/energy_v2.log',
#    '../sim_output_v2.5_ns500/energy_v2.5.log',
#    '../sim_output_v3_ns500/energy_v3.log',
    '../sim_output_v3.5_ns1000/energy_v3.5.log',
    '../sim_output_v4_ns1000/energy_v4.log',

]


# Corresponding voltage levels for each file
voltage_levels = [0.5, 1.5, 2, 3.5, 4]
#voltage_levels = [0, 2, 4]
# voltage_levels = [2, 3.5, 4]
# voltage_levels = [4]

Startframe = 5000
Endframe = 9999 # count all charges (lines) before endframe iteration, ex Endframe = 9999 -> counting lines is 999800 not 1000000  

# Initialize dictionaries to store results for cathode and anode charges
cathode_charges = {}
anode_charges = {}

for file_path, voltage_level in zip(electrode_total_charge_files, voltage_levels):
    etc_ns = ETC([file_path], [voltage_level], Startframe, Endframe)
    
    # Calculate and store cathode charge
    average_total_cathode_charge = etc_ns.cal_ETC_2done()
    cathode_charges[voltage_level] = average_total_cathode_charge[0]
    
    # Calculate and store anode charge
    average_total_anode_charge = etc_ns.cal_ETC_2done()
    anode_charges[voltage_level] = average_total_anode_charge[1]

# Combine the cathode and anode results into a single dictionary
SC_dict = {'v{}_cat'.format(v): cathode_charges[v] for v in voltage_levels}
SC_dict.update({'v{}_an'.format(v): anode_charges[v] for v in voltage_levels})

print(SC_dict)
