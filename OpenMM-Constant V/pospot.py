from DiffCaptAnlys_dat import *

pdb_v0 = '../sim_output_v0_ns500/start_drudes.pdb'
dcd_v0 = '../sim_output_v0_ns500/FV_NVT.dcd'
u= mda.Universe(pdb_v0, dcd_v0)
grp_A_list = u.select_atoms('resname grp and segid A and name C1')
grp_B_list = u.select_atoms('resname grp and segid B and name C1')
grp_A_pos = grp_A_list.positions[:, 2]
grp_B_pos = grp_B_list.positions[:, 2]
electrode_z_positions= (grp_A_pos, grp_B_pos)

# num_bins=120
Lcell = max(grp_B_pos) - max(grp_A_pos)

bin_width = 0.1
# bin_width = Lcell/num_bins

n_bins = int(Lcell/bin_width)

# hist_range = (min(grp_B_pos) - dmax , min(grp_B_pos))
dz = bin_width
L_all = 300.422
V_app = [0.5, 1.5, 2, 3.5, 4]
# V_app = [0, 2, 4]
# V_app = [2, 3.5, 4]
# V_app = [4]
ns = 1000

SC_dict=  {'v0.5_cat': 0.9540805456819854, 'v1.5_cat': 2.0026758380090977, 'v2_cat': 2.5422177942899786, 'v3.5_cat': 5.3178739259177386, 'v4_cat': 6.177155215270774, 'v0.5_an': -0.9600805456823817, 'v1.5_an': -2.0086758380094527, 'v2_an': -2.5482177942903887, 'v3.5_an': -5.323873925918181, 'v4_an': -6.183155215271373}

Q_cat = [
#    SC_dict['v0_cat'],
    SC_dict['v0.5_cat'],
    SC_dict['v1.5_cat'],
    SC_dict['v2_cat'],
    SC_dict['v3.5_cat'],
    SC_dict['v4_cat']
         
]

V_init = [(V_init_i)/2 for V_init_i in V_app ]
NTCV = [
#    '../sim_output_v0_ns1500/AnaRes/normalized_total_charge.dat',
    '../sim_output_v0.5_ns1000/AnaRes/normalized_total_charge.dat',
    '../sim_output_v1.5_ns1000/AnaRes/normalized_total_charge.dat',
    '../sim_output_v2_ns1000/AnaRes/normalized_total_charge.dat',
#    '../sim_output_dir/200[N1888][TFSI]/sim_output_v2.5_ns500/AnaRes/normalized_total_charge.dat',
#    '../sim_output_dir/200[N1888][TFSI]/sim_output_v3_ns500/AnaRes/normalized_total_charge.dat',
    '../sim_output_v3.5_ns1000/AnaRes/normalized_total_charge.dat',
    '../sim_output_v4_ns1000/AnaRes/normalized_total_charge.dat',

      ]

NTC_list=[]

for i in range(len(NTCV)):
    NTCV_list = NTC(NTCV[i])
    NTCV_lists = NTCV_list.NTCV_list()
    NTC_list.append(NTCV_lists)

pos_pot = pospot(u, dz, L_all, V_app, Q_cat, V_init, NTC_list, electrode_z_positions, n_bins, bin_width)
pos_pot.write_pospot(V_app, ns)

V_drop_dict = {}
for i, V_level in enumerate(V_app):
    pos_pot_i = pospot(u, dz, L_all, [V_level], [Q_cat[i]], [V_init[i]], [NTC_list[i]], electrode_z_positions, n_bins, bin_width)
    V_cat, V_anode, V_mid, V_drop_cat, V_drop_anode = pos_pot_i.cal_V_drop()
    V_drop_dict[f'v{V_level}_cat'] = (V_cat, V_drop_cat, V_mid)
    V_drop_dict[f'v{V_level}_an'] = (V_anode, V_drop_anode, V_mid)

print(V_drop_dict)
