pdb_files = [
	'../MC_sim_output_ns/start_drudes.pdb'
]
dcd_files = [
	'../MC_sim_output_ns/MC_equil.dcd'
]

u_list = [mda.Universe(pdb, dcd) for pdb, dcd in zip(pdb_files, dcd_files)]

OCT_lists= [u.select_atoms('resname OCT ') for u in u_list]
OTF_lists = [u.select_atoms('resname tf2 ') for u in u_list]

grp_AB_list = [u.select_atoms('resname grp and segid A') | u.select_atoms('resname grp and segid B') for u in u_list]
electrode_z_positions_list = [np.array(grp.positions[:, 2]) for grp in grp_AB_list]

hist_range_list = [(min(positions), max(positions)) for positions in electrode_z_positions_list]
bin_width = 0.1
num_bins = (max(hist_range_list[0])-min(hist_range_list[0]))/bin_width

Startframe=
Endframe=

ddp_list= []

# Loop through components
for i, component in enumerate([OCT_lists, OTF_lists]):
    for j, u in enumerate(u_list):
        ddp = SDD(u, Startframe, Endframe, component[j], hist_range_list[j], num_bins)
        ddp_list = ddp.cal_dens()

	file_name = f'ILs Density Profile.dat'
	out_path = f'../MC_sim_output_ns/'
	output_file = os.path.join(out_path, file_name)

        with open(output_file, "w") as file:
            for value in ddp_list:
                file.write(f"{value}\n")

print(done!)
