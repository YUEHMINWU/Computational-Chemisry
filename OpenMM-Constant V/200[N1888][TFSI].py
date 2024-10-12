import sys
from DensDistAnlys import *
from DiffCaptAnlys import *
from joblib import Parallel, delayed
import multiprocessing

############################## Whole Molecule density distribution ################################
pdb_files = [
    'sim_output_v0_ns100/start_drudes.pdb',
    'sim_output_v0.5_ns100/start_drudes.pdb',
    'sim_output_v1_ns100/start_drudes.pdb',
    'sim_output_v1.5_ns100/start_drudes.pdb',
    'sim_output_v2_ns100/start_drudes.pdb',
    'sim_output_v2.5_ns100/start_drudes.pdb',
    'sim_output_v3_ns100/start_drudes.pdb',
    'sim_output_v4_ns100/start_drudes.pdb',
]

dcd_files = [
    'sim_output_v0_ns100/FV_NVT.dcd',
    'sim_output_v0.5_ns100/FV_NVT.dcd',
    'sim_output_v1_ns100/FV_NVT.dcd',
    'sim_output_v1.5_ns100/FV_NVT.dcd',
    'sim_output_v2_ns100/FV_NVT.dcd',
    'sim_output_v2.5_ns100/FV_NVT.dcd',
    'sim_output_v3_ns100/FV_NVT.dcd',
    'sim_output_v4_ns100/FV_NVT.dcd',
]

u_list = [mda.Universe(pdb, dcd) for pdb, dcd in zip(pdb_files, dcd_files)]

Startframe = 5000
Endframe = 9999

########################## Density Distribution ################################333
 
#ddp_list= []
#
# Loop through components
#for i, component in enumerate([N113_lists, TFSI_lists]):
#    for j, u in enumerate(u_list):
#        ddp = SDD(u, Startframe, Endframe, component[j], hist_range_list[j], num_bins)
#        ddp_list = ddp.cal_dens()
#
#        file_name = f'{filenames[i]} Density Profile.dat'
#        out_path = f'../sim_output_dir/400[N1113][TFSI]/sim_output_v{voltage_name[j]}_ns100/'
#        output_file = os.path.join(out_path, file_name)
#
#        with open(output_file, "w") as file:
#            for value in ddp_list:
#                file.write(f"{value}\n")
#
#print('Complete Whole Molecule density distribution calculation')

################ Functional Groups density distribution ######################

dmin, dmax = 0.0, 16.0
bin_width = 0.2
n_bins = int(dmax/bin_width)

grp_A_list = [u.select_atoms('resname grp and segid A and name C1') for u in u_list]
grp_B_list = [u.select_atoms('resname grp and segid B and name C1') for u in u_list]
grp_A_pos = grp_A_list[0].positions[:, 2]
grp_B_pos = grp_B_list[0].positions[:, 2]

OCT_Octyls_list = [u.select_atoms('resname OCT and name C* and not name C09') for u in u_list]
OCT_Methyl_list = [u.select_atoms('resname OCT and (name C09 ) ') for u in u_list]

Startframe=5000
Endframe=9999

hist_range_cat = (min(grp_A_pos)  , min(grp_A_pos) + dmax)
hist_range_anod = (min(grp_B_pos) - dmax  , min(grp_B_pos))

voltage_name= [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
filenames= ['OCT_Octyls', 'OCT_Methyl']
ddp_list= []

def compute_FGDDP(i, j, u):
    # TMP_Propyl_list and TMP_Methyls_list should be defined outside this function
    component_list = [OCT_Propyl_list[j], OCT_Methyls_list[j]]  # i corresponds to the list index

    # Calculate for Cathode
    ddp = SDD(u, Startframe, Endframe, component_list[i], hist_range_cat, n_bins)
    ddp_list = ddp.cal_dens()

    # Write the cathode density profile to a file
    file_name = f'{filenames[i]} Density Profile at Cathode.dat'
    out_path = f'sim_output_v{voltage_name[j]}_ns100/AnaRes/'
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, file_name)

    with open(output_file, "w") as file:
        for value in ddp_list:
            file.write(f"{value}\n")

    # Calculate for Anode (flip the density profile)
    ddp_anod = SDD(u, Startframe, Endframe, component_list[i], hist_range_anod, n_bins)
    ddp_list = ddp_anod.cal_dens()
    rev_ddp_list = np.flip(ddp_list)

    file_name = f'{filenames[i]} Density Profile at Anode.dat'
    output_file = os.path.join(out_path, file_name)

    with open(output_file, "w") as file:
        for value in rev_ddp_list:
            file.write(f"{value}\n")

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Parallelize both loops (i and j)
Parallel(n_jobs=num_cores)(
    delayed(compute_FGDDP)(i, j, u) for j, u in enumerate(u_list) for i in range(len(filenames))
)

TFSI_S_list = [u.select_atoms('resname Tf2 and (name Stf or name Stf1) ') for u in u_list]
TFSI_CF3_list = [u.select_atoms('resname Tf2 and (name Ctf or name Ctf1)') for u in u_list]

Startframe=5000
Endframe=9999


voltage_name= [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
filenames= ['TFSI_S', 'TFSI_CF3']
ddp_list= []    

def compute_FGDDP(i, j, u):
    # TMP_Propyl_list and TMP_Methyls_list should be defined outside this function
    component_list = [OTF_S_list[j], OTF_CF3_list[j]]  # i corresponds to the list index
    
    ddp = SDD(u, Startframe, Endframe, component_list[i], hist_range_cat, n_bins)
    ddp_list = ddp.cal_dens()

    # Write the cathode density profile to a file
    file_name = f'{filenames[i]} Density Profile at Cathode.dat'
    out_path = f'sim_output_v{voltage_name[j]}_ns100/AnaRes/'
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, file_name)

    with open(output_file, "w") as file:
        for value in ddp_list:
            file.write(f"{value}\n")

    # Calculate for Anode (flip the density profile)
    ddp_anod = SDD(u, Startframe, Endframe, component_list[i], hist_range_anod, n_bins)
    ddp_list = ddp_anod.cal_dens()
    # Calculate for Anode (flip the density profile)
    rev_ddp_list = np.flip(ddp_list)

    file_name = f'{filenames[i]} Density Profile at Anode.dat'
    output_file = os.path.join(out_path, file_name)

    with open(output_file, "w") as file:
        for value in rev_ddp_list:
            file.write(f"{value}\n")

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Parallelize both loops (i and j)
Parallel(n_jobs=num_cores)(
    delayed(compute_FGDDP)(i, j, u) for j, u in enumerate(u_list) for i in range(len(filenames))
)

print('DDP calculation completed!')

######################## Radial Distribution ###################################

#Select atoms for intermolecular RDF calculation
OCT_Octyls_list = [u.select_atoms('resname OCT and name C* and not name C09') for u in u_list]
OCT_Methyl_list = [u.select_atoms('resname OCT and (name C09 ) ') for u in u_list]
OCT_N_list = [u.select_atoms('resname TMP and (name N08)') for u in u_list]
TFSI_S_list = [u.select_atoms('resname Tf2 and (name Stf or name Stf1) ') for u in u_list]
TFSI_CF3_list = [u.select_atoms('resname Tf2 and (name Ctf or name Ctf1)') for u in u_list]

voltage_name= [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]

atoms_ex_C=('Ctf','Ctf1')
atoms_ex_S=('Stf','Stf1')
atoms_ex_Oct=('C00','C01','C02','C03','C04','C05','C06','C07','C0A','C0B','C0C', 'C0D','C0E','C0F', 'C0G','C0H','C0I', 'C0J','C0K','C0M','C0N','C0O','C0P', 'C0Q') 
atoms_ex_Met=('C09', 'C09')

exclusions_C=(len(atoms_ex_C),len(atoms_ex_C))
exclusions_S=(len(atoms_ex_S),len(atoms_ex_S))
# exclusions_N=(len(atoms_ex_N),len(atoms_ex_N))
exclusions_Oct=(len(atoms_ex_Oct),len(atoms_ex_Oct))
exclusions_Met=(len(atoms_ex_Met),len(atoms_ex_Met))
exclusions_Alkyls=(len(atoms_ex_Met),len(atoms_ex_Oct))

filenames= ['OCT_methyl-TFSI_C', 'TFSI_C-TFSI_C', 'TFSI_C-TFSI_C_anod', 'OCT_methyl-OCT_methyl', 'TFSI_C-OCT_methyl']
filenames_oct= ['OCT_octyls-TFSI_C', 'OCT_methyl-OCT_octyls', 'OCT_octyls-OCT_octyls', 'OCT_octyls-OCT_octyls_cat', 'TFSI_C-OCT_octyls']
filenames_N = ['OCT_N-TFSI_S', 'TFSI_S-OCT_N']

filenames_inEDL= ['inEDL_OCT_methyl-TFSI_C', 'inEDL_TFSI_C-TFSI_C', 'inEDL_TFSI_C-TFSI_C_anod', 'inEDL_OCT_methyl-OCT_methyl', 'inEDL_TFSI_C-OCT_methyl']
filenames_oct_inEDL= ['inEDL_OCT_octyls-TFSI_C', 'inEDL_OCT_methyl-OCT_octyls', 'inEDL_OCT_octyls-OCT_octyls', 'inEDL_OCT_octyls-OCT_octyls_cat', 'inEDL_TFSI_C-OCT_octyls']
filenames_N_inEDL = ['inEDL_OCT_N-TFSI_S', 'inEDL_TFSI_S-OCT_N']

filenames_outEDL= ['outEDL_OCT_methyl-TFSI_C', 'outEDL_TFSI_C-TFSI_C', 'outEDL_TFSI_C-TFSI_C_anod', 'outEDL_OCT_methyl-OCT_methyl', 'outEDL_TFSI_C-OCT_methyl']
filenames_oct_outEDL= ['outEDL_OCT_octyls-TFSI_C', 'outEDL_OCT_methyl-OCT_octyls', 'outEDL_OCT_octyls-OCT_octyls', 'outEDL_OCT_octyls-OCT_octyls_cat', 'outEDL_TFSI_C-OCT_octyls']
filenames_N_outEDL = ['outEDL_OCT_N-TFSI_S', 'outEDL_TFSI_S-OCT_N']

ion2_Met= OCT_Methyl_list
ion2_Oct= OCT_Octyls_lists
ion2_N= OCT_N_list
ion1_S= TFSI_S_lists
ion1_C= TFSI_CF3_lists
grp_cat = grp_A_list 
grp_anod = grp_B_list 

Startframe= 4000
Endframe=9999
dmin, dmax = 0.0, 13.0
bin_width = 0.2
n_bins = int(dmax/bin_width)
layer1_cutoff = 5.5

rdf_CaA=[]
rdf_AaA=[]
rdf_CaC=[]
rdf_AaC=[]


#################### RDF of EDL ions and all ions ############################# 
# Define the number of CPU cores available
num_cores = multiprocessing.cpu_count()

# RDF Calculation Function
def compute_rdf(j, u, ion1, ion2, grp, filename, exclusions=None, rdf_type="12"):
    if rdf_type == "12":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_rdfs_12(dmin, dmax)
    elif rdf_type == "11":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_rdfs_11(dmin, dmax, exclusions)
    elif rdf_type == "12_ex":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_rdfs_12_ex(dmin, dmax, exclusions)
# rdf_type= inEDL
    elif rdf_type == "cal_dec_rdfs_12_inEDL":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_12_inEDL(dmin, dmax)
    elif rdf_type == "cal_dec_rdfs_11_inEDL":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_11_inEDL(dmin, dmax, exclusions)
    elif rdf_type == "cal_dec_rdfs_12_inEDL_ex":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_12_inEDL_ex(dmin, dmax, exclusions)
# rdf_type = outEDL    
    elif rdf_type == "cal_dec_rdfs_12_outEDL":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_12_outEDL(dmin, dmax)
    elif rdf_type == "cal_dec_rdfs_11_outEDL":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_11_outEDL(dmin, dmax, exclusions)
    elif rdf_type == "cal_dec_rdfs_12_outEDL_ex":
        RDFs_instance = RDFs(u, Startframe, Endframe, ion1, ion2, grp, layer1_cutoff, n_bins)
        rdf = RDFs_instance.cal_dec_rdfs_12_outEDL_ex(dmin, dmax, exclusions)
    
    out_path = f'sim_output_v{voltage_name[j]}_ns100/AnaRes/'
    output_file = os.path.join(out_path, f"{filename} RDFs.dat")
    
    with open(output_file, "w") as file:
        for value in rdf:
            file.write(f"{value}\n")

# Parallelize the calculation across voltages and tasks
for j, u in enumerate(u_list):
    tasks = [
        {"ion1": ion1_C[j], "ion2": ion2_Met[j], "grp": grp_cat[j], "filename": filenames[0], "rdf_type": "12"},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_cat[j], "filename": filenames[1], "rdf_type": "11", "exclusions": exclusions_C},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames[2], "rdf_type": "11", "exclusions": exclusions_C},
        {"ion1": ion2_Met[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames[3], "rdf_type": "11", "exclusions": exclusions_Met},
        {"ion1": ion2_Met[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames[4], "rdf_type": "12"},
        {"ion1": ion1_C[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct[0], "rdf_type": "12"},
        {"ion1": ion2_Oct[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames_oct[1], "rdf_type": "12_ex", "exclusions": exclusions_Alkyls},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_anod[j], "filename": filenames_oct[2], "rdf_type": "12_ex", "exclusions": exclusions_Alkyls},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct[3], "rdf_type": "11", "exclusions": exclusions_Oct},
        {"ion1": ion2_Oct[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_oct[4], "rdf_type": "12"},
        {"ion1": ion1_S[j], "ion2": ion2_N[j], "grp": grp_cat[j], "filename": filenames_N[0], "rdf_type": "12"},
        {"ion1": ion2_N[j], "ion2": ion1_S[j], "grp": grp_anod[j], "filename": filenames_N[1], "rdf_type": "12"},
# Decomposed RDF outEDL
        {"ion1": ion1_C[j], "ion2": ion2_Met[j], "grp": grp_cat[j], "filename": filenames_inEDL[0], "rdf_type": "cal_dec_rdfs_12_inEDL"},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_cat[j], "filename": filenames_inEDL[1], "rdf_type": "cal_dec_rdfs_11_inEDL", "exclusions": exclusions_C},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_inEDL[2], "rdf_type": "cal_dec_rdfs_11_inEDL", "exclusions": exclusions_C},
        {"ion1": ion2_Met[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames_inEDL[3], "rdf_type": "cal_dec_rdfs_11_inEDL", "exclusions": exclusions_Met},
        {"ion1": ion2_Met[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_inEDL[4], "rdf_type": "cal_dec_rdfs_12_inEDL"},
        {"ion1": ion1_C[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct_inEDL[0], "rdf_type": "cal_dec_rdfs_12_inEDL"},
        {"ion1": ion2_Oct[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames_oct_inEDL[1], "rdf_type": "cal_dec_rdfs_12_inEDL_ex", "exclusions": exclusions_Alkyls},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_anod[j], "filename": filenames_oct_inEDL[2], "rdf_type": "cal_dec_rdfs_11_inEDL", "exclusions": exclusions_Oct},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct_inEDL[3], "rdf_type": "cal_dec_rdfs_11_inEDL", "exclusions": exclusions_Met},
        {"ion1": ion2_Oct[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_oct_inEDL[4], "rdf_type": "cal_dec_rdfs_12_inEDL"},
        {"ion1": ion1_S[j], "ion2": ion2_N[j], "grp": grp_cat[j], "filename": filenames_N_inEDL[0], "rdf_type": "cal_dec_rdfs_12_inEDL"},
        {"ion1": ion2_N[j], "ion2": ion1_S[j], "grp": grp_anod[j], "filename": filenames_N_inEDL[1], "rdf_type": "cal_dec_rdfs_12_inEDL"},
# Decomposed RDF outEDL
        {"ion1": ion1_C[j], "ion2": ion2_Met[j], "grp": grp_cat[j], "filename": filenames_outEDL[0], "rdf_type": "cal_dec_rdfs_12_outEDL"},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_cat[j], "filename": filenames_outEDL[1], "rdf_type": "cal_dec_rdfs_11_outEDL", "exclusions": exclusions_C},
        {"ion1": ion1_C[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_outEDL[2], "rdf_type": "cal_dec_rdfs_11_outEDL", "exclusions": exclusions_C},
        {"ion1": ion2_Met[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames_outEDL[3], "rdf_type": "cal_dec_rdfs_11_outEDL", "exclusions": exclusions_Met},
        {"ion1": ion2_Met[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_outEDL[4], "rdf_type": "cal_dec_rdfs_12_outEDL"},
        {"ion1": ion1_C[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct_outEDL[0], "rdf_type": "cal_dec_rdfs_12_outEDL"},
        {"ion1": ion2_Oct[j], "ion2": ion2_Met[j], "grp": grp_anod[j], "filename": filenames_oct_outEDL[1], "rdf_type": "cal_dec_rdfs_12_outEDL_ex", "exclusions": exclusions_Alkyls},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_anod[j], "filename": filenames_oct_outEDL[2], "rdf_type": "cal_dec_rdfs_11_outEDL", "exclusions": exclusions_Oct},
        {"ion1": ion2_Oct[j], "ion2": ion2_Oct[j], "grp": grp_cat[j], "filename": filenames_oct_outEDL[3], "rdf_type": "cal_dec_rdfs_11_outEDL", "exclusions": exclusions_Met},
        {"ion1": ion2_Oct[j], "ion2": ion1_C[j], "grp": grp_anod[j], "filename": filenames_oct_outEDL[4], "rdf_type": "cal_dec_rdfs_12_outEDL"},
        {"ion1": ion1_S[j], "ion2": ion2_N[j], "grp": grp_cat[j], "filename": filenames_N_outEDL[0], "rdf_type": "cal_dec_rdfs_12_outEDL"},
        {"ion1": ion2_N[j], "ion2": ion1_S[j], "grp": grp_anod[j], "filename": filenames_N_outEDL[1], "rdf_type": "cal_dec_rdfs_12_outEDL"}
    ]

    # Parallel computation for current u
    Parallel(n_jobs=num_cores)(
        delayed(compute_rdf)(
            j, u, task['ion1'], task['ion2'], task['grp'], task['filename'],
            exclusions=task.get('exclusions'), rdf_type=task['rdf_type']
        ) for task in tasks
    )

print('RDF caculation completed!')


####################### Charge density distribution ############################

############################ L_cell ##################################

ILs_list=[[u.select_atoms('resname OCT '), u.select_atoms('resname Tf2 ')] for u in u_list]

pdb= 'RD_[N1888][TFSI].pdb'
pdb_bd= ['../ffdir/graph_residue_c.xml', '../ffdir/graph_residue_n.xml',
         '../ffdir/graph_residue_s.xml', '../ffdir/sapt_residues.xml']
forcefield=['../ffdir/graph_c_freeze.xml','../ffdir/graph_n_freeze.xml',
            '../ffdir/graph_s_freeze.xml','../ffdir/sapt_noDB_2sheets.xml']

grp_A_list = [u.select_atoms('resname grp and segid A and name C1') for u in u_list]
grp_B_list = [u.select_atoms('resname grp and segid B and name C1') for u in u_list]
grp_A_pos = grp_A_list[0].positions[:, 2]
grp_B_pos = grp_B_list[0].positions[:, 2]

Startframe= 5000
Endframe=9999
Lcell = max(grp_B_pos) - max(grp_A_pos)
bin_width = 0.1
n_bins = int(Lcell/bin_width)


hist_range = (max(grp_A_pos), max(grp_B_pos))

cd_list = []
file_name = "normalized_total_charge.dat"
voltage_name= [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]

# Get the number of CPU cores available
num_cores = multiprocessing.cpu_count()

# Charge Density Calculation Function
def compute_charge_density(i, u, pdb, pdb_bd, forcefield, n_bins, ILs, hist_range):
    # Create an instance of STC class and calculate charge density
    stc = STC(u, Startframe, Endframe, pdb, pdb_bd, forcefield, n_bins, ILs, hist_range)
    cd_list = stc.cal_cd()
    
    # Define the output file path and name
    outPath = f'sim_output_v{voltage_name[i]}_ns100/AnaRes/'
    os.makedirs(outPath, exist_ok=True)  # Create directory if it doesn't exist
    output_file = os.path.join(outPath, "normalized_total_charge.dat")
    
    # Write the charge density data to file
    with open(output_file, "w") as file:
        for value in cd_list:
            file.write(f"{value}\n")

# Parallelize the calculation across voltages and simulations
Parallel(n_jobs=num_cores)(
    delayed(compute_charge_density)(
        i, u_list[i], pdb, pdb_bd, forcefield, n_bins, ILs_list[i], hist_range
    ) for i in range(len(u_list))
)

print('Complete charge density calculation.')
