import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# from mdcraft.analysis.structure import StructureFactor
from MDAnalysis.analysis.rdf import InterRDF
import MDAnalysis.lib.distances as mda_distances
import ase.io
import os

# --- Configuration ---
TRAJECTORY_FILE = "../results/liotf_h2o_10ips-prod-pos.xyz"
TOPOLOGY_FILE = "../data/gro_md/final_nvt_npt_elements.pdb"
N_FRAMES_FOR_ANALYSIS = 20000
Q_MAX_SIM = 10.0  # Å⁻¹
N_Q_POINTS = 500
RDF_RANGE = (0.0, 10.0)
RDF_NBINS = 150
K_B_T = 0.592  # k_B * T in kcal/mol at 298 K
TIMESTEP_FS = 10  # Timestep in fs
FRAME_STEP = 1  # Step for frame selection (every 10th frame)


# Convert LAMMPS dump to ASE-readable XYZ format
def convert_lammps_to_xyz(input_file, output_file):
    print("Converting LAMMPS trajectory to XYZ using ASE...")
    atoms_list = ase.io.read(input_file, format="lammps-dump-text", index=":")
    ase.io.write(output_file, atoms_list, format="xyz")
    print("Conversion complete.")


CONVERTED_TRAJ = "converted_traj_ase.xyz"
convert_lammps_to_xyz(TRAJECTORY_FILE, CONVERTED_TRAJ)

# --- Load the trajectory ---
try:
    universe = mda.Universe(TOPOLOGY_FILE, CONVERTED_TRAJ, format="XYZ")
    print(
        f"Universe loaded with {universe.atoms.n_atoms} atoms and {universe.trajectory.n_frames} frames."
    )
    if universe.dimensions is None or np.all(universe.dimensions == 0):
        print("Warning: Box dimensions not loaded from PDB. Setting to 12.465 Å cubic.")
        universe.dimensions = [12.465, 12.465, 12.465, 90.0, 90.0, 90.0]
    print(f"Initial box dimensions: {universe.dimensions}")
except Exception as e:
    print(f"Error loading universe: {e}")
    exit()

# n_frames_total = universe.trajectory.n_frames
# start_frame_analysis = max(0, n_frames_total - N_FRAMES_FOR_ANALYSIS)
# print(
#     f"Analyzing frames {start_frame_analysis} to {n_frames_total - 1} ({N_FRAMES_FOR_ANALYSIS} frames)."
# )
# print(f"Using every {FRAME_STEP}th frame for calculations.")

# # --- Check box dimensions ---
# box_sizes = np.array(
#     [ts.dimensions[:3] for ts in universe.trajectory[start_frame_analysis:]]
# )
# if np.any(np.std(box_sizes, axis=0) > 0.01):
#     print(
#         "Warning: Box dimensions vary (NPT detected). Using initial dimensions for simplicity."
#     )
# else:
#     print("Box dimensions constant (NVT or fixed box).")

# --- RDF and Distance Calculations ---
print("\nComputing RDFs and Distances...")
li = universe.select_atoms("name LI")
o_anion = universe.select_atoms("resname TFO and name OB")
o_water = universe.select_atoms("resname SOL and name OW")

if not li or not o_anion or not o_water:
    print("Warning: Some atom selections for RDFs are empty.")
    print(
        f"Li atoms: {len(li)}, Anion O atoms: {len(o_anion)}, Water O atoms: {len(o_water)}"
    )
    if len(li) == 0 or len(o_anion) == 0:
        print(
            "Error: Critical atom groups empty. Cannot proceed with RDF calculations."
        )
        exit()

rdf_li_li = InterRDF(
    g1=li, g2=li, range=RDF_RANGE, nbins=RDF_NBINS, exclusion_block=(1, 1)
)
rdf_li_oa = InterRDF(g1=li, g2=o_anion, range=RDF_RANGE, nbins=RDF_NBINS)
rdf_li_ow = InterRDF(g1=li, g2=o_water, range=RDF_RANGE, nbins=RDF_NBINS)
rdf_li_li.run(start=0, step=FRAME_STEP)
rdf_li_oa.run(start=0, step=FRAME_STEP)
rdf_li_ow.run(start=0, step=FRAME_STEP)
print("RDFs computation complete.")

g_r = rdf_li_li.results.rdf
g_r_safe = np.where(g_r > 1e-6, g_r, 1e-6)
F_r = -K_B_T * np.log(g_r_safe)

# Select the specific Li ions (closest pair: Li5 and Li9 in the initial structure)
li_i = universe.select_atoms("resname LI and resid 9")
li_j = universe.select_atoms("resname LI and resid 10")
o_i = universe.select_atoms("name OW and resid 42")

n_frames_total = universe.trajectory.n_frames
start_frame_analysis = 0

if len(li_i) != 1 or len(li_j) != 1 or len(o_i) != 1:
    print("Error: Could not select Li_i, Li_j, or O_i. Check residue IDs.")
    exit()

times = []
distances_li_li = []
distances_li_i_o = []
distances_li_j_o = []
average_li_o_distances = []
print("\nCalculating distances over time...")
for i, ts in enumerate(universe.trajectory[start_frame_analysis:]):
    t = (start_frame_analysis + i) * TIMESTEP_FS
    times.append(t)
    pos_li_i = li_i.positions[0]
    pos_li_j = li_j.positions[0]
    pos_o_i = o_i.positions[0]
    dist_li_li = mda_distances.calc_bonds(pos_li_i, pos_li_j, box=universe.dimensions)
    dist_li_i_o = mda_distances.calc_bonds(pos_li_i, pos_o_i, box=universe.dimensions)
    dist_li_j_o = mda_distances.calc_bonds(pos_li_j, pos_o_i, box=universe.dimensions)
    avg_li_o = (dist_li_i_o + dist_li_j_o) / 2
    distances_li_li.append(dist_li_li)
    distances_li_i_o.append(dist_li_i_o)
    distances_li_j_o.append(dist_li_j_o)
    average_li_o_distances.append(avg_li_o)

mean_distance_li_li = np.mean(distances_li_li)
std_distance_li_li = np.std(distances_li_li)
print(
    f"Average distance between two Li ions: {mean_distance_li_li:.3f} Å ± {std_distance_li_li:.3f} Å"
)

mean_average_li_o = np.mean(average_li_o_distances)
std_average_li_o = np.std(average_li_o_distances)
print(
    f"Overall average Li-O distance (average of the two bonds): {mean_average_li_o:.3f} Å ± {std_average_li_o:.3f} Å"
)


# Plot All RDFs
plt.figure(figsize=(5.5, 5))
plt.plot(rdf_li_li.results.bins, rdf_li_li.results.rdf, color="#1f77b4", label="Li-Li")
plt.plot(
    rdf_li_oa.results.bins, rdf_li_oa.results.rdf, color="#ff7f0e", label="Li-O_anion"
)
plt.plot(
    rdf_li_ow.results.bins, rdf_li_ow.results.rdf, color="#2ca02c", label="Li-O_water"
)
plt.xlim(0, 8)
plt.ylim(0, 25)
plt.title("Radial Distribution Functions")
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot Li-Li RDF (zoomed)
plt.figure(figsize=(5.5, 5))
plt.plot(rdf_li_li.results.bins, rdf_li_li.results.rdf, color="#1f77b4", label="Li-Li")
plt.title("Radial Distribution Function (Li-Li)")
plt.xlim(0, 8)
plt.ylim(0, 20)
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot Distance Between Li_i and Li_j
plt.figure(figsize=(5.5, 5))
plt.plot(times, distances_li_li, color="#9467bd")
plt.title("Distance Between two Li Ions Over Time")
plt.xlabel("Time (fs)")
plt.ylabel("Distance (Å)")
plt.tight_layout()
plt.show()


# Plot Average Li-O Distance Over Time
plt.figure(figsize=(5.5, 5))
plt.plot(times, average_li_o_distances, color="#ff7f0e")
plt.title("Average Distance Between the Two Li-O Bonds Over Time")
plt.xlabel("Time (fs)")
plt.ylabel("Average Distance (Å)")
plt.tight_layout()
plt.show()

# Plot Li-Li Free Energy Profile
plt.figure(figsize=(5.5, 5))
plt.plot(rdf_li_li.results.bins, F_r, color="#9467bd")
plt.title("Li-Li Free Energy Profile")
plt.xlim(0, 8)
plt.ylim(-0.5, 0.5)
plt.xlabel("Distance (Å)")
plt.ylabel("F(r) (kcal/mol)")
plt.tight_layout()
plt.show()
