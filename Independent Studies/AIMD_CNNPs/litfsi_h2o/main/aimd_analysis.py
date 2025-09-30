import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# from mdcraft.analysis.structure import StructureFactor
from MDAnalysis.analysis.rdf import InterRDF
import MDAnalysis.lib.distances as mda_distances
from MDAnalysis.lib.mdamath import make_whole

# --- Configuration ---
TRAJECTORY_FILE = "../results/litfsi_h2o-pos.xyz"
TOPOLOGY_FILE = "../data/gro_md/final_nvt_npt_elements.pdb"
N_FRAMES_FOR_ANALYSIS = 10000
Q_MAX_SIM = 10.0  # Å⁻¹
N_Q_POINTS = 500
RDF_RANGE = (0.0, 15.0)
RDF_NBINS = 300
K_B_T = 0.592  # k_B * T in kcal/mol at 298 K
TIMESTEP_FS = 0.5  # Timestep in fs
EXPT_SQ_FILE = "../data/expt_sq.dat"  # Experimental S(q) data file


# --- Form Factor Calculation ---
# def compute_form_factor(element, q):
#     """
#     Compute the atomic form factor f(q) for a given element and q-values using Cromer-Mann coefficients.

#     Parameters:
#     - element (str): Element symbol ('Li', 'S', 'O', 'N', 'C', 'F', 'H')
#     - q (np.array): Array of q-values (Å⁻¹)

#     Returns:
#     - f (np.array): Form factor values for each q
#     """
#     coefficients = {
#         "Li": {
#             "a": [1.1282, 0.7508, 0.6175, 0.4653],
#             "b": [3.9546, 1.0524, 85.3905, 168.261],
#             "c": 0.0377,
#         },
#         "S": {
#             "a": [6.9053, 5.2034, 1.4379, 1.5863],
#             "b": [1.4679, 14.0408, 0.5391, 35.2385],
#             "c": 0.8670,
#         },
#         "O": {
#             "a": [3.0485, 2.2868, 1.5463, 0.8670],
#             "b": [13.2771, 5.7011, 0.3239, 32.9089],
#             "c": 0.2508,
#         },
#         "N": {
#             "a": [12.2126, 3.1322, 2.0125, 1.1663],
#             "b": [0.0057, 9.8933, 28.9975, 0.5826],
#             "c": -11.529,
#         },
#         "C": {
#             "a": [2.31, 1.02, 1.5886, 0.865],
#             "b": [20.8439, 10.2075, 0.5687, 51.6512],
#             "c": 0.2156,
#         },
#         "F": {
#             "a": [3.5392, 2.6412, 1.517, 1.0243],
#             "b": [10.2825, 4.2944, 0.2615, 26.1476],
#             "c": 0.2776,
#         },
#         "H": {
#             "a": [0.489918, 0.262003, 0.196767, 0.049879],
#             "b": [20.6593, 7.74039, 49.5519, 2.20159],
#             "c": 0.001305,
#         },
#     }
#     if element not in coefficients:
#         return np.ones_like(q)  # Default to 1 if element not found
#     a = coefficients[element]["a"]
#     b = coefficients[element]["b"]
#     c = coefficients[element]["c"]
#     s = q / (4 * np.pi)
#     f = c
#     for ai, bi in zip(a, b):
#         f += ai * np.exp(-bi * s**2)
#     return f


# --- Load the trajectory ---
try:
    universe = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE)
    print(
        f"Universe loaded with {universe.atoms.n_atoms} atoms and {universe.trajectory.n_frames} frames."
    )
    if universe.dimensions is None or np.all(universe.dimensions == 0):
        print("Warning: Box dimensions not loaded from PDB. Setting to 14.936 Å cubic.")
        universe.dimensions = [14.936, 14.936, 14.936, 90.0, 90.0, 90.0]
    print(f"Initial box dimensions: {universe.dimensions}")
except Exception as e:
    print(f"Error loading universe: {e}")
    exit()

n_frames_total = universe.trajectory.n_frames
start_frame_analysis = max(0, n_frames_total - N_FRAMES_FOR_ANALYSIS)
print(
    f"Analyzing frames {start_frame_analysis} to {n_frames_total - 1} ({N_FRAMES_FOR_ANALYSIS} frames)."
)

# --- Check box dimensions ---
box_sizes = np.array(
    [ts.dimensions[:3] for ts in universe.trajectory[start_frame_analysis:]]
)
if np.any(np.std(box_sizes, axis=0) > 0.01):
    print(
        "Warning: Box dimensions vary (NPT detected). Using initial dimensions for simplicity."
    )
else:
    print("Box dimensions constant (NVT or fixed box).")

# # --- Debug: Print atom types ---
# print("Debug: Atom types in topology:", np.unique(universe.atoms.types))

# # --- Unwrap Residues ---
# print("Checking if unwrapping is needed...")
# needs_unwrapping = False
# for ts in universe.trajectory[:1]:  # Check first frame
#     for residue in universe.residues:
#         positions = residue.atoms.positions
#         max_dist = np.max(
#             mda_distances.distance_array(positions, positions, box=universe.dimensions)
#         )
#         if max_dist > universe.dimensions[0] / 2:  # Use first dimension for comparison
#             print(
#                 f"Warning: Residue {residue.resname} {residue.resid} may be wrapped (max dist: {max_dist:.2f} Å)"
#             )
#             needs_unwrapping = True
#             break
#     if needs_unwrapping:
#         break

# if needs_unwrapping:
#     print("Making residues whole...")
#     try:
#         universe.atoms.guess_bonds(
#             vdwradii={
#                 "Li": 1.82,
#                 "N": 1.55,
#                 "S": 1.80,
#                 "O": 1.52,
#                 "C": 1.70,
#                 "F": 1.47,
#                 "H": 1.20,
#             }
#         )
#         print("Bonds successfully guessed. Number of bonds:", len(universe.bonds))
#         fragments = universe.atoms.fragments
#         print(
#             f"Number of fragments: {len(fragments)}, Number of residues: {len(universe.residues)}"
#         )
#         for i, frag in enumerate(fragments):
#             res = frag.residues
#             if len(res) != 1:
#                 print(
#                     f"Warning: Fragment {i} contains {len(res)} residues: {[r.resname for r in res]}"
#                 )
#         for residue in universe.residues:
#             make_whole(residue.atoms, reference_atom=None, inplace=True)
#         print("Completed making residues whole.")
#         for ts in universe.trajectory[:1]:
#             for residue in universe.residues:
#                 positions = residue.atoms.positions
#                 max_dist = np.max(
#                     mda_distances.distance_array(
#                         positions, positions, box=universe.dimensions
#                     )
#                 )
#                 if max_dist > universe.dimensions[0] / 2:
#                     print(
#                         f"Warning: Residue {residue.resname} {residue.resid} still wrapped (max dist: {max_dist:.2f} Å)"
#                     )
#     except Exception as e:
#         print(f"Error during bond guessing or making whole: {e}")
#         print(
#             "Warning: Proceeding without unwrapping. Results may be inaccurate if residues are split across boundaries."
#         )
# else:
#     print("No unwrapping needed; trajectory appears to be unwrapped.")

# # --- S(q) Calculation ---
# print("\nComputing Total X-ray Static Structure Factor S(q) using residues...")
# residue_types = sorted(np.unique(universe.residues.resnames))
# print(f"Residue types identified: {residue_types}")

# sf_residue_groups = [
#     universe.select_atoms(f"resname {resname}")
#     for resname in residue_types
#     if universe.select_atoms(f"resname {resname}").n_atoms > 0
# ]
# residue_list = [
#     group.resnames[0] for group in sf_residue_groups
# ]  # Fixed to get single resname
# for group in sf_residue_groups:
#     print(
#         f"Residue {group.resnames[0]}: {group.n_atoms} atoms ({len(group.residues)} residues)"
#     )

# if not sf_residue_groups:
#     print("Error: No valid residue groups for S(q). Exiting.")
#     exit()

# try:
#     sf_calculator = StructureFactor(
#         groups=sf_residue_groups,
#         groupings="residues",
#         mode="partial",
#         dimensions=[14.936, 14.936, 14.936],
#         n_points=N_Q_POINTS,
#         q_max=Q_MAX_SIM,
#         sort=True,
#     )
#     sf_calculator.run(start=start_frame_analysis)

#     q_sim = sf_calculator.results.wavenumbers
#     ssf_array = sf_calculator.results.ssf
#     ssf_pairs = sf_calculator.results.pairs

#     partial_sf_data = {}
#     for i, (idx_alpha, idx_beta) in enumerate(ssf_pairs):
#         res_alpha = residue_list[idx_alpha]
#         res_beta = residue_list[idx_beta]
#         partial_sf_data[(res_alpha, res_beta)] = ssf_array[i]

#     print(f"Computed partial structure factors for {len(partial_sf_data)} pairs.")
#     print("Sample partial SF keys:", list(partial_sf_data.keys())[:5])

#     # Plot each partial S(q)
#     for pair, s_ab in partial_sf_data.items():
#         plt.figure()
#         plt.plot(q_sim, s_ab, label=f"S_{pair[0]}-{pair[1]}(q)", color="blue")
#         plt.xlabel("q (Å⁻¹)")
#         plt.ylabel("S(q)")
#         plt.xlim(0.42, 10)
#         plt.ylim(-1.5, 1.5)
#         plt.title(f"Partial Structure Factor for {pair[0]}-{pair[1]}")
#         plt.legend()
#         plt.show()

#     # --- Compute q-dependent form factors for residues based on stoichiometry ---
#     f_Li = compute_form_factor("Li", q_sim)
#     f_N = compute_form_factor("N", q_sim)
#     f_S = compute_form_factor("S", q_sim)
#     f_O = compute_form_factor("O", q_sim)
#     f_C = compute_form_factor("C", q_sim)
#     f_F = compute_form_factor("F", q_sim)
#     f_H = compute_form_factor("H", q_sim)

#     form_factors = {}
#     for resname in residue_types:
#         if resname == "LI":
#             form_factors[resname] = f_Li  # Single Li atom
#         elif resname == "NSC":
#             # TFSI anion: Approximate stoichiometry (e.g., N1S2O4C2F6H0 from typical TFSI structure)
#             form_factors[resname] = f_N + 2 * f_S + 4 * f_O + 2 * f_C + 6 * f_F
#         elif resname == "SOL":
#             # Water molecule: O1H2
#             form_factors[resname] = f_O + 2 * f_H
#         else:
#             print(
#                 f"Warning: Unknown residue type {resname}. Using default form factor of 1."
#             )
#             form_factors[resname] = np.ones_like(q_sim)

#     print("Residue form factors computed for:", list(form_factors.keys()))

#     # Compute weighted total S(q)
#     atom_counts = {
#         r: len(universe.select_atoms(f"resname {r}").atoms) for r in residue_types
#     }
#     total_atoms = sum(atom_counts.values())
#     total_S_q = np.zeros_like(q_sim)
#     for (res_alpha, res_beta), s_ab in partial_sf_data.items():
#         c_alpha = atom_counts[res_alpha] / total_atoms
#         c_beta = atom_counts[res_beta] / total_atoms
#         f_alpha = form_factors[res_alpha]
#         f_beta = form_factors[res_beta]
#         total_S_q += c_alpha * c_beta * f_alpha * f_beta * s_ab
#     f_total = np.sum(
#         [atom_counts[res] / total_atoms * form_factors[res] for res in residue_types],
#         axis=0,
#     )
#     total_S_q /= f_total**2
#     total_S_q_sum = total_S_q
#     total_S_q_std = (
#         np.std(ssf_array, axis=0) if ssf_array.shape[0] > 1 else np.zeros_like(q_sim)
#     )
#     print("Total S(q) computed as weighted sum of partial structure factors.")

#     # Debug: Print S(q) values at specific q points
#     idx_q1 = np.argmin(np.abs(q_sim - 1))
#     idx_q2 = np.argmin(np.abs(q_sim - 2))
#     print(f"S(q=1 Å⁻¹) = {total_S_q_sum[idx_q1]:.3f} ± {total_S_q_std[idx_q1]:.3f}")
#     print(f"S(q=2 Å⁻¹) = {total_S_q_sum[idx_q2]:.3f} ± {total_S_q_std[idx_q2]:.3f}")

# except Exception as e:
#     print(f"Error computing S(q): {e}")
#     total_S_q_sum = np.array([])
#     q_sim = np.array([])

# --- RDF and Distance Calculations ---
print("\nComputing RDFs and Distances...")
li = universe.select_atoms("name LI")
o_anion = universe.select_atoms("resname NSC and name OBT")
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
rdf_li_li.run(start=start_frame_analysis)
rdf_li_oa.run(start=start_frame_analysis)
rdf_li_ow.run(start=start_frame_analysis)
print("RDFs computation complete.")

g_r = rdf_li_li.results.rdf
g_r_safe = np.where(g_r > 1e-6, g_r, 1e-6)
F_r = -K_B_T * np.log(g_r_safe)

times = []
avg_distances = []
print("\nCalculating average Li+ pair distances over time...")
for i, ts in enumerate(universe.trajectory[start_frame_analysis:]):
    t = (start_frame_analysis + i) * TIMESTEP_FS
    times.append(t)
    li_positions = li.positions
    dist_array = mda_distances.distance_array(
        li_positions, li_positions, box=universe.dimensions
    )
    indices = np.triu_indices(len(li), k=1)
    pair_distances = dist_array[indices]
    avg_distances.append(np.mean(pair_distances))

mean_distance = np.mean(avg_distances)
std_distance = np.std(avg_distances)
print(
    f"Average distance between all Li+ pairs: {mean_distance:.3f} Å ± {std_distance:.3f} Å"
)

# # --- Load Experimental S(q) Data ---
# try:
#     q_exp, s_q_exp = np.loadtxt(EXPT_SQ_FILE, delimiter=",", unpack=True)
#     print(
#         f"Loaded experimental S(q) data from '{EXPT_SQ_FILE}' with {len(q_exp)} points."
#     )
#     mask = (q_exp >= 0) & (q_exp <= 10)
#     q_exp = q_exp[mask]
#     s_q_exp = s_q_exp[mask]
#     print(f"Filtered experimental data to range: {len(q_exp)} points remain.")
# except Exception as e:
#     print(f"Error loading experimental S(q): {e}")
#     q_exp = np.array([])
#     s_q_exp = np.array([])

# # --- Plotting ---
# print("\nGenerating plots...")

# # Plot total S(q) without interpolation
# plt.figure(figsize=(7, 5))
# if total_S_q_sum.size > 0 and q_sim.size > 0:
#     plt.plot(q_sim, total_S_q_sum, color="purple", label="AIMD S(q) (Weighted)")
# if q_exp.size > 0 and s_q_exp.size > 0:
#     plt.scatter(q_exp, s_q_exp, color="orange", label="Experimental S(q)", marker="x")
# plt.title("Structure Factor S(q): AIMD (Weighted) vs Experimental")
# plt.xlabel("q (Å⁻¹)")
# plt.ylabel("S(q)")
# plt.xlim(0.42, 10)
# plt.ylim(-1.5, 1.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

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
plt.ylim(0, 5)
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot Average Li+ Pair Distances
plt.figure(figsize=(5.5, 5))
plt.plot(times, avg_distances, color="#9467bd")
plt.title("Average Distance Between All Li+ Pairs Over Time")
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


########## Unweighted S(q) Calculation ##########
# import MDAnalysis as mda
# import numpy as np
# import matplotlib.pyplot as plt
# from mdcraft.analysis.structure import StructureFactor
# from MDAnalysis.analysis.rdf import InterRDF
# import MDAnalysis.lib.distances as mda_distances
# from MDAnalysis.lib.mdamath import make_whole

# # --- Configuration ---
# TRAJECTORY_FILE = "../results/litfsi_h2o_relax-pos.xyz"
# TOPOLOGY_FILE = "../data/gro_md/final_npt_elements.pdb"
# N_FRAMES_FOR_ANALYSIS = 4000
# Q_MAX_SIM = 12.0  # Å⁻¹
# N_Q_POINTS = 500
# RDF_RANGE = (0.0, 15.0)
# RDF_NBINS = 300
# K_B_T = 0.592  # k_B * T in kcal/mol at 298 K
# TIMESTEP_FS = 0.5  # Timestep in fs
# EXPT_SQ_FILE = "../data/expt_sq.dat"  # Experimental S(q) data file

# # --- Load the trajectory ---
# try:
#     universe = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE)
#     print(
#         f"Universe loaded with {universe.atoms.n_atoms} atoms and {universe.trajectory.n_frames} frames."
#     )
#     if universe.dimensions is None or np.all(universe.dimensions == 0):
#         print("Warning: Box dimensions not loaded from PDB. Setting to 14.936 Å cubic.")
#         universe.dimensions = [14.936, 14.936, 14.936, 90.0, 90.0, 90.0]
#     print(f"Initial box dimensions: {universe.dimensions}")
# except Exception as e:
#     print(f"Error loading universe: {e}")
#     exit()

# n_frames_total = universe.trajectory.n_frames
# start_frame_analysis = max(0, n_frames_total - N_FRAMES_FOR_ANALYSIS)
# print(
#     f"Analyzing frames {start_frame_analysis} to {n_frames_total - 1} ({N_FRAMES_FOR_ANALYSIS} frames)."
# )

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

# # --- Debug: Print atom types ---
# print("Debug: Atom types in topology:", np.unique(universe.atoms.types))

# # --- Unwrap Residues ---
# # Check if residues need unwrapping
# print("Checking if unwrapping is needed...")
# needs_unwrapping = False
# for ts in universe.trajectory[:1]:  # Check first frame
#     for residue in universe.residues:
#         positions = residue.atoms.positions
#         max_dist = np.max(mda_distances.distance_array(positions, positions, box=universe.dimensions))
#         if max_dist > universe.dimensions[0] / 2:
#             print(f"Warning: Residue {residue.resname} {residue.resid} may be wrapped (max dist: {max_dist:.2f} Å)")
#             needs_unwrapping = True
#             break
#     if needs_unwrapping:
#         break

# if needs_unwrapping:
#     print("Making residues whole...")
#     try:
#         # Guess bonds with corrected vdwradii for atom types
#         universe.atoms.guess_bonds(vdwradii={'Li': 1.82, 'N': 1.55, 'S': 1.80, 'O': 1.52, 'C': 1.70, 'F': 1.47, 'H': 1.20})
#         print("Bonds successfully guessed. Number of bonds:", len(universe.bonds))
#         # Debug: Verify fragments match residues
#         fragments = universe.atoms.fragments
#         print(f"Number of fragments: {len(fragments)}, Number of residues: {len(universe.residues)}")
#         for i, frag in enumerate(fragments):
#             res = frag.residues
#             if len(res) != 1:
#                 print(f"Warning: Fragment {i} contains {len(res)} residues: {[r.resname for r in res]}")
#         # Apply make_whole to each residue
#         for residue in universe.residues:
#             make_whole(residue.atoms, reference_atom=None, inplace=True)
#         print("Completed making residues whole.")
#         # Verify unwrapping
#         for ts in universe.trajectory[:1]:  # Check first frame again
#             for residue in universe.residues:
#                 positions = residue.atoms.positions
#                 max_dist = np.max(mda_distances.distance_array(positions, positions, box=universe.dimensions))
#                 if max_dist > universe.dimensions[0] / 2:
#                     print(f"Warning: Residue {residue.resname} {residue.resid} still wrapped (max dist: {max_dist:.2f} Å)")
#     except Exception as e:
#         print(f"Error during bond guessing or making whole: {e}")
#         print("Warning: Proceeding without unwrapping. Results may be inaccurate if residues are split across boundaries.")
#         print("Suggestion: Pre-unwrap the trajectory using tools like 'gmx trjconv -pbc res' or provide a topology with bond information (e.g., PSF, GRO).")
# else:
#     print("No unwrapping needed; trajectory appears to be unwrapped.")

# # # --- S(q) Calculation ---
# # print("\nComputing Total X-ray Static Structure Factor S(q) using residues...")
# # residue_types = sorted(np.unique(universe.residues.resnames))
# # print(f"Residue types identified: {residue_types}")

# # sf_residue_groups = [
# #     universe.select_atoms(f"resname {resname}")
# #     for resname in residue_types
# #     if universe.select_atoms(f"resname {resname}").n_atoms > 0
# # ]
# # residue_list = [group.resnames[0] for group in sf_residue_groups]
# # for group in sf_residue_groups:
# #     print(
# #         f"Residue {group.resnames[0]}: {group.n_atoms} atoms ({len(group.residues)} residues)"
# #     )

# # if not sf_residue_groups:
# #     print("Error: No valid residue groups for S(q). Exiting.")
# #     exit()

# # try:
# #     sf_calculator = StructureFactor(
# #         groups=sf_residue_groups,
# #         groupings="residues",
# #         mode="partial",
# #         dimensions=[14.936, 14.936, 14.936],
# #         n_points=N_Q_POINTS,
# #         q_max=Q_MAX_SIM,
# #         sort=True,
# #     )
# #     sf_calculator.run(start=start_frame_analysis)

# #     q_sim = sf_calculator.results.wavenumbers
# #     ssf_array = sf_calculator.results.ssf
# #     ssf_pairs = sf_calculator.results.pairs

# #     partial_sf_data = {}
# #     for i, (idx_alpha, idx_beta) in enumerate(ssf_pairs):
# #         res_alpha = residue_list[idx_alpha]
# #         res_beta = residue_list[idx_beta]
# #         partial_sf_data[(res_alpha, res_beta)] = ssf_array[i]

# #     print(f"Computed partial structure factors for {len(partial_sf_data)} pairs.")
# #     print("Sample partial SF keys:", list(partial_sf_data.keys())[:5])

# #     # Debug: Plot partial structure factors for key pairs
# #     key_pairs = [
# #         ("LI", "LI"),
# #         ("NSC", "NSC"),
# #         ("LI", "NSC"),
# #         ("LI", "SOL"),
# #         ("NSC", "SOL"),
# #         ("SOL", "SOL"),
# #     ]
# #     for pair in key_pairs:
# #         if pair in partial_sf_data:
# #             s_ab = partial_sf_data[pair]
# #             print(
# #                 f"{pair} S(q): min={s_ab.min():.3f}, max={s_ab.max():.3f}, mean={s_ab.mean():.3f}"
# #             )
# #             plt.figure(figsize=(5, 4))
# #             plt.plot(q_sim, s_ab, label=f"S_{pair[0]}_{pair[1]}(q)")
# #             plt.title(f"Partial S(q) for {pair}")
# #             plt.xlabel("q (Å⁻¹)")
# #             plt.ylabel("S(q)")
# #             plt.xlim(0.42, 12)
# #             plt.ylim(-1.0, 2)
# #             plt.legend()
# #             plt.show()

# #     # Compute total S(q) by unweighted sum of partials
# #     total_S_q_sum = np.sum(ssf_array, axis=0)
# #     print("Total S(q) computed as unweighted sum of partial structure factors.")

# #     # Debug: Print S(q) values at specific q points
# #     idx_q1 = np.argmin(np.abs(q_sim - 1))
# #     idx_q2 = np.argmin(np.abs(q_sim - 2))
# #     print(f"S(q=1 Å⁻¹) = {total_S_q_sum[idx_q1]:.3f}")
# #     print(f"S(q=2 Å⁻¹) = {total_S_q_sum[idx_q2]:.3f}")

# # except Exception as e:
# #     print(f"Error computing S(q): {e}")
# #     total_S_q_sum = np.array([])
# #     q_sim = np.array([])

# # --- RDF and Distance Calculations ---
# print("\nComputing RDFs and Distances...")
# li = universe.select_atoms("name LI")
# o_anion = universe.select_atoms("resname NSC and name OBT")
# o_water = universe.select_atoms("resname SOL and name OW")

# if not li or not o_anion or not o_water:
#     print("Warning: Some atom selections for RDFs are empty.")
#     print(
#         f"Li atoms: {len(li)}, Anion O atoms: {len(o_anion)}, Water O atoms: {len(o_water)}"
#     )
#     if len(li) == 0 or len(o_anion) == 0:
#         print(
#             "Error: Critical atom groups empty. Cannot proceed with RDF calculations."
#         )
#         exit()

# rdf_li_li = InterRDF(
#     g1=li, g2=li, range=RDF_RANGE, nbins=RDF_NBINS, exclusion_block=(1, 1)
# )
# rdf_li_oa = InterRDF(g1=li, g2=o_anion, range=RDF_RANGE, nbins=RDF_NBINS)
# rdf_li_ow = InterRDF(g1=li, g2=o_water, range=RDF_RANGE, nbins=RDF_NBINS)
# rdf_li_li.run(start=start_frame_analysis)
# rdf_li_oa.run(start=start_frame_analysis)
# rdf_li_ow.run(start=start_frame_analysis)
# print("RDFs computation complete.")

# g_r = rdf_li_li.rdf
# g_r_safe = np.where(g_r > 1e-6, g_r, 1e-6)
# F_r = -K_B_T * np.log(g_r_safe)

# times = []
# avg_distances = []
# print("\nCalculating average Li+ pair distances over time...")
# for i, ts in enumerate(universe.trajectory[start_frame_analysis:]):
#     t = (start_frame_analysis + i) * TIMESTEP_FS
#     times.append(t)
#     li_positions = li.positions
#     dist_array = mda_distances.distance_array(
#         li_positions, li_positions, box=universe.dimensions
#     )
#     indices = np.triu_indices(len(li), k=1)
#     pair_distances = dist_array[indices]
#     avg_distances.append(np.mean(pair_distances))

# mean_distance = np.mean(avg_distances)
# std_distance = np.std(avg_distances)
# print(
#     f"Average distance between all Li+ pairs: {mean_distance:.3f} Å ± {std_distance:.3f} Å"
# )

# # --- Load Experimental S(q) Data ---
# try:
#     q_exp, s_q_exp = np.loadtxt(EXPT_SQ_FILE, delimiter=",", unpack=True)
#     print(
#         f"Loaded experimental S(q) data from '{EXPT_SQ_FILE}' with {len(q_exp)} points."
#     )
#     mask = (q_exp >= 0) & (q_exp <= 12)
#     q_exp = q_exp[mask]
#     s_q_exp = s_q_exp[mask]
#     print(f"Filtered experimental data to range [0, 12]: {len(q_exp)} points remain.")
# except Exception as e:
#     print(f"Error loading experimental S(q): {e}")
#     q_exp = np.array([])
#     s_q_exp = np.array([])

# # --- Plotting ---
# print("\nGenerating plots...")

# # Plot S(q) Comparison
# plt.figure(figsize=(7, 5))
# if total_S_q_sum.size > 0 and q_sim.size > 0:
#     plt.plot(q_sim, total_S_q_sum, color="purple", label="AIMD S(q) (Unweighted Sum)")
# else:
#     print("Warning: Simulated S(q) data unavailable for plotting.")
# if q_exp.size > 0 and s_q_exp.size > 0:
#     plt.plot(q_exp, s_q_exp, color="orange", linestyle="-", label="Experimental S(q)")
# else:
#     print("Warning: Experimental S(q) data unavailable for plotting.")
# plt.title("Structure Factor S(q): AIMD (Unweighted) vs Experimental")
# plt.xlabel("q (Å⁻¹)")
# plt.ylabel("S(q)")
# plt.xlim(0.42, 12)
# plt.ylim(-1.0, 2)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot All RDFs
# plt.figure(figsize=(5.5, 5))
# plt.plot(rdf_li_li.bins, rdf_li_li.rdf, color="#1f77b4", label="Li-Li")
# plt.plot(rdf_li_oa.bins, rdf_li_oa.rdf, color="#ff7f0e", label="Li-O_anion")
# plt.plot(rdf_li_ow.bins, rdf_li_ow.rdf, color="#2ca02c", label="Li-O_water")
# plt.xlim(0, 8)
# plt.ylim(0, 25)
# plt.title("Radial Distribution Functions")
# plt.xlabel("Distance (Å)")
# plt.ylabel("g(r)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot Li-Li RDF (zoomed)
# plt.figure(figsize=(5.5, 5))
# plt.plot(rdf_li_li.bins, rdf_li_li.rdf, color="#1f77b4", label="Li-Li")
# plt.title("Radial Distribution Function (Li-Li)")
# plt.xlim(0, 8)
# plt.ylim(0, 5)
# plt.xlabel("Distance (Å)")
# plt.ylabel("g(r)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot Average Li+ Pair Distances
# plt.figure(figsize=(5.5, 5))
# plt.plot(times, avg_distances, color="#9467bd")
# plt.title("Average Distance Between All Li+ Pairs Over Time")
# plt.xlabel("Time (fs)")
# plt.ylabel("Average Distance (Å)")
# plt.tight_layout()
# plt.show()

# # Plot Li-Li Free Energy Profile
# plt.figure(figsize=(5.5, 5))
# plt.plot(rdf_li_li.bins, F_r, color="#9467bd")
# plt.title("Li-Li Free Energy Profile")
# plt.xlim(0, 8)
# plt.ylim(-0.5, 0.5)
# plt.xlabel("Distance (Å)")
# plt.ylabel("F(r) (kcal/mol)")
# plt.tight_layout()
# plt.show()
