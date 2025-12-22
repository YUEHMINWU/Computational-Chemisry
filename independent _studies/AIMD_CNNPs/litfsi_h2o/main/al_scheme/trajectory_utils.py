import numpy as np
import os
from ase.io import read, write as ase_write
from ase import Atoms
from io import StringIO
import sys
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Constants for LiTFSI water system
HARTREE_TO_KCAL_MOL = 627.50960803
BOHR_TO_ANGSTROM = 0.5291772109
FORCE_CONVERSION_FACTOR = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM
CELL_EDGE_LENGTH = 14.936
CUBIC_CELL = [CELL_EDGE_LENGTH] * 3
SPECIES = ["Li", "F", "S", "O", "C", "N", "H"]  # LiTFSI water system elements
N_ATOMS = 244  # Total number of atoms in the system
RUNS = [
    {
        "pos": "../results/litfsi_h2o_relax-pos.xyz",
        "frc": "../results/litfsi_h2o_relax-frc.xyz",
        "ener": "litfsi_h2o_relax-1.ener",
        "start_step": 0,
        "end_step": 7500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re7500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re7500-frc.xyz",
        "ener": "litfsi_h2o_prod_re7500-1.ener",
        "start_step": 7501,
        "end_step": 20500,
    },
    {
        "pos": "../results/litfsi_h2o_prod_re20500-pos.xyz",
        "frc": "../results/litfsi_h2o_prod_re20500-frc.xyz",
        "ener": "litfsi_h2o_prod_re20500-1.ener",
        "start_step": 20501,
        "end_step": 39999,
    },
]
PRIMARY_TRAIN_VAL_FRAMES = 4500  # Total frames for primary model (train + val)
ENSEMBLE_TRAIN_VAL_FRAMES = 3000  # Total frames per ensemble model (train + val)
COMBINED_TEST_FRAMES = 500  # Absolute for combined test from last 5 ps
NUM_ENSEMBLES = 3  # Number of ensemble models
NUM_PRIMARY = 1  # Primary model
NUM_INIT_STRUCTS = 3  # Number of diverse initial structures per iteration
SILHOUETTE_THRESHOLD = -1.0  # Minimum silhouette score for filtering
FINAL_TEST_FILE = "final_test.extxyz"  # Combined test for final model evaluation
PRIMARY_TRAIN_VAL_FILE = "aimd_trajectory_primary_train_val.extxyz"  # For primary model
TRUE_CALIB_FILE = "true_calib.extxyz"
TRUE_CALIB_FRAMES = 601


def read_xyz_frames(file_path):
    """Read XYZ file and return a dictionary mapping step numbers to frame strings."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        frames = {}
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            n_atoms = int(lines[i].strip())
            comment_line = lines[i + 1].strip()
            try:
                step_str = comment_line.split("i =")[1].split(",")[0].strip()
                step = int(step_str)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse step from comment: {comment_line}")
                i += n_atoms + 2
                continue
            frame_lines = lines[i : i + n_atoms + 2]
            frames[step] = "".join(frame_lines)
            i += n_atoms + 2
        return frames
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        raise


def process_trajectory_chunk(frame_data):
    """Process a chunk of frame data into Atoms objects with per-atom energy and max force magnitude."""
    processed_atoms = []
    for pos_string, force_string, energy_hartree, step in frame_data:
        try:
            pos_atoms = read(StringIO(pos_string), format="xyz")
            energy_kcal_mol = float(energy_hartree) * HARTREE_TO_KCAL_MOL
            energy_per_atom = energy_kcal_mol / N_ATOMS  # Calculate per-atom energy
            force_lines = force_string.strip().split("\n")
            n_atoms = int(force_lines[0].strip())
            if len(pos_atoms) != n_atoms or n_atoms != N_ATOMS:
                raise ValueError(
                    f"Atom count mismatch or unexpected count {n_atoms} at step {step}"
                )
            force_data = []
            for line in force_lines[2 : 2 + n_atoms]:
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"Line has {len(parts)} columns, expected at least 4: {line}"
                    )
                force_components = [float(x) for x in parts[-3:]]
                force_data.append(force_components)
            forces_hartree_bohr = np.array(force_data, dtype=np.float32)
            forces_kcal_mol_angstrom = forces_hartree_bohr * FORCE_CONVERSION_FACTOR
            max_force = np.max(
                np.linalg.norm(forces_kcal_mol_angstrom, axis=1)
            )  # Calculate max force magnitude
            atoms = Atoms(
                symbols=pos_atoms.symbols,
                positions=pos_atoms.positions.astype(np.float32),
            )
            atoms.set_cell(CUBIC_CELL)
            atoms.arrays["forces"] = forces_kcal_mol_angstrom
            atoms.info["energy"] = float(energy_kcal_mol)
            atoms.info["energy_per_atom"] = float(energy_per_atom)
            atoms.info["max_force"] = float(max_force)
            atoms.info["step"] = int(step)
            atoms.set_pbc(True)
            processed_atoms.append(atoms)
        except Exception as e:
            print(f"Error processing frame at step {step}: {e}")
            continue
    return processed_atoms


def compute_soap_descriptors(atoms_list):
    """Compute SOAP descriptors for a list of Atoms objects."""
    soap = SOAP(
        species=SPECIES,
        periodic=True,
        r_cut=6.0,  # Match Allegro cutoff
        n_max=8,
        l_max=6,
        sigma=0.5,
        average="inner",
    )
    return soap.create(atoms_list)


def sample_dataset(
    valid_atoms,
    combined_features_scaled,
    output_traj_file,
    frames_to_use,
    random_seed,
    excluded_steps,
):
    # Filter to remaining frames (exclude previously selected steps)
    remaining_indices = [
        i
        for i in range(len(valid_atoms))
        if valid_atoms[i].info["step"] not in excluded_steps
    ]
    if len(remaining_indices) < frames_to_use:
        print(
            f"Warning: Only {len(remaining_indices)} frames remaining, cannot sample {frames_to_use}. Using all remaining."
        )
        selected_indices = remaining_indices
    else:
        remaining_features = combined_features_scaled[remaining_indices]

        k = min(frames_to_use, len(remaining_features))
        km = KMeans(n_clusters=k, random_state=random_seed).fit(remaining_features)
        cluster_labels = km.labels_

        # Compute silhouette scores on remaining
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(remaining_features, cluster_labels)
            sil_samples = silhouette_samples(remaining_features, cluster_labels)
            print(f"Silhouette Score (Remaining Dataset): {sil_score:.3f}")
        else:
            print(
                "Warning: Only one cluster detected in remaining, silhouette score not computed."
            )
            sil_samples = np.zeros(len(remaining_features))

        # Filter based on silhouette threshold
        filtered_rel_indices = [
            j
            for j in range(len(remaining_features))
            if sil_samples[j] > SILHOUETTE_THRESHOLD
        ]
        n_filtered = len(filtered_rel_indices)
        print(
            f"Filtered to {n_filtered} frames with silhouette coefficient > {SILHOUETTE_THRESHOLD}"
        )

        if n_filtered >= k:
            silhouette_scores_filtered = np.array(
                [sil_samples[j] for j in filtered_rel_indices]
            )
            top_k_rel_indices = np.argsort(silhouette_scores_filtered)[::-1][:k]
            selected_rel_indices = [filtered_rel_indices[m] for m in top_k_rel_indices]
        else:
            print(
                f"Warning: Only {n_filtered} frames passed silhouette filtering, supplementing with random selection."
            )
            selected_rel_indices = filtered_rel_indices.copy()
            all_rel_indices = list(range(len(remaining_features)))
            remaining_rel_indices = list(
                set(all_rel_indices) - set(filtered_rel_indices)
            )
            if len(remaining_rel_indices) > 0:
                additional_count = k - n_filtered
                additional_rel_indices = np.random.choice(
                    remaining_rel_indices, additional_count, replace=False
                )
                selected_rel_indices.extend(additional_rel_indices)
            selected_rel_indices = selected_rel_indices[:k]

        # Map back to absolute indices
        selected_indices = [remaining_indices[j] for j in selected_rel_indices]

    sampled_atoms = [valid_atoms[i] for i in selected_indices]

    # Compute Silhouette Score for the selected frames (on full features for consistency)
    if len(selected_indices) > 1:
        selected_features = combined_features_scaled[selected_indices]
        rel_cluster_labels = km.predict(selected_features)
        if len(set(rel_cluster_labels)) > 1:
            sil_score_selected = silhouette_score(selected_features, rel_cluster_labels)
            print(
                f"Silhouette Score (Selected {len(selected_indices)} Frames): {sil_score_selected:.3f}"
            )
        else:
            print(
                "Warning: Insufficient clusters in selected frames for Silhouette Score."
            )

    # Save dataset
    ase_write(output_traj_file, sampled_atoms)
    print(f"Successfully wrote {len(sampled_atoms)} frames to {output_traj_file}")

    # Return selected steps for exclusion
    selected_steps = {valid_atoms[i].info["step"] for i in selected_indices}

    return selected_steps


def sample_init_structs(
    valid_atoms,
    combined_features_scaled,
    output_dir,
    num_structs,
    random_seed,
):
    """Sample diverse initial structures for MLIP-MD."""
    os.makedirs(output_dir, exist_ok=True)
    # Cluster to find diverse reps
    k = num_structs * 2  # Oversample clusters
    km = KMeans(n_clusters=k, random_state=random_seed).fit(combined_features_scaled)
    cluster_labels = km.labels_
    unique_clusters = np.unique(cluster_labels)
    selected_indices = []
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_labels == cluster)[0]
        selected = np.random.choice(cluster_idx)
        selected_indices.append(selected)
        if len(selected_indices) >= num_structs:
            break
    # If fewer clusters, supplement random
    if len(selected_indices) < num_structs:
        additional = np.random.choice(
            len(valid_atoms), num_structs - len(selected_indices), replace=False
        )
        selected_indices.extend(additional)

    selected_atoms = [valid_atoms[i] for i in selected_indices[:num_structs]]
    for i, atoms in enumerate(selected_atoms):
        output_file = os.path.join(output_dir, f"init_struct_{i}.extxyz")
        ase_write(output_file, atoms)
        print(f"Saved initial structure {i} to {output_file}")

    return [
        os.path.join(output_dir, f"init_struct_{i}.extxyz") for i in range(num_structs)
    ]


if __name__ == "__main__":
    # Process full trajectory once
    print("--- Processing full trajectory ---")
    combined_data = {}
    for i, run_info in enumerate(RUNS):
        pos_file, force_file, energy_file = (
            run_info["pos"],
            run_info["frc"],
            run_info["ener"],
        )
        start_step, end_step = run_info["start_step"], run_info["end_step"]

        print(f" > Processing run {i + 1}: {os.path.basename(pos_file)}")
        try:
            pos_frames = read_xyz_frames(pos_file)
            force_frames = read_xyz_frames(force_file)
            energies_data = np.loadtxt(energy_file, skiprows=1, dtype=np.float32)
            energies = {int(row[0]): float(row[4]) for row in energies_data}

            for step in energies:
                if start_step <= step <= end_step:
                    if step in pos_frames and step in force_frames:
                        combined_data[step] = (
                            pos_frames[step],
                            force_frames[step],
                            energies[step],
                            step,
                        )
                    else:
                        print(f"Warning: Step {step} missing in pos or force files")
        except Exception as e:
            print(f"Error processing run {i + 1}: {e}", file=sys.stderr)
            sys.exit(1)

    frame_data = list(combined_data.values())
    all_atoms = process_trajectory_chunk(frame_data)

    # Filter unphysical
    energies = [atom.info["energy"] for atom in all_atoms]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    valid_atoms = [
        atom
        for atom in all_atoms
        # if mean_energy - 5 * std_energy
        # < atom.info["energy"]
        # < mean_energy + 5 * std_energy
    ]

    # Save full
    full_output = "aimd_trajectory_full.extxyz"
    ase_write(full_output, valid_atoms)
    print(f"Saved full trajectory ({len(valid_atoms)} frames) to {full_output}")

    # Split into temporal parts
    n_early = 30000
    n_late = 10000
    if len(valid_atoms) != n_early + n_late:
        print(f"Warning: Total frames {len(valid_atoms)}, expected {n_early + n_late}")
    early_atoms = valid_atoms[:n_early]
    late_atoms = valid_atoms[n_early:]

    # Compute features on early and late
    descriptors_early = compute_soap_descriptors(early_atoms)

    # Determine optimal PCA components on early
    explained_variance_ratio = []
    for n in range(1, 16):
        pca = PCA(n_components=n)
        pca.fit(descriptors_early)
        explained_variance_ratio.append(sum(pca.explained_variance_ratio_))
        if n > 1 and explained_variance_ratio[-1] - explained_variance_ratio[-2] < 0.01:
            optimal_n = n - 1
            break
    else:
        optimal_n = 15
    pca = PCA(n_components=optimal_n)
    reduced_early = pca.fit_transform(descriptors_early)
    energy_per_atom_early = np.array(
        [[atom.info["energy_per_atom"]] for atom in early_atoms]
    )
    max_force_early = np.array([[atom.info["max_force"]] for atom in early_atoms])
    combined_early = np.hstack((reduced_early, energy_per_atom_early, max_force_early))
    scaler = StandardScaler()
    combined_early_scaled = scaler.fit_transform(combined_early)

    # Features for late using same PCA and scaler
    descriptors_late = compute_soap_descriptors(late_atoms)
    reduced_late = pca.transform(descriptors_late)
    energy_per_atom_late = np.array(
        [[atom.info["energy_per_atom"]] for atom in late_atoms]
    )
    max_force_late = np.array([[atom.info["max_force"]] for atom in late_atoms])
    combined_late = np.hstack((reduced_late, energy_per_atom_late, max_force_late))
    combined_late_scaled = scaler.transform(combined_late)

    # Global excluded steps for early
    excluded_early = set()

    # Generate disjoint datasets for primary + ensembles (train/val from early)
    # Primary first
    print("\nGenerating dataset for primary model")
    primary_selected_steps = sample_dataset(
        early_atoms,
        combined_early_scaled,
        PRIMARY_TRAIN_VAL_FILE,
        PRIMARY_TRAIN_VAL_FRAMES,
        0,  # Seed for primary
        excluded_early,
    )
    excluded_early.update(primary_selected_steps)

    # Then ensembles
    for ensemble_id in range(NUM_ENSEMBLES):
        print(f"\nGenerating dataset for ensemble model {ensemble_id}")
        ensemble_selected_steps = sample_dataset(
            early_atoms,
            combined_early_scaled,
            f"aimd_trajectory_ensemble_{ensemble_id}_train_val.extxyz",
            ENSEMBLE_TRAIN_VAL_FRAMES,
            ensemble_id + 1,  # Unique seeds
            excluded_early,
        )
        excluded_early.update(ensemble_selected_steps)

    # Sample for true calib
    print("\nGenerating true calibration dataset")
    true_calib_selected_steps = sample_dataset(
        early_atoms,
        combined_early_scaled,
        TRUE_CALIB_FILE,
        TRUE_CALIB_FRAMES,
        999,  # some seed
        excluded_early,
    )

    # Sample combined test from late (no exclusions, separate pool)
    print("\nGenerating combined test dataset from last 5 ps")
    test_excluded = set()  # No exclusions for test
    test_selected_steps = sample_dataset(
        late_atoms,
        combined_late_scaled,
        FINAL_TEST_FILE,
        COMBINED_TEST_FRAMES,
        42,  # Fixed seed for reproducibility
        test_excluded,
    )

    # Sample initial structures for MLIP-MD from early (diverse)
    print("\nGenerating diverse initial structures for MLIP-MD")
    init_structs_dir = "init_structs"
    init_struct_files = sample_init_structs(
        early_atoms,
        combined_early_scaled,
        init_structs_dir,
        NUM_INIT_STRUCTS,
        123,  # Fixed seed
    )

    all_selected_steps = []
    all_selected_max_forces = []
    all_selected_energies_per_atom = []
    # Primary
    primary_train_val = read(PRIMARY_TRAIN_VAL_FILE, index=":")
    primary_steps = [frame.info["step"] for frame in primary_train_val]
    all_selected_steps.extend(primary_steps)
    all_selected_max_forces.extend(
        [frame.info["max_force"] for frame in primary_train_val]
    )
    all_selected_energies_per_atom.extend(
        [frame.info["energy_per_atom"] for frame in primary_train_val]
    )
    # Ensembles
    ensemble_sets = []
    for i in range(NUM_ENSEMBLES):
        train_val = read(f"aimd_trajectory_ensemble_{i}_train_val.extxyz", index=":")
        steps = [frame.info["step"] for frame in train_val]
        all_selected_steps.extend(steps)
        ensemble_sets.append(set(steps))
        all_selected_max_forces.extend([frame.info["max_force"] for frame in train_val])
        all_selected_energies_per_atom.extend(
            [frame.info["energy_per_atom"] for frame in train_val]
        )

    # Add combined test to metrics
    combined_test = read(FINAL_TEST_FILE, index=":")
    test_steps = [frame.info["step"] for frame in combined_test]
    all_selected_steps.extend(test_steps)
    all_selected_max_forces.extend([frame.info["max_force"] for frame in combined_test])
    all_selected_energies_per_atom.extend(
        [frame.info["energy_per_atom"] for frame in combined_test]
    )

    # Plot Force Distribution Frequency
    full_max_forces = [atom.info["max_force"] for atom in valid_atoms]
    plt.figure(figsize=(10, 6))
    plt.hist(full_max_forces, bins=50, alpha=0.5, label="Full Trajectory")
    plt.hist(
        all_selected_max_forces, bins=50, alpha=0.5, label="Selected Across Models"
    )
    plt.title("Max Force Distribution Frequency Across All Models")
    plt.xlabel("Max Force (kcal/mol/Å)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot Per-Atom Energy Distribution Frequency
    full_energies_per_atom = [atom.info["energy_per_atom"] for atom in valid_atoms]
    plt.figure(figsize=(10, 6))
    plt.hist(full_energies_per_atom, bins=50, alpha=0.5, label="Full Trajectory")
    plt.hist(
        all_selected_energies_per_atom,
        bins=50,
        alpha=0.5,
        label="Selected Across Models",
    )
    plt.title("Per-Atom Energy Distribution Frequency Across All Models")
    plt.xlabel("Energy per Atom (kcal/mol/atom)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Metrics
    total_unique = len(set(all_selected_steps))
    total_selections = len(all_selected_steps)
    coverage = total_unique / len(valid_atoms) * 100

    jaccards = []
    for i in range(NUM_ENSEMBLES):
        for j in range(i + 1, NUM_ENSEMBLES):
            intersection = len(ensemble_sets[i] & ensemble_sets[j])
            union = len(ensemble_sets[i] | ensemble_sets[j])
            jaccards.append(intersection / union if union > 0 else 0)

    avg_jaccard = np.mean(jaccards) if jaccards else 0
    print(f"Total unique frames across models: {total_unique}")
    print(f"Coverage of full trajectory: {coverage:.2f}%")
    print(f"Average Jaccard similarity between ensemble pairs: {avg_jaccard:.4f}")
