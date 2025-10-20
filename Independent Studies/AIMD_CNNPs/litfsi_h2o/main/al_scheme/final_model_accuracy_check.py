import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for KDE plots
from scipy.stats import linregress
from scipy.stats import gaussian_kde

# unc_score_data = np.load("unc_data_iter_1.npy", allow_pickle=True).item()
# unc_score = unc_score_data["max_cal_unc_scores"]
# print(min(unc_score), max(unc_score))

# Load data
PARITY_DATA_FILE = "temp_parity_iter_0.npy"
# UNC_DATA_FILE = "unc_rmse_plot_iter_1.npy"
parity_results = np.load(PARITY_DATA_FILE, allow_pickle=True).item()
# unc_results = np.load(UNC_DATA_FILE, allow_pickle=True).item()
energy_error = parity_results["pred_energy"] - parity_results["true_energy"]
force_error = parity_results["pred_force"] - parity_results["true_force"]
# Calculate errors for display
energy_mae = np.mean(np.abs(energy_error))
energy_rmse = np.sqrt(np.mean(energy_error**2))
force_mae = np.mean(np.abs(force_error))
force_rmse = np.sqrt(np.mean(force_error**2))
# Calculate R^2 for energy
true_energy = parity_results["true_energy"]
pred_energy = parity_results["pred_energy"]
ss_res_energy = np.sum((true_energy - pred_energy) ** 2)
ss_tot_energy = np.sum((true_energy - np.mean(true_energy)) ** 2)
r2_energy = 1 - (ss_res_energy / ss_tot_energy) if ss_tot_energy != 0 else np.nan
# Calculate R^2 for force
true_force = parity_results["true_force"]
pred_force = parity_results["pred_force"]
ss_res_force = np.sum((true_force - pred_force) ** 2)
ss_tot_force = np.sum((true_force - np.mean(true_force)) ** 2)
r2_force = 1 - (ss_res_force / ss_tot_force) if ss_tot_force != 0 else np.nan

# For atomic force norms
true_forces_reshaped = parity_results["true_force"].reshape(-1, 3)
pred_forces_reshaped = parity_results["pred_force"].reshape(-1, 3)
true_force_norms = np.linalg.norm(true_forces_reshaped, axis=1)
pred_force_norms = np.linalg.norm(pred_forces_reshaped, axis=1)
force_norm_error = pred_force_norms - true_force_norms
force_norm_mae = np.mean(np.abs(force_norm_error))
force_norm_rmse = np.sqrt(np.mean(force_norm_error**2))
# Calculate R^2 for force norms
ss_res_force_norm = np.sum((true_force_norms - pred_force_norms) ** 2)
ss_tot_force_norm = np.sum((true_force_norms - np.mean(true_force_norms)) ** 2)
r2_force_norm = (
    1 - (ss_res_force_norm / ss_tot_force_norm) if ss_tot_force_norm != 0 else np.nan
)

# Create Figure for parity
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle("Model Error vs. DFT", fontsize=16)
# Compute 2D KDE for energy error
xy_energy = np.vstack([parity_results["true_energy"], energy_error])
z_energy = gaussian_kde(xy_energy)(xy_energy)
# Normalize density for better color mapping
z_energy = z_energy / np.max(z_energy)
# Energy Error Plot with scatterplot colored by density
sns.scatterplot(
    x=parity_results["true_energy"],
    y=energy_error,
    hue=z_energy,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=ax1,
    legend=False,
)
ax1.axhline(y=0, color="r", linestyle="--", alpha=0.75)
ax1.ticklabel_format(style="plain", axis="x")  # Disable scientific notation
# max_abs_energy_error = np.max(np.abs(energy_error))
# ax1.set_xlim(
# [np.min(parity_results["true_energy"]), np.max(parity_results["true_energy"])]
# )
# ax1.set_ylim([-max_abs_energy_error, max_abs_energy_error]) # Symmetric y-axis
ax1.set_xlabel("DFT Energy [kcal/mol/atom]")
ax1.set_ylabel("MLIP energy error [kcal/mol/atom]")
ax1.set_title("Energy Error Correlation")
ax1.grid(False)
ax1.text(
    0.05,
    0.10,
    f"MAE/atom = {energy_mae:.4f} kcal/mol",
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment="top",
)
# Compute 2D KDE for force error
xy_force = np.vstack([parity_results["true_force"], force_error])
z_force = gaussian_kde(xy_force)(xy_force)
# Normalize density for better color mapping
z_force = z_force / np.max(z_force)
# Force Error Plot with scatterplot colored by density
sns.scatterplot(
    x=parity_results["true_force"],
    y=force_error,
    hue=z_force,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=ax2,
    legend=False,
)
ax2.axhline(y=0, color="r", linestyle="--", alpha=0.75)
# max_abs_force_error = np.max(np.abs(force_error))
# ax2.set_xlim(
# [np.min(parity_results["true_force"]), np.max(parity_results["true_force"])]
# )
# ax2.set_ylim([-max_abs_force_error, max_abs_force_error]) # Symmetric y-axis
ax2.set_xlabel("DFT force [kcal/mol·Å]")
ax2.set_ylabel("MLIP force error [kcal/mol·Å]")
ax2.set_title("Force Error Correlation")
ax2.grid(False)
ax2.text(
    0.05,
    0.10,
    f"RMSE = {force_rmse:.4f} kcal/mol·Å",
    transform=ax2.transAxes,
    fontsize=12,
    verticalalignment="top",
)
# Compute 2D KDE for force norm error
xy_force_norm = np.vstack([true_force_norms, force_norm_error])
z_force_norm = gaussian_kde(xy_force_norm)(xy_force_norm)
# Normalize density for better color mapping
z_force_norm = z_force_norm / np.max(z_force_norm)
# Force Norm Error Plot with scatterplot colored by density
sns.scatterplot(
    x=true_force_norms,
    y=force_norm_error,
    hue=z_force_norm,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=ax3,
    legend=False,
)
ax3.axhline(y=0, color="r", linestyle="--", alpha=0.75)
ax3.set_xlabel("DFT Atomic Force Magnitude [kcal/mol·Å]")
ax3.set_ylabel("MLIP Atomic Force Magnitude Error [kcal/mol·Å]")
ax3.set_title("Atomic Force Magnitude Error Correlation")
ax3.grid(False)
ax3.text(
    0.05,
    0.10,
    f"RMSE = {force_norm_rmse:.4f} kcal/mol·Å",
    transform=ax3.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Energy Parity with Marginals
fig_energy, ax_energy = plt.subplots(figsize=(6, 6))
# Energy Parity Plot with scatterplot in single color
sns.scatterplot(
    x=parity_results["true_energy"],
    y=parity_results["pred_energy"],
    color="green",
    alpha=0.5,
    s=10,
    ax=ax_energy,
    legend=False,
)
# Add y = x line extended to plot corners
xmin, xmax = ax_energy.get_xlim()
ymin, ymax = ax_energy.get_ylim()
line_min = max(xmin, ymin)
line_max = min(xmax, ymax)
ax_energy.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="gray",
    linestyle="-",
    alpha=0.5,
    lw=0.75,
)
ax_energy.ticklabel_format(style="plain", axis="both")  # Disable scientific notation
ax_energy.set_xlabel("DFT Energy per Atom [kcal/mol/atom]")
ax_energy.set_ylabel("MLIP Energy per Atom [kcal/mol/atom]")
# ax_energy.set_title("Energy Parity")
ax_energy.grid(False)
ax_energy.set_aspect("equal", adjustable="box")
# Enable all spines to form a square box
ax_energy.spines["top"].set_visible(True)
ax_energy.spines["right"].set_visible(True)
# Add text for MAE and RMSE
ax_energy.text(
    0.05,
    0.95,
    f"Energy MAE: {energy_mae:.4f} kcal/mol\nEnergy RMSE: {energy_rmse:.4f} kcal/mol\nR²: {r2_energy:.4f}",
    transform=ax_energy.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.show()
# Force Parity with Marginals
fig_force, ax_force = plt.subplots(figsize=(6, 6))
# Force Parity Plot with scatterplot in single color
sns.scatterplot(
    x=parity_results["true_force"],
    y=parity_results["pred_force"],
    color="blue",
    alpha=0.5,
    s=10,
    ax=ax_force,
    legend=False,
)
# Add y = x line extended to plot corners
xmin_f, xmax_f = ax_force.get_xlim()
ymin_f, ymax_f = ax_force.get_ylim()
line_min_f = max(xmin_f, ymin_f)
line_max_f = min(xmax_f, ymax_f)
ax_force.plot(
    [line_min_f, line_max_f],
    [line_min_f, line_max_f],
    color="gray",
    linestyle="-",
    alpha=0.5,
    lw=0.75,
)
ax_force.set_xlabel("DFT Force [kcal/mol·Å]")
ax_force.set_ylabel("MLIP Force [kcal/mol·Å]")
# ax_force.set_title("Force Parity")
ax_force.grid(False)
ax_force.set_aspect("equal", adjustable="box")
# Enable all spines to form a square box
ax_force.spines["top"].set_visible(True)
ax_force.spines["right"].set_visible(True)
# Add text for MAE and RMSE
ax_force.text(
    0.05,
    0.95,
    f"Force MAE: {force_mae:.4f} kcal/mol·Å\nForce RMSE: {force_rmse:.4f} kcal/mol·Å\nR²: {r2_force:.4f}",
    transform=ax_force.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.show()

# Atomic Force Magnitude Parity with Marginals
fig_force_norm, ax_force_norm = plt.subplots(figsize=(6, 6))
# Force Norm Parity Plot with scatterplot in single color
sns.scatterplot(
    x=true_force_norms,
    y=pred_force_norms,
    color="blue",
    alpha=0.5,
    s=10,
    ax=ax_force_norm,
    legend=False,
)
# Add y = x line extended to plot corners
xmin_fn, xmax_fn = ax_force_norm.get_xlim()
ymin_fn, ymax_fn = ax_force_norm.get_ylim()
line_min_fn = max(xmin_fn, ymin_fn)
line_max_fn = min(xmax_fn, ymax_fn)
ax_force_norm.plot(
    [line_min_fn, line_max_fn],
    [line_min_fn, line_max_fn],
    color="gray",
    linestyle="-",
    alpha=0.5,
    lw=0.75,
)
ax_force_norm.set_xlabel("DFT Atomic Force [kcal/mol·Å]")
ax_force_norm.set_ylabel("MLIP Atomic Force [kcal/mol·Å]")
# ax_force_norm.set_title("Atomic Force Parity")
ax_force_norm.grid(False)
ax_force_norm.set_aspect("equal", adjustable="box")
# Enable all spines to form a square box
ax_force_norm.spines["top"].set_visible(True)
ax_force_norm.spines["right"].set_visible(True)
# Add text for MAE and RMSE
ax_force_norm.text(
    0.05,
    0.95,
    f"Force MAE: {force_norm_mae:.4f} kcal/mol·Å\nForce RMSE: {force_norm_rmse:.4f} kcal/mol·Å\nR²: {r2_force_norm:.4f}",
    transform=ax_force_norm.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.show()
