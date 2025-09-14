import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for KDE plots
from scipy.stats import gaussian_kde

# Load data
PARITY_DATA_FILE = "temp_parity_iter_0.npy"
# UNC_DATA_FILE = "unc_rmse_plot_iter_0.npy"
parity_results = np.load(PARITY_DATA_FILE, allow_pickle=True).item()
# unc_results = np.load(UNC_DATA_FILE, allow_pickle=True).item()
shifted_true_energy = parity_results["true_energy"] - 7892
shifted_pred_energy = parity_results["pred_energy"] - 7892
energy_error = shifted_pred_energy - shifted_true_energy
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
# Create Figure for parity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
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
ax1.set_xlabel("DFT Energy [7892 kcal/mol/atom]")
ax1.set_ylabel("MLIP energy error [kcal/mol/atom]")
ax1.set_title("Energy Error Correlation")
ax1.grid(False, linestyle="--", alpha=0.6)
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
ax2.grid(False, linestyle="--", alpha=0.6)
ax2.text(
    0.05,
    0.10,
    f"RMSE = {force_rmse:.4f} kcal/mol·Å",
    transform=ax2.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Energy Parity with Marginals
g_energy = sns.JointGrid(
    x=parity_results["true_energy"], y=parity_results["pred_energy"], height=6, ratio=5
)
# Compute 2D KDE for energy parity
xy_energy_parity = np.vstack(
    [parity_results["true_energy"], parity_results["pred_energy"]]
)
z_energy_parity = gaussian_kde(xy_energy_parity)(xy_energy_parity)
# Normalize density for better color mapping
z_energy_parity = z_energy_parity / np.max(z_energy_parity)
# Energy Parity Plot with scatterplot colored by density
sns.scatterplot(
    x=parity_results["true_energy"],
    y=parity_results["pred_energy"],
    hue=z_energy_parity,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=g_energy.ax_joint,
    legend=False,
)
# Add y = x dashed line
# min_val = min(
# np.min(parity_results["true_energy"]), np.min(parity_results["pred_energy"])
# )
# max_val = max(
# np.max(parity_results["true_energy"]), np.max(parity_results["pred_energy"])
# )
# g_energy.ax_joint.plot(
# [min_val, max_val], [min_val, max_val], color="r", linestyle="--", alpha=0.75
# )
g_energy.ax_joint.ticklabel_format(
    style="plain", axis="both"
)  # Disable scientific notation
g_energy.set_axis_labels(
    "DFT Energy per Atom [kcal/mol/atom]", "MLIP Energy per Atom [kcal/mol/atom]"
)
# g_energy.ax_joint.set_title("Energy Parity")
g_energy.ax_joint.grid(False, linestyle="--", alpha=0.6)
# Add marginal KDEs (or use histplot for histograms)
sns.kdeplot(
    x=parity_results["true_energy"], fill=True, ax=g_energy.ax_marg_x, color="blue"
)
sns.kdeplot(
    y=parity_results["pred_energy"], fill=True, ax=g_energy.ax_marg_y, color="green"
)
# Add text for MAE and RMSE
g_energy.ax_joint.text(
    0.05,
    0.95,
    f"Energy MAE: {energy_mae:.4f} kcal/mol\nEnergy RMSE: {energy_rmse:.4f} kcal/mol\nR²: {r2_energy:.4f}",
    transform=g_energy.ax_joint.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.show()
# Force Parity with Marginals
g_force = sns.JointGrid(
    x=parity_results["true_force"], y=parity_results["pred_force"], height=6, ratio=5
)
# Compute 2D KDE for force parity
xy_force_parity = np.vstack(
    [parity_results["true_force"], parity_results["pred_force"]]
)
z_force_parity = gaussian_kde(xy_force_parity)(xy_force_parity)
# Normalize density for better color mapping
z_force_parity = z_force_parity / np.max(z_force_parity)
# Force Parity Plot with scatterplot colored by density
sns.scatterplot(
    x=parity_results["true_force"],
    y=parity_results["pred_force"],
    hue=z_force_parity,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=g_force.ax_joint,
    legend=False,
)
# Add y = x dashed line
# min_val_f = min(
# np.min(parity_results["true_force"]), np.min(parity_results["pred_force"])
# )
# max_val_f = max(
# np.max(parity_results["true_force"]), np.max(parity_results["pred_force"])
# )
# g_force.ax_joint.plot(
# [min_val_f, max_val_f],
# [min_val_f, max_val_f],
# color="r",
# linestyle="--",
# alpha=0.75,
# )
g_force.set_axis_labels("DFT Force [kcal/mol·Å]", "MLIP Force [kcal/mol·Å]")
# g_force.ax_joint.set_title("Force Parity")
g_force.ax_joint.grid(False, linestyle="--", alpha=0.6)
# Add marginal KDEs
sns.kdeplot(
    x=parity_results["true_force"], fill=True, ax=g_force.ax_marg_x, color="blue"
)
sns.kdeplot(
    y=parity_results["pred_force"], fill=True, ax=g_force.ax_marg_y, color="green"
)
# Add text for MAE and RMSE
g_force.ax_joint.text(
    0.05,
    0.95,
    f"Force MAE: {force_mae:.4f} kcal/mol·Å\nForce RMSE: {force_rmse:.4f} kcal/mol·Å\nR²: {r2_force:.4f}",
    transform=g_force.ax_joint.transAxes,
    fontsize=12,
    verticalalignment="top",
)
plt.show()
# # Uncertainty vs RMSE plot
# fig_unc = plt.figure(figsize=(8, 6))
# ax_unc = fig_unc.add_subplot(1, 1, 1)
# # Compute 2D KDE for cal_unc vs rmse
# xy_unc_rmse = np.vstack(
# [unc_results["atomic_force_rmse"], unc_results["atomic_force_uncertainty"]]
# )
# z_unc = gaussian_kde(xy_unc_rmse)(xy_unc_rmse)
# # Normalize density
# z_unc = z_unc / np.max(z_unc)
# # Scatter with KDE coloring
# sns.scatterplot(
# x=unc_results["atomic_force_rmse"],
# y=unc_results["atomic_force_uncertainty"],
# hue=z_unc,
# palette="viridis",
# alpha=0.5,
# s=10,
# ax=ax_unc,
# legend=False,
# )
# ax_unc.plot(
# [0, max(unc_results["atomic_force_rmse"])],
# [0, max(unc_results["atomic_force_rmse"])],
# color="r",
# linestyle="--",
# alpha=0.75,
# ) # y=x line
# ax_unc.set_title("Uncertainty vs Force MAE per Atom")
# ax_unc.set_xlabel("Force MAE per Atom [kcal/mol·Å]")
# ax_unc.set_ylabel("Uncertainty [kcal/mol·Å]")
# ax_unc.grid(False, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()
