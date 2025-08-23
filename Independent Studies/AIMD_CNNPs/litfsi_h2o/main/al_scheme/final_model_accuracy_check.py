import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for KDE plots
from scipy.stats import gaussian_kde

# Load data
PARITY_DATA_FILE = "temp_parity.npy"
UNC_DATA_FILE = "unc_data_iter_0.npy"

parity_results = np.load(PARITY_DATA_FILE, allow_pickle=True).item()
unc_results = np.load(UNC_DATA_FILE, allow_pickle=True).item()

shifted_true_energy = parity_results["true_energy"] - 7892
shifted_pred_energy = parity_results["pred_energy"] - 7892
energy_error = shifted_pred_energy - shifted_true_energy
force_error = parity_results["pred_force"] - parity_results["true_force"]

# Calculate errors for display
energy_mae = np.mean(np.abs(energy_error))
force_rmse = np.sqrt(np.mean(force_error**2))

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
#     [np.min(parity_results["true_energy"]), np.max(parity_results["true_energy"])]
# )
# ax1.set_ylim([-max_abs_energy_error, max_abs_energy_error])  # Symmetric y-axis
ax1.set_xlabel("DFT Energy [7892 kcal/mol/atom]")
ax1.set_ylabel("MLIP energy error [kcal/mol/atom]")
ax1.set_title("Energy Error Correlation")
ax1.grid(True, linestyle="--", alpha=0.6)
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
#     [np.min(parity_results["true_force"]), np.max(parity_results["true_force"])]
# )
# ax2.set_ylim([-max_abs_force_error, max_abs_force_error])  # Symmetric y-axis
ax2.set_xlabel("DFT force [kcal/mol·Å]")
ax2.set_ylabel("MLIP force error [kcal/mol·Å]")
ax2.set_title("Force Error Correlation")
ax2.grid(True, linestyle="--", alpha=0.6)
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

# Uncertainty vs RMSE plot
fig_unc = plt.figure(figsize=(8, 6))
ax_unc = fig_unc.add_subplot(1, 1, 1)

# Compute 2D KDE for cal_unc vs rmse
xy_unc_rmse = np.vstack([unc_results["rmse_scores"], unc_results["cal_unc_scores"]])
z_unc = gaussian_kde(xy_unc_rmse)(xy_unc_rmse)
# Normalize density
z_unc = z_unc / np.max(z_unc)

# Scatter with KDE coloring
sns.scatterplot(
    x=unc_results["rmse_scores"],
    y=unc_results["cal_unc_scores"],
    hue=z_unc,
    palette="viridis",
    alpha=0.5,
    s=10,
    ax=ax_unc,
    legend=False,
)
ax_unc.plot(
    [0, max(unc_results["rmse_scores"])],
    [0, max(unc_results["rmse_scores"])],
    color="r",
    linestyle="--",
    alpha=0.75,
)  # y=x line
ax_unc.set_title("Uncertainty vs Force RMSE")
ax_unc.set_xlabel("Force RMSE [kcal/mol·Å]")
ax_unc.set_ylabel("Uncertainty [kcal/mol·Å]")
ax_unc.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
