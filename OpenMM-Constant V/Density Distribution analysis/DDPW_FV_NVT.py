import MDAnalysis as mda
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot

class DensityDistributionPlotter:
    def __init__(self, u, component, component_name , electrode_z_positions, num_bins, midpoint):
        self.u = u
        self.component = component
        self.electrode_z_positions = electrode_z_positions
        self.num_bins = num_bins
        self.midpoint = midpoint
        self.left_densities = []
        self.right_densities = []
        self.component_name = component_name

    def calculate_density(self):
        densities = []
        for ts in self.u.trajectory:
            component_z_coords = np.array(self.component.positions[:, 2])
            hist, bin_edges = np.histogram(
                component_z_coords,
                bins=self.num_bins,
                range=(min(self.electrode_z_positions), max(self.electrode_z_positions))
            )
            bin_width = (max(self.electrode_z_positions) - min(self.electrode_z_positions)) / self.num_bins
            density = hist / (bin_width * len(component_z_coords))
            left_density = density[bin_edges[:-1] < self.midpoint]
            right_density = density[bin_edges[:-1] >= self.midpoint]
            densities.append((left_density, right_density))
        return densities
    
    def write_density(self,directory, filename):
        densities = self.calculate_density()
        left_density = densities[-1][0]
        right_density = densities[-1][1]
        bin_width = (max(self.electrode_z_positions) - min(self.electrode_z_positions)) / self.num_bins
        distances_from_cathode = np.arange(0, len(left_density)) * bin_width
        distances_from_anode = np.arange(0, len(right_density)) * bin_width
        
        filepath = os.path.join(directory, filename)

        with open(filepath, 'w') as f:
            # f.write("# Cathode Data\n")
            f.write("Distance from cathode (Å)\tDensity from cathode (N_ion/Å^3)\n")
            for distance, density in zip(distances_from_cathode, left_density):
                f.write(f"{distance:.3f}\t{density:.6f}\n")
            # f.write("\n# Anode Data\n")
            f.write("Distance from anode (Å)\tDensity from anode (N_ion/Å^3)\n")
            for distance, density in zip(distances_from_anode, right_density):
                f.write(f"{distance:.3f}\t{density:.6f}\n")
        
#         data = {
#             'Distance from cathode (Å)': distances_from_cathode,
#             'Density from cathode (N_ion/Å^3)': left_density,
#             'Distance from anode (Å)': distances_from_anode,
#             'Density from anode (N_ion/Å^3)': right_density
#         }
        
#         df = pd.DataFrame(data)

#         # To save the DataFrame to a separate CSV file (if needed):
#         df.to_csv(filepath, sep='\t', index=False)
        

    def plot_density_profiles(self):
        densities = self.calculate_density()
        left_density = densities[-1][0]
        right_density = densities[-1][1]

        plt.figure(figsize=(12, 4))
        left_subplot = plt.subplot(1, 2, 1)
        right_subplot = plt.subplot(1, 2, 2)
        
        bin_width = (max(self.electrode_z_positions) - min(self.electrode_z_positions)) / self.num_bins
        left_subplot.plot(
            np.arange(0, len(left_density)) * bin_width,
            left_density,
            lw="1.0",
            label= self.component_name
        )

        left_subplot.set_xlabel("Distance from the cathode ($\AA$)")
        left_subplot.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        left_subplot.set_ylim(0, 0.05)
        left_subplot.spines['right'].set_visible(False)
        left_subplot.spines['top'].set_visible(False)
        left_subplot.legend(fontsize=7)

        right_subplot.plot(
            np.arange(0, len(right_density)) * bin_width,
            right_density[::-1],
            lw="1.0",
            label=self.component_name
        )

        right_subplot.set_xlabel("Distance from the anode ($\AA$)")
        right_subplot.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        right_subplot.set_ylim(0, 0.05)
        right_subplot.spines['left'].set_visible(False)
        right_subplot.spines['top'].set_visible(False)
        right_subplot.legend(fontsize=7)

        plt.gca().invert_xaxis()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.tight_layout()
        plt.show()