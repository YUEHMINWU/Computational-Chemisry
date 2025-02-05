import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import numpy
import MDAnalysis.analysis.distances as distances
import numpy as np
import os
import argparse
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

############################# System Density Distribution ################################
class SDD:
    def __init__(self, u, Startframe, Endframe, component , hist_range, num_bins):
        self.u = u
        self.Startframe= Startframe
        self.Endframe= Endframe
        self.component = component
        self.hist_range= hist_range
        self.num_bins = num_bins

    def cal_dens(self):
        histograms = []
        
        vec_x = self.u.trajectory[0].triclinic_dimensions[0]
        vec_y = self.u.trajectory[0].triclinic_dimensions[1]
        area = LA.norm(np.cross(vec_x, vec_y))

        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            component_z_coords = np.array(self.component.positions[:, 2])
            accumulated_histogram = np.zeros(self.num_bins)
            hist, bin_edges = np.histogram(
                component_z_coords,
                bins=self.num_bins,
                range=self.hist_range
            )
            histograms.append(hist)
        
        accumulated_histogram = np.sum(histograms, axis=0) # sum the values of each bin across all frames
        avg_histogram = accumulated_histogram / (self.Endframe- self.Startframe)
        bin_width = (max(self.hist_range)-min(self.hist_range))/self.num_bins
        density= avg_histogram / (area*bin_width)
        return density

    def cal_dens_bulk(self, n_ionpair, mw_ionpair):  
        histograms = []
        vol_list = []
        # NPT simulation shall notice the box size change with trajectory so the dimension of box should use the start frame doesn't have any big change in box size.
        conv=10*mw_ionpair*n_ionpair*(1/6.023)

        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            vec_x = self.u.trajectory.dimensions[0]
            vec_y = self.u.trajectory.dimensions[1]
            vec_z = self.u.trajectory.dimensions[2]
            area = vec_x * vec_y
            vol = area * vec_z
            vol_list.append(vol)
            component_z_coords = np.array(self.component.positions[:, 2])
            accumulated_histogram = np.zeros(self.num_bins)
            hist, bin_edges = np.histogram(
                component_z_coords,
                bins=self.num_bins,
                range=self.hist_range
            )
            histograms.append(hist)
        
        accumulated_histogram = np.sum(histograms, axis=0) # sum the values of each bin across all frames
        avg_histogram = accumulated_histogram / (self.Endframe- self.Startframe)
        bin_width = (max(self.hist_range)-min(self.hist_range))/self.num_bins
        density= avg_histogram / (area*bin_width)
        avg_dens = np.sum(density[(self.num_bins // 2 - 50):(self.num_bins // 2 + 50)]) / 100
        print(avg_dens)
        vol = np.array(vol_list)
        avg_vol = np.mean(vol)
        print(conv/(avg_vol))
        return density

class NTD:
    def __init__(self, filename):
        self.filename = filename
    
    def NTDV_list(self):
        NTDV_list = []
        with open(self.filename, 'r') as NTD_data:
            for line in NTD_data:
                value = float(line.strip())
                NTDV_list.append(value)
        return NTDV_list

############################# Density Distribution Plot Class ####################################333
class DDP:
    def __init__(self, component_name, left_data, right_data, bin_width, x_0, x_l, y_0, y_l):
        self.component_name = component_name
        self.left_data = left_data
        self.right_data = right_data
        self.bin_width = bin_width
        self.x_0 = x_0
        self.x_l = x_l
        self.y_0 = y_0
        self.y_l = y_l

        
        
    def w_m_plot(self, voltages, colors, linestyles):
        
        plt.figure(figsize=(12, 4))
        
        left_subplot = plt.subplot(1, 2, 1)
        left_subplot.set_xlabel("Distance from the cathode ($\AA$)")
        left_subplot.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        left_subplot.set_title(f'{self.component_name} Density Profile')
        left_subplot.set_xlim(self.x_0, self.x_l)
        left_subplot.set_ylim(self.y_0,self.y_l)
        left_subplot.spines['right'].set_visible(False)
        left_subplot.spines['top'].set_visible(False)

        right_subplot = plt.subplot(1, 2, 2)
        right_subplot.set_xlabel("Distance from the anode ($\AA$)")
        right_subplot.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        right_subplot.set_title(f'{self.component_name} Density Profile')
        right_subplot.set_xlim(self.x_0, self.x_l)
        right_subplot.set_ylim(self.y_0,self.y_l)
        right_subplot.spines['left'].set_visible(False)
        right_subplot.spines['top'].set_visible(False)
            
        for densities, subplot in [(self.left_data, left_subplot), (self.right_data, right_subplot)]:
            for i in range(len(voltages)):
                subplot.plot(
                    np.arange(0, len(densities[i])) * self.bin_width,
                    densities[i],
                    lw="1.0",
                    label=f'{voltages[i]/2} / -{voltages[i]/2} V',
                    linestyle=linestyles[i],
                    color=colors[i]
                )
        
        
        plt.gca().invert_xaxis()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        # Add legends
        left_subplot.legend(fontsize=10, frameon=False, ncol=2)
        # right_subplot.legend(frameon=False)
        plt.text(0.5, 0.5, '//',fontsize=15, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)

        # Show the plots
        plt.tight_layout()
        plt.show()
        
    def f_g_plot(self, molename, y_shift, V_drops_cat, colors, linestyles):
        
        plt.figure(figsize=(12, 4))  # Increase figure height to accommodate both subplots
        left_ax = plt.subplot(1, 2, 1)
        right_ax = plt.subplot(1, 2, 2)  # Share y-axis with the left subplot

        left_ax.set_xlabel("Distance from the cathode ($\AA$)")
        left_ax.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        left_ax.set_title(f'{molename} Density Profiles')
        left_ax.set_xlim(self.x_0, self.x_l)
        left_ax.set_ylim(self.y_0, self.y_l)
        left_ax.spines['right'].set_visible(False)
        left_ax.spines['top'].set_visible(False)

        right_ax.set_xlabel("Distance from the anode ($\AA$)")
        right_ax.set_ylabel("Density $(N_{ion}/(\AA^{3}))$")
        right_ax.set_title(f'{molename} Density Profiles')
        right_ax.set_xlim(self.x_0, self.x_l)
        right_ax.set_ylim(self.y_0, self.y_l)
        right_ax.spines['left'].set_visible(False)
        right_ax.spines['top'].set_visible(False)
        right_ax.yaxis.tick_right()
        right_ax.yaxis.set_label_position("right")

        for k in range(len(V_drops)):
            for i in range(len(self.left_data)):
                labels = f'{V_drops[k]/2} / -{V_drops[k]/2} V' if i == 0 else None
                left_ax.plot(
                    np.arange(0, len(self.right_data[i][k])) * self.bin_width,
                    [val + i * y_shift for val in self.left_data[i][k]],
                    lw="1.0",
                    label= labels,
                    linestyle=linestyles[k],
                    color=colors[i]
                )

                right_ax.plot(
                    np.arange(0, len(self.right_data[i][k])) * self.bin_width,
                    [val + i * y_shift for val in self.right_data[i][k]],
                    lw="1.0",
                    linestyle=linestyles[k],
                    color=colors[i]
                )

        # Set plot properties
        plt.gca().invert_xaxis()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        left_ax.legend(fontsize=10, frameon=False, ncol=3)
        plt.text(0.5, 0.5, '//',fontsize=12, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        # Show the plot
        plt.tight_layout()
        plt.show()
        
class DDP_oneside:
    def __init__(self, bin_width, dmax, ylim):
        self.bin_width = bin_width
        self.dmax = dmax
        self.ylim = ylim

    def f_g_plot_oneside(self, y1, y2, y_shift, y_ticks, NTD_list, V_drops, colors, linestyles, fg_names):
        fig, ax = plt.subplots(figsize=(5.5, 5))

        for k in range(len(V_drops)):
            labels = f'{ V_drops[k] } V' 
            for i in range(len(NTD_list)):
                label = labels if i == 0 else None
                plt.plot(
                    np.arange(0, len(NTD_list[i][k])) * self.bin_width,
                    [val + i * y_shift for val in NTD_list[i][k]],
                    lw="1.0",
                    label=label,
                    linestyle=linestyles[k],
                    color=colors[k]
                )

        plt.xlim(0.0, self.dmax)
        plt.ylim(0.0, self.ylim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(self.dmax/6))
        ax.yaxis.set_major_locator(plt.MultipleLocator(self.ylim/y_ticks))
        plt.axvline(x=0, color='lightgray', linewidth=20)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.text(12.5, y1, fg_names[0], fontsize=12)
        plt.text(12.5, y2, fg_names[1], fontsize=12)
        plt.xlabel("Distance from the electrode ($\\AA$)", fontsize = 12)
        plt.ylabel("Density $(N_{ion}/(\\AA^{3}))$",fontsize = 12)
        # plt.title(f'{molename} Density Profiles')
        plt.legend(frameon=False, fontsize=12)
        plt.show()
        
######################### RDFs Calculation Class for SC system #############################
class RDFs:
    def __init__(self, u, Startframe, Endframe, ion1, ion2, grp1, layer1_cutoff, n_bins): # 1 is for adosorption, 2 is for surrounding
        self.u= u
        self.Startframe= Startframe
        self.Endframe= Endframe
        self.ion1= ion1
        self.ion2= ion2
        self.grp1=  grp1
        self.layer1_cutoff= layer1_cutoff
        self.n_bins= n_bins

    def cal_rdfs_12(self, dmin, dmax):
        # Cation-anion
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
        
            # Cation around anion
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_crd, dmax , min_cutoff=dmin, box=ts.dimensions )
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12

    def cal_rdfs_12_ex(self, dmin, dmax, exclusion):
        # Cation-Cation
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
            ion2_resids = self.ion2.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_crd, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_12[i,0]]) for i in range(len(pairs_12))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion2_resids[pairs_12[i,1]]) for i in range(len(pairs_12)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_12 = dist_12[mask]
        
            # Cation around Cation
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12
            
    def cal_rdfs_11(self, dmin, dmax, exclusion):
        # Anion- anion
        rdf_ion1_ion1, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion1 = numpy.array(rdf_ion1_ion1,dtype=numpy.float64)
        rdf_ion1_ion1 *= 0
        n_ion_pairs_11 = 0.
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            pairs_11, dist_11 = distances.capped_distance(  ion1_layer1_pos, ion1_crd, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_11[i,0]]) for i in range(len(pairs_11))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion1_resids[pairs_11[i,1]]) for i in range(len(pairs_11)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_11 = dist_11[mask]
            new_rdf_11, edges = numpy.histogram(numpy.ravel(dist_11), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_11 += len(dist_11)
            new_rdf_11 = numpy.array(new_rdf_11 ,dtype=numpy.float64)
            rdf_ion1_ion1 += new_rdf_11

        # Normalize RDF
        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_11 = n_ion_pairs_11 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion1 = rdf_ion1_ion1 / n_ion_pairs_11 / (vol*frameCount)*vol_sphere
        newrdf_11 = numpy.array(rdf_ion1_ion1)

        return newrdf_11

################################# Decomposed RDFs in EDL #############################################
        
    def cal_dec_rdfs_12_inEDL(self, dmin, dmax):
        # Cation-anion
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            ion2_layer1_pos = []
            ion2_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            for ion2_i in self.ion2:
                #print(dir(ion1_i))
                ion2_i_crd = ion2_i.position
                ion2_i_residx = ion2_i.resid
                if (abs(float(ion2_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion2_layer1_pos.append(ion2_i.position)
                    ion2_layer1_resid.append(ion2_i.resid)
            
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            ion2_layer1_pos = numpy.array(ion2_layer1_pos,dtype=numpy.float64)
            ion2_layer1_resid = numpy.array(ion2_layer1_resid,dtype=numpy.float64)
        
            # Cation around anion
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_layer1_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12

    def cal_dec_rdfs_12_inEDL_ex(self, dmin, dmax, exclusion):
        # Cation-Cation
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            ion2_layer1_pos = []
            ion2_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            for ion2_i in self.ion2:
                #print(dir(ion1_i))
                ion2_i_crd = ion2_i.position
                ion2_i_residx = ion2_i.resid
                if (abs(float(ion2_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion2_layer1_pos.append(ion2_i.position)
                    ion2_layer1_resid.append(ion2_i.resid)
            
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            ion2_layer1_pos = numpy.array(ion2_layer1_pos,dtype=numpy.float64)
            ion2_layer1_resid = numpy.array(ion2_layer1_resid,dtype=numpy.float64)
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_layer1_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_12[i,0]]) for i in range(len(pairs_12))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion2_layer1_resid[pairs_12[i,1]]) for i in range(len(pairs_12)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_12 = dist_12[mask]
        
            # Cation around Cation
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12
            
    def cal_dec_rdfs_11_inEDL(self, dmin, dmax, exclusion):
        # Anion- anion
        rdf_ion1_ion1, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion1 = numpy.array(rdf_ion1_ion1,dtype=numpy.float64)
        rdf_ion1_ion1 *= 0
        n_ion_pairs_11 = 0.
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            pairs_11, dist_11 = distances.capped_distance(  ion1_layer1_pos, ion1_layer1_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_11[i,0]]) for i in range(len(pairs_11))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion1_layer1_resid[pairs_11[i,1]]) for i in range(len(pairs_11)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_11 = dist_11[mask]
            new_rdf_11, edges = numpy.histogram(numpy.ravel(dist_11), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_11 += len(dist_11)
            new_rdf_11 = numpy.array(new_rdf_11 ,dtype=numpy.float64)
            rdf_ion1_ion1 += new_rdf_11

        # Normalize RDF
        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_11 = n_ion_pairs_11 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion1 = rdf_ion1_ion1 / n_ion_pairs_11 / (vol*frameCount)*vol_sphere
        newrdf_11 = numpy.array(rdf_ion1_ion1)

        return newrdf_11

################################# Decomposed RDFs out EDL #############################################
        
    def cal_dec_rdfs_12_outEDL(self, dmin, dmax):
        # Cation-anion
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            ion2_layer1_pos = []
            ion2_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            for ion2_i in self.ion2:
                #print(dir(ion1_i))
                ion2_i_crd = ion2_i.position
                ion2_i_residx = ion2_i.resid
                if (abs(float(ion2_i_crd[2]) - float(group1_crd[0,2])) > self.layer1_cutoff):
                    ion2_layer1_pos.append(ion2_i.position)
                    ion2_layer1_resid.append(ion2_i.resid)
            
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            ion2_layer1_pos = numpy.array(ion2_layer1_pos,dtype=numpy.float64)
            ion2_layer1_resid = numpy.array(ion2_layer1_resid,dtype=numpy.float64)
        
            # Cation around anion
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_layer1_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12

    def cal_dec_rdfs_12_outEDL_ex(self, dmin, dmax, exclusion):
        # Cation-Cation
        rdf_ion1_ion2, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
        rdf_ion1_ion2 *= 0
        n_ion_pairs_12 = 0.
        
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            ion2_layer1_pos = []
            ion2_layer1_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            for ion2_i in self.ion2:
                #print(dir(ion1_i))
                ion2_i_crd = ion2_i.position
                ion2_i_residx = ion2_i.resid
                if (abs(float(ion2_i_crd[2]) - float(group1_crd[0,2])) > self.layer1_cutoff):
                    ion2_layer1_pos.append(ion2_i.position)
                    ion2_layer1_resid.append(ion2_i.resid)
            
            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            ion2_layer1_pos = numpy.array(ion2_layer1_pos,dtype=numpy.float64)
            ion2_layer1_resid = numpy.array(ion2_layer1_resid,dtype=numpy.float64)
            pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_layer1_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_12[i,0]]) for i in range(len(pairs_12))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion2_layer1_resid[pairs_12[i,1]]) for i in range(len(pairs_12)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_12 = dist_12[mask]
        
            # Cation around Cation
            new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_12 += len(dist_12)
            new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
            rdf_ion1_ion2 += new_rdf_12

        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_12 = n_ion_pairs_12 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
        newrdf_12 = numpy.array(rdf_ion1_ion2)

        return newrdf_12
            
    def cal_dec_rdfs_11_outEDL(self, dmin, dmax, exclusion):
        # Anion- anion
        rdf_ion1_ion1, edges = numpy.histogram([0], bins=self.n_bins, range=(dmin, dmax))
        rdf_ion1_ion1 = numpy.array(rdf_ion1_ion1,dtype=numpy.float64)
        rdf_ion1_ion1 *= 0
        n_ion_pairs_11 = 0.
        for ts in self.u.trajectory[self.Startframe:self.Endframe]:
            box = ts.dimensions
            group1_crd = self.grp1.positions
            ion1_crd = self.ion1.positions
            ion2_crd = self.ion2.positions
            ion1_resids = self.ion1.resids
        
            ion1_layer1_pos = []
            ion1_layer1_resid = []
            ion1_outlayer_pos = []
            ion1_outlayer_resid = []
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < self.layer1_cutoff):
                    ion1_layer1_pos.append(ion1_i.position)
                    ion1_layer1_resid.append(ion1_i.resid)
            for ion1_i in self.ion1:
                #print(dir(ion1_i))
                ion1_i_crd = ion1_i.position
                ion1_i_residx = ion1_i.resid
                if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) > self.layer1_cutoff):
                    ion1_outlayer_pos.append(ion1_i.position)
                    ion1_outlayer_resid.append(ion1_i.resid)

            ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
            ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
            ion1_outlayer_pos = numpy.array(ion1_outlayer_pos,dtype=numpy.float64)
            ion1_outlayer_resid = numpy.array(ion1_outlayer_resid,dtype=numpy.float64)
            pairs_11, dist_11 = distances.capped_distance(  ion1_layer1_pos, ion1_outlayer_pos, dmax , min_cutoff=dmin, box=ts.dimensions )
            if exclusion is not None:
                idxA = [ int(ion1_layer1_resid[pairs_11[i,0]]) for i in range(len(pairs_11))]
                idxA = numpy.array(idxA, dtype=numpy.float64)
                idxB = [int(ion1_outlayer_resid[pairs_11[i,1]]) for i in range(len(pairs_11)) ]
                idxB = numpy.array(idxB, dtype=numpy.float64)
                #print("idxA", idxA)
                #print("idxB", idxB)
                mask = numpy.where(idxA != idxB)[0]
                nomask = numpy.where(idxA == idxB)[0]
                #print("mask", mask, len(mask))
                #print("nomask", nomask, len(nomask))
                dist_11 = dist_11[mask]
            new_rdf_11, edges = numpy.histogram(numpy.ravel(dist_11), bins=self.n_bins, range=(dmin, dmax))
            n_ion_pairs_11 += len(dist_11)
            new_rdf_11 = numpy.array(new_rdf_11 ,dtype=numpy.float64)
            rdf_ion1_ion1 += new_rdf_11

        # Normalize RDF
        frameCount= self.Endframe - self.Startframe
        n_ion_pairs_11 = n_ion_pairs_11 / frameCount
        vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
        vol_sphere =  (4 / 3) * numpy.pi *dmax**3
        rdf_ion1_ion1 = rdf_ion1_ion1 / n_ion_pairs_11 / (vol*frameCount)*vol_sphere
        newrdf_11 = numpy.array(rdf_ion1_ion1)

        return newrdf_11

class RDFs_dat:
    def __init__(self, filename):
        self.filename = filename
    
    def dat_list(self):
        dat_list = []
        with open(self.filename, 'r') as RDFs_data:
            for line in RDFs_data:
                value = float(line.strip())
                dat_list.append(value)
        return dat_list
        
class RPS:
    def __init__(self, adsorbed_atoms, surrounding_atoms):
        self.adsorbed_atoms= adsorbed_atoms
        self.surrounding_atoms= surrounding_atoms

        
    def plot_rdfs_12(self, dmax, ylim, edge, rdf, V_drops, colors, linestyles):
        
        fig, ax = plt.subplots(figsize=(5.5, 5)) 
    
        for idx, rdf in enumerate(rdf):
            labels = f'{ V_drops[idx]} V'
            plt.plot(edge, rdf, label=labels, color=colors[idx], linestyle=linestyles[idx],lw=1.2)
    
        plt.xlim(0.0, dmax)
        plt.ylim(0.0, ylim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(dmax/6))   
        ax.yaxis.set_major_locator(plt.MultipleLocator(ylim/8))
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel('r $(\AA)$', fontsize=12)
        plt.ylabel('g(r)', fontsize=12)
        plt.title(f'RDF of {self.surrounding_atoms} around adsorbed {self.adsorbed_atoms} ')
        plt.legend(frameon=False, fontsize=12)
        plt.show()
        
    def plot_rdfs_11(self, dmax, ylim, edge, rdf, V_drops, colors, linestyles):

        fig, ax = plt.subplots(figsize=(5.5, 5)) 
        
        for idx, rdf in enumerate(rdf):
            labels = f'{ V_drops[idx]} V'
            plt.plot(edge, rdf, label=labels, color=colors[idx], linestyle=linestyles[idx],lw=1.2)
    
        plt.xlim(0.0, dmax)
        plt.ylim(0.0, ylim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(dmax/6))   
        ax.yaxis.set_major_locator(plt.MultipleLocator(ylim/8))
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel('r $(\AA)$', fontsize=12)
        plt.ylabel('g(r)', fontsize=12)
        plt.title(f'RDF of {self.surrounding_atoms} around adsorbed {self.adsorbed_atoms} ')
        plt.legend(frameon=False, fontsize=12)
        plt.show()

################################# InterRDFs from Mdanalysis #######################################################3
# class RDFs:
#     def __init__(self, u, Startframe, Endframe, adsorbed_atoms, surrounded_atoms, electrode_z_positions, first_peak_dist):
#         self.u= u
#         self.Startframe= Startframe
#         self.Endframe= Endframe
#         self.adsorbed_atoms= adsorbed_atoms
#         self.surrounded_atoms= surrounded_atoms
#         self.electrode_z_positions=  electrode_z_positions
#         self.first_peak_dist= first_peak_dist

#     def cal_rdfs_CaA(self, irdf_range):
#         rdf_results_CaA = []
#         for ts in self.u.trajectory[self.Startframe:self.Endframe]:
#             adsorbed_atoms_z_coords = np.array(self.adsorbed_atoms.positions[:, 2])
#             adsorbed_atoms_apt = [apt_prober for apt_prober in adsorbed_atoms_z_coords if apt_prober <= (min(self.electrode_z_positions) + self.first_peak_dist)]
#             z_mask = np.array([apt_prober in adsorbed_atoms_apt for apt_prober in adsorbed_atoms_z_coords])
#             filtered_atoms = self.adsorbed_atoms.atoms[z_mask]
#             irdf = rdf.InterRDF(self.surrounded_atoms, filtered_atoms, nbins=120, range=irdf_range)
#         irdf.run()
#         rdf_results_CaA.append(irdf)
#         return rdf_results_CaA
    
    
#     def cal_rdfs_AaA(self, irdf_range):
#         rdf_results_AaA = []
#         for ts in self.u.trajectory[self.Startframe:self.Endframe]:
#             adsorbed_atoms_z_coords = np.array(self.adsorbed_atoms.positions[:, 2])
#             adsorbed_atoms_apt = [apt_prober for apt_prober in adsorbed_atoms_z_coords if apt_prober <= (min(self.electrode_z_positions) + self.first_peak_dist)]
#             z_mask = np.array([apt_prober in adsorbed_atoms_apt for apt_prober in adsorbed_atoms_z_coords])
#             filtered_atoms = self.adsorbed_atoms.atoms[z_mask]
#             # irdf = rdf.InterRDF(filtered_atoms, filtered_atoms, nbins=120, range=irdf_range)
#             irdf = rdf.InterRDF(self.surrounded_atoms, filtered_atoms, nbins=120, range=irdf_range)
#         irdf.run()
#         rdf_results_AaA.append(irdf)
#         return rdf_results_AaA
    
    
#     def cal_rdfs_AaC(self, irdf_range):
#         rdf_results_AaC = []
#         for ts in self.u.trajectory[self.Startframe:self.Endframe]:
#             adsorbed_atoms_z_coords = np.array(self.adsorbed_atoms.positions[:, 2])
#             adsorbed_atoms_apt = [apt_prober for apt_prober in adsorbed_atoms_z_coords if apt_prober >= (max(self.electrode_z_positions) - self.first_peak_dist)]
#             z_mask = np.array([apt_prober in adsorbed_atoms_apt for apt_prober in adsorbed_atoms_z_coords])
#             filtered_atoms = self.adsorbed_atoms.atoms[z_mask]
#             irdf = rdf.InterRDF(self.surrounded_atoms, filtered_atoms, nbins=120, range=irdf_range)
#         irdf.run()
#         rdf_results_AaC.append(irdf)
#         return rdf_results_AaC
    
#     def cal_rdfs_CaC(self, irdf_range):
#         rdf_results_CaC = []
#         for ts in self.u.trajectory[self.Startframe:self.Endframe]:
#             adsorbed_atoms_z_coords = np.array(self.adsorbed_atoms.positions[:, 2])
#             adsorbed_atoms_apt = [apt_prober for apt_prober in adsorbed_atoms_z_coords if apt_prober >= (max(self.electrode_z_positions) - self.first_peak_dist)]
#             z_mask = np.array([apt_prober in adsorbed_atoms_apt for apt_prober in adsorbed_atoms_z_coords])
#             filtered_atoms = self.adsorbed_atoms.atoms[z_mask]
#             # irdf = rdf.InterRDF(filtered_atoms, filtered_atoms, nbins=120, range=irdf_range)
#             irdf = rdf.InterRDF(self.surrounded_atoms, filtered_atoms, nbins=120, range=irdf_range)
#         irdf.run()
#         rdf_results_CaC.append(irdf)
#         return rdf_results_CaC

# ############################### Anion-Cation RDFs Plots Class ########################################3
# class RPS:
#     def __init__(self, cation_name, anion_name):
#         self.cation_name= cation_name
#         self.anion_name= anion_name

#     def plot_rdfs_CaA(self, rdf_results_CaA, xlim, ylim, voltages, colors, linestyles):

#             fig, ax = plt.subplots() 

#             for idx, irdf in enumerate(rdf_results_CaA):
#                 plt.plot(irdf.results.bins, irdf.results.rdf, label=f'{voltages[idx]} V', color=colors[idx], linestyle=linestyles[idx],lw=1.2)

#             plt.xlim(0.0, xlim)
#             plt.ylim(0.0, ylim)
#             ax.xaxis.set_major_locator(plt.MultipleLocator(1))   
#             ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#             plt.xlabel('r $(\AA)$')
#             plt.ylabel('g(r)')
#             plt.title(f'RDF of {self.cation_name} around adsorbed {self.anion_name} ')
#             plt.legend(frameon=False)
#             plt.show()
    
#     def plot_rdfs_AaA(self, rdf_results_AaA, xlim, ylim, voltages, colors, linestyles):

#             fig, ax = plt.subplots() 

#             for idx, irdf in enumerate(rdf_results_AaA):
#                 plt.plot(irdf.results.bins, irdf.results.rdf, label=f'{voltages[idx]} V', color=colors[idx], linestyle=linestyles[idx],lw=1.2)

#             plt.xlim(0.0, xlim)
#             plt.ylim(0.0, ylim)
#             ax.xaxis.set_major_locator(plt.MultipleLocator(1))   
#             ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#             plt.xlabel('r $(\AA)$')
#             plt.ylabel('g(r)')
#             plt.title(f'RDF of {self.anion_name} around adsorbed {self.anion_name} ')
#             plt.legend(frameon=False)
#             plt.show()
        
#     def plot_rdfs_AaC(self, rdf_results_AaC, xlim, ylim, voltages, colors, linestyles):
        
#         fig, ax = plt.subplots()
#         for idx, irdf in enumerate(rdf_results_AaC): 
#             plt.plot(irdf.results.bins, irdf.results.rdf, label=f'{voltages[idx]} V', color=colors[idx], linestyle=linestyles[idx],lw=1.2)

#         plt.xlim(0, xlim)
#         plt.ylim(0, ylim)
#         ax.xaxis.set_major_locator(plt.MultipleLocator(1))   
#         ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#         plt.xlabel('r $(\AA)$')
#         plt.ylabel('g(r)')
#         plt.title(f'RDF of {self.anion_name} around adsorbed {self.cation_name} ')
#         plt.legend(frameon=False)
#         plt.show()
   
#     def plot_rdfs_CaC(self, rdf_results_CaC, xlim, ylim, voltages, colors, linestyles):
        
#         fig, ax = plt.subplots()
#         for idx, irdf in enumerate(rdf_results_CaC): 
#             plt.plot(irdf.results.bins, irdf.results.rdf, label=f'{voltages[idx]} V', color=colors[idx], linestyle=linestyles[idx],lw=1.2)

#         plt.xlim(0, xlim)
#         plt.ylim(0, ylim)
#         ax.xaxis.set_major_locator(plt.MultipleLocator(1))   
#         ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#         plt.xlabel('r $(\AA)$')
#         plt.ylabel('g(r)')
#         plt.title(f'RDF of {self.cation_name} around adsorbed {self.cation_name} ')
#         plt.legend(frameon=False)
#         plt.show()