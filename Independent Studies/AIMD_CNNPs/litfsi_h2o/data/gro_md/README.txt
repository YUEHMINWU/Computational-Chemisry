The commands of running the GROMACS MD simulation and please execute all the commands step by step.

1. Run energy minimizer:
gmx grompp -f em.mdp -c litfsi_h2o.pdb -p litfsi_h2o.top -o em.tpr -maxwarn 1
gmx mdrun -v -deffnm em

2. Run nvt simulation:
gmx grompp -f nvt.mdp -c em.gro -p litfsi_h2o.top -o nvt.tpr -maxwarn 1
gmx mdrun -v -deffnm nvt

3. Run npt simulation (you may add the openmpi command to call the mutiple cores to execute the job):
export OMP_NUM_THREADS=8
gmx grompp -f npt.mdp -c nvt.gro -p litfsi_h2o.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

3.1. Run in cluster:


4. Check the density of system:
gmx energy -f npt.edr -o density.xvg # Select 'Density' from the menu

5. Transform gro file (last frame) to final pdb file after npt simulation:
gmx editconf -f npt.gro -o final_npt.pdb # final_npt.pdb may be lack of element symbol, so please use pdb_ed.py before running aimd
