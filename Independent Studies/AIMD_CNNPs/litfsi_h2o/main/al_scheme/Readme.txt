The order of execution: trajectory_utils.py -> al_iteration.py
In al_iteration.py, it will train the primary model and ensemble models in first iteration 
-> using primary model to run the MLIP simulation to produce new trajectory by run_mlpmd.py
-> 20 % of MLIP trajectory will be labeled with DFT forces as calibration set, 80 % of MILP trajectory as test set for uncertainty calculation.
-> high uncertain frames will be labeled with DFT (by CP2K) and augmented to primary model for training the augmented primary model in next iteration, if the force rmse doesn't be smaller than specific threshold.
