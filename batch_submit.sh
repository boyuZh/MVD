#!/bin/bash

# check_exist=1 means check if the trained model exists, 0 means not check. 
# If the trained model exists, the job will not be submitted.
check_exist=0
do_train=1

bash slurm_bin.sh $check_exist $do_train
bash slurm_mc_focal_adj.sh $check_exist $do_train
bash slurm_mc_focal.sh $check_exist $do_train
bash slurm_mc_freeze.sh $check_exist $do_train
bash slurm_mc_inc.sh $check_exist $do_train
bash slurm_mc.sh $check_exist $do_train
bash slurm_mcadj.sh $check_exist $do_train
bash slurm_single.sh $check_exist $do_train
