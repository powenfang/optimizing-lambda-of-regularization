#!/bin/bash

#SBATCH --job-name=trial_job
#SBATCH --account=csci_ga_2565_0001
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

bash /scratch/SP21ML/run-pytorch.bash python -u cnn_test.py