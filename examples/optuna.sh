#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --constraint=gpu
#SBATCH -n 4
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --account=m4505
#SBATCH -C gpu&hbm80g

# set up for problem & define any environment variables here
conda activate /pscratch/sd/r/rgeorge/env/myenv
CUDA_VISIBLE_DEVICES=0 python Optuna_Training.py & CUDA_VISIBLE_DEVICES=1 python Optuna_Training1.py & CUDA_VISIBLE_DEVICES=2 python Optuna_Training2.py & CUDA_VISIBLE_DEVICES=3 python Optuna_Training3.py 