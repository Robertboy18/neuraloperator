#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --account=m4505

# set up for problem & define any environment variables here
module load conda
conda activate myenv
python train_navier_stokes-3d.py
#& CUDA_VISIBLE_DEVICES=3 python train_darcy.py --incremental.incremental_res=True