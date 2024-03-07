#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH -n 4
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --account=m4505

# set up for problem & define any environment variables here
module load conda
conda activate myenv
CUDA_VISIBLE_DEVICES=0 python train_navier_stokes-time.py --incremental.incremental_loss_gap=True --incremental.incremental_res=True & CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time.py --incremental.incremental_grad=True --incremental.incremental_res=True & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_res=True #& CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_loss_gap=True & CUDA_VISIBLE_DEVICES=3 python train_navier_stokes-time.py --incremental.incremental_grad=True
#& CUDA_VISIBLE_DEVICES=2 python train_navier_stokes.py --incremental.incremental_res=True
 #& CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_res=True