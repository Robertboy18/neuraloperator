#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH -n 4
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --account=m4505

# set up for problem & define any environment variables here
module load conda
conda activate myenv
for s in 0 1 2;
do
CUDA_VISIBLE_DEVICES=0 python train_navier_stokes-time.py --seed=$s --mode=60 --ns2dtime=True & CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time.py --mode=90 --seed=$s --ns2dtime=True
#& CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-3d.py --mode=30 --seed=2 --nstime=True
done
#& CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-3d.py --incremental.incremental_res=True --incremental.incremental_loss_gap=True --seed=$s & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-3d.py --incremental.incremental_grad=True --incremental.incremental_res=True --seed=$s & CUDA_VISIBLE_DEVICES=3 python train_navier_stokes-3d.py --incremental.incremental_loss_gap=True --seed=$s