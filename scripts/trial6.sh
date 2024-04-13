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
for s in 0;
do
    CUDA_VISIBLE_DEVICES=0 python train_navier_stokes-high.py --seed=0 --mode=60 & CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-high.py --mode=60 --seed=2 & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-high.py --mode=30 --seed=2 & CUDA_VISIBLE_DEVICES=3 python train_navier_stokes-high.py --mode=30 --seed=0
done