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
for s in 0 1 2;
do
    python train_burgers.py --incremental.incremental_loss_gap=True --seed=$s
    python train_burgers.py --incremental.incremental_loss_gap=True --incremental.incremental_res=True --seed=$s
    python train_burgers.py --incremental.incremental_res=True --seed=$s
    python train_burgers.py --incremental.incremental_grad=True --seed=$s
    python train_burgers.py --incremental.incremental_grad=True --incremental.incremental_res=True --seed=$s
    python train_burgers.py --seed=$s
    python train_burgers.py --seed=$s --mode=10
    python train_burgers.py --seed=$s --mode=30
    python train_burgers.py --seed=$s --mode=60
done

