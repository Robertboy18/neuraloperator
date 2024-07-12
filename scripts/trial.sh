for s in 0
do
    CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time.py & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_loss_gap=True
    CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time.py --incremental.incremental_loss_gap=True --incremental.incremental_res=True & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_res=True
    CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time.py --incremental.incremental_grad=True --incremental.incremental_res=True & CUDA_VISIBLE_DEVICES=2 python train_navier_stokes-time.py --incremental.incremental_grad=True
done