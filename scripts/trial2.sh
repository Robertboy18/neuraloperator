for proj in 10 50
do
    for scale in 0.01 0.1 0.25 0.5 0.75 1.0
    do
        CUDA_VISIBLE_DEVICES=0 python train_navier_stokes-time_galore.py --rank=$scale --scale=0.25 --proj_gap=$proj & CUDA_VISIBLE_DEVICES=1 python train_navier_stokes-time_galore.py --rank=$scale --scale=0.5 --proj_gap=$proj
    done
done