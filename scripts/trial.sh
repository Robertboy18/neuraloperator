for scale in 0.01
do
    CUDA_VISIBLE_DEVICES=0 python train_darcy_galore.py --rank=0.75 --scale=0.25 & CUDA_VISIBLE_DEVICES=1 python train_darcy_galore.py --rank=0.25 --scale=0.5
done