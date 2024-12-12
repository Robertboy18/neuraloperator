CUDA_VISIBLE_DEVICES=0 python em-t1-fc.py
for rank in 0.01 0.1 0.25 0.5 0.75 1.00
do
    CUDA_VISIBLE_DEVICES=0 python em-t1-fc-galore.py --rank=$rank
done
#python em-t1-fc-galore.py --rank=0.25 --scale=0.25
#python em-t1-fc-galore.py --rank=0.25 --scale=0.50
#python em-t1-fc-galore.py --rank=0.25 --scale=0.75
#python em-t1-fc-galore.py --rank=0.25 --scale=1.00
