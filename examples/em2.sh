for rank in 8 16 32 64 128
do
    CUDA_VISIBLE_DEVICES=1 python em-t1-fc-galore.py --rank=$rank --update_proj_gap=50 --dim=1 --matrix=True     
    CUDA_VISIBLE_DEVICES=2 python em-t1-fc-galore.py --rank=$rank --update_proj_gap=50 --dim=2 --matrix=True
    #CUDA_VISIBLE_DEVICES=3 python em-t1-fc-galore.py --rank=$rank --update_proj_gap=50 --dim=3 --matrix=True
done
#for rank in 0.01 0.1 0.25 0.5 0.75 1.00
#do
#CUDA_VISIBLE_DEVICES=2 python em-t1-fc-galore.py --rank=$rank
#done