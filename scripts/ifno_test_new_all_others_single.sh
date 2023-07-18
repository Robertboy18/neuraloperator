############# Mixed grad #####################
TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_incremental1

for BASE_LR in 1e-5
do
    for MAX_LR in 1e-4
    do
        for THRESHOLD in 0.99 0.999 0.9999
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.32g.2.norm \
                --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
                --result /results \
                --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
                --datasetid 23145:/dataset \
                --datasetid 110516:/high_res_ns_dataset \
                --team nvr-aialgo \
                --port 6006 --port 1234 --port 8888 \
                --commandline "bash -c '\
                    sh /ngc_workspace/jiawei/set_wandb.sh; \
                    pip install configmypy zarr mpi4py; \
                    pip install -U tensorly; \
                    pip install -U tensorly-torch ; \
                    cd /workspace; \
                    git clone https://github.com/Robertboy18/neuraloperator.git; \
                    cd /workspace/neuraloperator; \
                    pip install -e . ; \
                    cp /ngc_workspace/jiawei/wandb_api_key.txt config/wandb_api_key.txt; \
                    cd /workspace/neuraloperator/scripts; \
                    git checkout robert-test-incremental; \
                    cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                    CUDA_VISIBLE_DEVICES=0,1 python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD --incremental.dataset.NavierStokes=1 & python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=0.001 --incremental.incremental_resolution.use=True --incremental.dataset.NavierStokes=8;\
                '"
        done
    done
done
