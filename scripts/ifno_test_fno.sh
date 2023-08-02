TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap_test_modes1

for BASE_LR in 1e-4
do
    for MAX_LR in 5e-3
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.32g.1.norm \
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
                python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --checkpoint.name="checkpoints15";\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap_test_modes2

for BASE_LR in 1e-4
do
    for MAX_LR in 5e-3
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.32g.1.norm \
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
                python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --checkpoint.name="checkpoints16";\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap_test_modes3

for BASE_LR in 1e-4
do
    for MAX_LR in 5e-3
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.32g.1.norm \
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
                python train_navier_stokes.py --opt.mode="exp_range" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --checkpoint.name="checkpoints17";\
            '"
    done
done