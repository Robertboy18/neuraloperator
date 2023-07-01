############ Base Line ###############

TASK_NAME=ifno_batch_script_test_fno_baseline_ablation

for LR in 1e-3
do
    for SCALE in 8
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True --incremental.dataset.NavierStokes=$SCALE;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_baseline_ablation

for LR in 1e-3
do
    for SCALE in 4
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True --incremental.dataset.NavierStokes=$SCALE;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_baseline_ablation

for LR in 1e-3
do
    for SCALE in 2
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True --incremental.dataset.NavierStokes=$SCALE;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno

for BF in 0
do
    for SCALE in 0
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py ;\
            '"
    done
done

############# Mixed grad #####################
TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_incremental

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for THRESHOLD in 0.9 0.99 0.999 0.9999
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                '"
        done
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_incremental

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for THRESHOLD in 0.9 0.99 0.999 0.9999
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                '"
        done
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_incremental

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for THRESHOLD in 0.9 0.99 0.999 0.9999
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="exp_range" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                '"
        done
    done
done

######### Mixed loss gap#####################

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for EPS in 0.1 0.01 0.001
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_loss_gap.use=True --incremental.incremental_loss_gap.eps=$EPS;\
                '"
        done
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for EPS in 0.1 0.01 0.001
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_loss_gap.use=True --incremental.incremental_loss_gap.eps=$EPS;\
                '"
        done
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for EPS in 0.1 0.01 0.001
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="exp_range" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_loss_gap.use=True --incremental.incremental_loss_gap.eps=$EPS;\
                '"
        done
    done
done

########### Singular resolution ############
TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_only

for BASE_LR in 1e-4 1e-3
do
    for MAX_LR in 1e-3 1e-2
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True;\
            '"
    done
done

for BASE_LR in 1e-4 1e-3
do
    for MAX_LR in 1e-3 1e-2
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True;\
            '"
    done
done

for BASE_LR in 1e-3
do
    for MAX_LR in 1e-2
    do
        for GAMMA in 0.5 0.9 1
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.16g.1.norm \
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
                    python train_navier_stokes.py --opt.mode="exp_range" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --opt.gamma=$GAMMA --incremental.incremental_resolution.use=True;\
                '"
        done
    done
done

############ Combined step lr ###############
TASK_NAME=ifno_batch_script_test_fno_incremental_res

for LR in 1e-3
do
    for THRESHOLD in 0.9 0.99 0.999 0.9999
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
            '"
    done
done

## RUNS
TASK_NAME=ifno_batch_script_test_fno_loss_gap
for LR in 1e-3
do
    for EPS in 0.1 0.01 0.001
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True --incremental.incremental_loss_gap.use=True --incremental.incremental_loss_gap.eps=$EPS;\
            '"
    done
done
## RUNS

########## Standard STEPLR #############

TASK_NAME=ifno_batch_script_test_fno_baseline

for LR in 1e-3
do
    for SCALE in 0
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR ;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_loss_gap
for LR in 1e-3
do
    for EPS in 0.1 0.01 0.001
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_loss_gap.use=True --incremental.incremental_loss_gap.eps=$EPS;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_incremental_resolution

for LR in 1e-3
do
    for SCALE in 0
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_resolution.use=True;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_incremental

for LR in 1e-3
do
    for THRESHOLD in 0.9 0.99 0.999 0.9999
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.16g.1.norm \
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
                python train_navier_stokes.py --opt.scheduler="StepLR" --opt.learning_rate=$LR --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
            '"
    done
done