TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_only

for BASE_LR in 1e-4, 1e-3
do
    for MAX_LR in 1e-3, 1e-2
    do
        ngc batch run \
            --name "ml-model.$TASK_NAME" \
            --priority NORMAL \
            --preempt RUNONCE \
            --ace nv-us-west-2 \
            --instance dgx1v.32g.4.norm \
            --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
            --result /results \
            --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
            --datasetid 23145:/dataset \
            --team nvr-aialgo \
            --port 6006 --port 1234 --port 8888 \
            --commandline "bash -c '\
                sh /ngc_workspace/jiawei/set_wandb.sh; \
                cd /workspace; \
                git clone https://github.com/Robertboy18/neuraloperator.git; \
                cd /workspace/neuraloperator/scripts; \
                git checkout robert-test-incremental; \
                cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use = True;\
                mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use = True;\
                mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="exp" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use = True;\
            '"
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_incremental

for BASE_LR in 1e-4, 1e-3
do
    for MAX_LR in 1e-3, 1e-2
    do
        for THRESHOLD in 0.9 0.99 0.999
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.32g.4.norm \
                --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
                --result /results \
                --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
                --datasetid 23145:/dataset \
                --team nvr-aialgo \
                --port 6006 --port 1234 --port 8888 \
                --commandline "bash -c '\
                    sh /ngc_workspace/jiawei/set_wandb.sh; \
                    cd /workspace; \
                    git clone https://github.com/Robertboy18/neuraloperator.git; \
                    cd /workspace/neuraloperator/scripts; \
                    git checkout robert-test-incremental; \
                    cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="exp" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.incremental_grad.use=True --incremental.incremental_grad.grad_explained_ratio_threshold=$THRESHOLD;\
                '"
        done
    done
done

TASK_NAME=ifno_batch_script_test_fno_different_modes_resolution_and_loss_gap

for BASE_LR in 1e-4, 1e-3
do
    for MAX_LR in 1e-3, 1e-2
    do
        for EPS in 0.1 0.01 0.001
        do
            ngc batch run \
                --name "ml-model.$TASK_NAME" \
                --priority NORMAL \
                --preempt RUNONCE \
                --ace nv-us-west-2 \
                --instance dgx1v.32g.4.norm \
                --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
                --result /results \
                --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
                --datasetid 23145:/dataset \
                --team nvr-aialgo \
                --port 6006 --port 1234 --port 8888 \
                --commandline "bash -c '\
                    sh /ngc_workspace/jiawei/set_wandb.sh; \
                    cd /workspace; \
                    git clone https://github.com/Robertboy18/neuraloperator.git; \
                    cd /workspace/neuraloperator/scripts; \
                    git checkout robert-test-incremental; \
                    cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.loss_gap.use=True --incremental.loss_gap.eps=$EPS;\
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.loss_gap.use=True --incremental.loss_gap.eps=$EPS;\
                    mpiexec --allow-run-as-root -n 8 python train_navier_stokes.py --opt.mode="exp" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR --incremental.incremental_resolution.use=True --incremental.loss_gap.use=True --incremental.loss_gap.eps=$EPS;\
                '"
        done
    done
done