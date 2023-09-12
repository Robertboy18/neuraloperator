TASK_NAME=ifno_batch_script_test_fno_turbulence-allwary

for BASE_LR in 1e-3
do
    for MODES in 8
    do
        for EPOCH in 1000
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
                    git checkout robert-turbulence; \
                    cp -r /ngc_workspace/jiawei/projects/ifno/data /workspace/fly-incremental/data; \
                    mpiexec --allow-run-as-root -n 8 python train_2d.py --opt.scheduler="StepLR" --opt.learning_rate=$BASE_LR --checkpoint.name="checkpoints40" --opt.n_epochs=$EPOCH;\
                '"     
        done    
    done
done