TASK_NAME=ifno_batch_script_re5000

for BASE_LR in 1e-5 1e-4
do
    for MAX_LR in 2e-4 1e-3
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
                    /usr/bin/python -m pip install --upgrade pip; \
                    sh /ngc_workspace/jiawei/set_wandb.sh; \
                    pip install configmypy zarr mpi4py; \
                    pip install -U tensorly; \
                    pip install -U tensorly-torch ; \
                    pip install torch_harmonics;\
                    cd /workspace; \    
                    git clone https://github.com/Robertboy18/neuraloperator.git; \
                    cd /workspace/neuraloperator; \
                    pip install -e . ; \
                    cp /ngc_workspace/jiawei/wandb_api_key.txt config/wandb_api_key.txt; \
                    cd /workspace/neuraloperator/scripts; \
                    git checkout robert-test-new-res; \
                    cd ..; \
                    cd ..; \
                    git clone https://github.com/Robertboy18/markov_neural_operator.git; \
                    cd /workspace/markov_neural_operator; \
                    cd scripts; \
                    git checkout robert-test; \
                    python NS_fno_baseline-trial.py --opt.mode="triangular2" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR;\
                '"     
        done    
    done
done

TASK_NAME=ifno_batch_script_re5000

for BASE_LR in 1e-5 1e-4
do
    for MAX_LR in 2e-4 1e-3
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
                    /usr/bin/python -m pip install --upgrade pip; \
                    sh /ngc_workspace/jiawei/set_wandb.sh; \
                    pip install configmypy zarr mpi4py; \
                    pip install -U tensorly; \
                    pip install -U tensorly-torch ; \
                    pip install torch_harmonics;\
                    cd /workspace; \    
                    git clone https://github.com/Robertboy18/neuraloperator.git; \
                    cd /workspace/neuraloperator; \
                    pip install -e . ; \
                    cp /ngc_workspace/jiawei/wandb_api_key.txt config/wandb_api_key.txt; \
                    cd /workspace/neuraloperator/scripts; \
                    git checkout robert-test-new-res; \
                    cd ..; \
                    cd ..; \
                    git clone https://github.com/Robertboy18/markov_neural_operator.git; \
                    cd /workspace/markov_neural_operator; \
                    cd scripts; \
                    git checkout robert-test; \
                    python NS_fno_baseline-trial.py --opt.mode="exp_range" --opt.base_lr=$BASE_LR --opt.max_lr=$MAX_LR;\
                '"     
        done    
    done
done