#!/usr/bin/env bash

ngc batch run \
  --name "ml-model.fly-incremental-debug" \
  --preempt RUNONCE \
  --min-timeslice 0s \
  --total-runtime 160h \
  --ace nv-us-west-2 \
  --instance dgx1v.32g.8.norm \
  --image nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest \
  --result /results \
  --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace \
  --datasetid 23145:/dataset \
  --datasetid 110516:/high_res_ns_dataset \
  --team nvr-aialgo \
  --port 6006 --port 1234 --port 8888 \
  --commandline "sh /ngc_workspace/jiawei/set_wandb.sh; \
    sh /ngc_workspace/jiawei/set_git.sh; \
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
    bash -c 'curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.9.1 ; \
    cp -r /ngc_workspace/jiawei/stored_ngc_info/.ssh /root/.ssh; \
    git config --global --add safe.directory /ngc_workspace/jiawei/projects/fly-incremental; \
    mkdir ~/.config/code-server/; \
    cp /ngc_workspace/jiawei/tools/code_server/config.yaml ~/.config/code-server/config.yaml; \
    code-server --install-extension ms-toolsai.jupyter; \
    code-server --install-extension esbenp.prettier-vscode; \
    code-server --install-extension eamodio.gitlens; \
    code-server --install-extension ms-python.python; \
    code-server --install-extension /ngc_workspace/jiawei/tools/code_server/vsix_packages/copilot.vsix; \
    code-server& \
    sleep 160h'"


# # Update Log: upgrade to nvidia/pytorch:22.01-py3 from 21.04


# # Instance type: dgx1v.16g.8.norm or dgxa100.20g.1.norm.mig.3(not working)(a100 nodes are still unavailable)
# # Image type: nvidia/pytorch:21.04-py3(previous imagenet run) or nvcr.io/nvidian/nvr-aialgo/fly-incremental:zoo_latest

# working example
# ngc batch run --name "ml-model.fly-incremental-debug" --priority NORMAL --preempt RUNONCE --total-runtime 576000s --ace nv-us-west-2 --instance dgx1v.16g.8.norm --commandline "sh /ngc_workspace/jiawei/set_wandb.sh; sh /ngc_workspace/jiawei/set_git.sh; bash -c 'curl -fsSL https://code-server.dev/install.sh | sh; mkdir ~/.config/code-server/; cp /ngc_workspace/jiawei/tools/code_server/config.yaml ~/.config/code-server/config.yaml; code-server --install-extension ms-toolsai.jupyter; code-server --install-extension esbenp.prettier-vscode; code-server --install-extension eamodio.gitlens; code-server --install-extension ms-python.python; code-server --install-extension /ngc_workspace/jiawei/tools/code_server/vsix_packages/copilot.vsix; code-server& sleep 160h'" --result /results --image "nvidian/nvr-aialgo/fly-incremental:zoo_latest" --org nvidian --team nvr-aialgo --datasetid 23145:/dataset --workspace 6Ubcqvn_Rn6uKFJw4ijJdw:/ngc_workspace:RW --port 1234 --port 6006 --port 8888 --order 50