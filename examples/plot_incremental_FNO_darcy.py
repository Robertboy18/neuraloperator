"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO and Incremental Resolution
"""

# %%
#
import wandb
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training.callbacks import IncrementalCallback
from neuralop.datasets import data_transforms
from neuralop import LpLoss, H1Loss
from neuralop.training import AdamW
import time
from neuralop.utils import get_wandb_api_key, count_model_params


# %%
# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
# %%
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up the incremental FNO model
# We start with 2 modes in each dimension
# We choose to update the modes by the incremental gradient explained algorithm

starting_modes = (10, 10)
incremental = False

scales = [0]
ranks = [1]
update_proj_gap = [1]
lr = [1e-1, 1e-2, 1e-3]
for m in scales:
    for k in lr:
        for j in update_proj_gap:
            for i in ranks:
                model = FNO(
                    max_n_modes=(20, 20),
                    n_modes=(20, 20),
                    hidden_channels=64,
                    in_channels=1,
                    out_channels=1,
                    n_layers=4
                )

                wandb.login(key=get_wandb_api_key())
                if i != 0:
                    wandb_name = f"fno_darcy_rank_{i}_tensor"
                else:
                    wandb_name = f"fno_darcy_baseline_tensor"
                wandb_args =  dict(
                    name=wandb_name,
                    group='',
                    project='darcy',
                    entity='research-pino_ifno',
                )
                callbacks = [
                    IncrementalCallback(
                        incremental_loss_gap=True,
                        incremental_grad=False,
                        incremental_grad_eps=0.9999,
                        incremental_buffer=5,
                        incremental_max_iter=1,
                        incremental_grad_max_iter=2,
                    ), BasicLoggerCallback(wandb_args)
                ]     
                model = model.to(device)
                n_params = count_model_params(model)
                if i != 0:
                    galore_params = []
                    galore_params.extend(list(model.fno_blocks.convs.parameters()))
                    print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
                    galore_params.pop(0)
                    id_galore_params = [id(p) for p in galore_params]
                    # make parameters without "rank" to another group
                    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
                    # then call galore_adamw
                    param_groups = [{'params': regular_params}, 
                                    {'params': galore_params, 'rank': i , 'update_proj_gap': j, 'scale': m, 'proj_type': "std", 'dim': 5}]
                    param_groups1 = [{'rank': i , 'update_proj_gap': j, 'scale': 0.25, 'proj_type': "std", 'dim':5}]
                    optimizer = AdamW(param_groups, lr=m*10*k)
                    wandb.log(param_groups1[0])
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=k)
                    param_groups = [{'rank': 'baseline'}]
                    wandb.log(param_groups[0])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
                data_transform = data_transforms.IncrementalDataProcessor(
                    in_normalizer=None,
                    out_normalizer=None,
                    positional_encoding=None,
                    device=device,
                    dataset_sublist=[1],
                    dataset_resolution=16,
                    dataset_indices=[2, 3],
                    epoch_gap=50,
                    verbose=True,
                )

                data_transform = data_transform.to(device)
                # %%
                # Set up the losses
                l2loss = LpLoss(d=2, p=2)
                h1loss = H1Loss(d=2)
                train_loss = h1loss
                eval_losses = {"h1": h1loss, "l2": l2loss}
                print("\n### OPTIMIZER rank ###\n", i, optimizer)
                sys.stdout.flush()

                epochs = 100
                wandb.log({"n_params": n_params, "epochs": epochs})
                # Finally pass all of these to the Trainer
                trainer = Trainer(
                    model=model,
                    n_epochs=epochs,
                    data_processor=data_transform,
                    callbacks=callbacks,
                    device=device,
                    verbose=True,
                )
                start = time.time()

                wandb.watch(model)

                # %%
                # Train the model
                trainer.train(
                    train_loader,
                    test_loaders,
                    optimizer,
                    scheduler,
                    regularizer=False,
                    training_loss=train_loss,
                    eval_losses=eval_losses,
                )

                wandb.finish()
