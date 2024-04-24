"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO and Incremental Resolution
"""

# %%
#

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training.callbacks import IncrementalCallback
from neuralop.datasets import data_transforms
from neuralop import LpLoss, H1Loss
from neuralop.training import AdamW
import time


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

ranks = [20]
final_times = []
for i in ranks:
    model = FNO(
        max_n_modes=(20, 20),
        n_modes=(20, 20),
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        n_layers=4
    )

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
                        {'params': galore_params, 'rank': i , 'update_proj_gap': 1, 'scale': 0.25, 'proj_type': "std"}]
        optimizer = AdamW(param_groups, lr=8e-3, weight_decay=1e-4)
    else:
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


    # If one wants to use Incremental Resolution, one should use the IncrementalDataProcessor - When passed to the trainer, the trainer will automatically update the resolution
    # Incremental_resolution : bool, default is False
    #    if True, increase the resolution of the input incrementally
    #    uses the incremental_res_gap parameter
    #    uses the dataset_sublist parameter - a list of resolutions to use
    #    uses the dataset_indices parameter - a list of indices of the dataset to slice to regularize the input resolution
    #    uses the dataset_resolution parameter - the resolution of the input
    #    uses the epoch_gap parameter - the number of epochs to wait before increasing the resolution
    #    uses the verbose parameter - if True, print the resolution and the number of modes
    data_transform = data_transforms.IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        positional_encoding=None,
        device=device,
        dataset_sublist=[2, 1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=10,
        verbose=True,
    )

    data_transform = data_transform.to(device)
    # %%
    # Set up the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}
    #print("\n### N PARAMS ###\n", n_params)
    print("\n### OPTIMIZER rank ###\n", i, optimizer)
    #print("\n### SCHEDULER ###\n", scheduler)
    #print("\n### LOSSES ###")
    #print("\n### INCREMENTAL RESOLUTION + GRADIENT EXPLAINED ###")
    #print(f"\n * Train: {train_loss}")
    #print(f"\n * Test: {eval_losses}")
    sys.stdout.flush()

    # %%
    # Set up the IncrementalCallback
    # other options include setting incremental_loss_gap = True
    # If one wants to use incremental resolution set it to True
    # In this example we only update the modes and not the resolution
    # When using the incremental resolution one should keep in mind that the numnber of modes initially set should be strictly less than the resolution
    # Again these are the various paramaters for the various incremental settings
    # incremental_grad : bool, default is False
    #    if True, use the base incremental algorithm which is based on gradient variance
    #    uses the incremental_grad_eps parameter - set the threshold for gradient variance
    #    uses the incremental_buffer paramater - sets the number of buffer modes to calculate the gradient variance
    #    uses the incremental_max_iter parameter - sets the initial number of iterations
    #    uses the incremental_grad_max_iter parameter - sets the maximum number of iterations to accumulate the gradients
    # incremental_loss_gap : bool, default is False
    #    if True, use the incremental algorithm based on loss gap
    #    uses the incremental_loss_eps parameter
    # One can use multiple Callbacks as well
    callbacks = [
        IncrementalCallback(
            incremental_loss_gap=True,
            incremental_grad=False,
            incremental_grad_eps=0.9999,
            incremental_buffer=5,
            incremental_max_iter=1,
            incremental_grad_max_iter=2,
        )
    ]

    # Finally pass all of these to the Trainer
    trainer = Trainer(
        model=model,
        n_epochs=20,
        data_processor=data_transform,
        callbacks=callbacks,
        device=device,
        verbose=True,
    )
    start = time.time()

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

    final_times.append(time.time() - start)

for i in range(len(ranks)):
    print("Rank ", ranks[i], final_times[i])
# %%
# Plot the prediction, and compare with the ground-truth
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
#
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs