"""
Training an incremental fourier neural operator on Darcy-Flow 
========================================
"""
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



# Loading the Darcy flow dataset
train_loader, test_loaders, output_encoder = load_darcy_flow_small(
    n_train=100,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[32, 32],
)


# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the incremental FNO model
# We start with 2 modes in each dimension
# We choose to update the modes by the incremental gradient explained algorithm
starting_modes = (2, 2)
incremental = True


# set up model
model = FNO(
    max_n_modes=(16, 16),
    n_modes=starting_modes,
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
)
model = model.to(device)
n_params = count_model_params(model)

# Set up the optimizer and scheduler
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

# Set up the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
print("\n### N PARAMS ###\n", n_params)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print("\n### INCREMENTAL RESOLUTION + GRADIENT EXPLAINED ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

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
        incremental_loss_gap=False,
        incremental_grad=True,
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