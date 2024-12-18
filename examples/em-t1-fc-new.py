# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
import torch
import matplotlib.pyplot as plt
import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from neuralop import LpLoss, H1Loss
from scipy.fft import fft
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from neuralop.training import AdamW
from neuralop.layers.fourier_continuation import FCLegendre

from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop import H1Loss, LpLoss
from neuralop.utils import count_model_params

from datetime import datetime
from timeit import default_timer

import argparse

import wandb

class emSHG_Dataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = input_series
        self.output_series = output_series

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]


def create_dataloaders(file_path, samples=1000, batch_size=128, test_size=0.1, use_truncation=False, values=None, use_fft=False, use_embeddings=None, embed_freqs=7):

    data = np.load(file_path)
    
    poling_region_length = torch.from_numpy(data['poling_region_length'])[:samples].to(torch.complex64)
    poling_period_mismatch = torch.from_numpy(data['poling_period_mismatch'])[:samples].to(torch.complex64)
    pump_energy = torch.from_numpy(data['pump_energy'])[:samples].to(torch.complex64)
    input_field = torch.from_numpy(data['input_field'])[:samples].to(torch.complex64)
    output_field = torch.from_numpy(data['output_field'])[:samples].to(torch.complex64)
    
    if use_truncation:
        num = len(values)
    else:
        num = output_field.shape[-1]
        values = [i for i in range(0, num)]
    

    if use_embeddings:  
        input_data = torch.zeros(samples, (2*embed_freqs+1)*3+1, num, dtype=torch.complex64)
    else:
        input_data = torch.zeros((samples, 4, num), dtype=torch.complex64)
    
    input_data[:, 0, :] = poling_region_length.repeat(num,1).permute(1,0)
    input_data[:, 1, :] = poling_period_mismatch.repeat(num,1).permute(1,0)
    input_data[:, 2, :] = pump_energy.repeat(num,1).permute(1,0)
    
    if use_embeddings:
        t = torch.linspace(0, 2*torch.pi, num, dtype=torch.complex64)
        for i in range(0,embed_freqs):
            input_data[:, 3+6*i, :] = poling_region_length.repeat(num,1).permute(1,0)  * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 4+6*i, :] = poling_region_length.repeat(num,1).permute(1,0)  * np.sin((i+1)*t).repeat(samples,1)
            input_data[:, 5+6*i, :] = poling_period_mismatch.repeat(num,1).permute(1,0) * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 6+6*i, :] = poling_period_mismatch.repeat(num,1).permute(1,0) * np.sin((i+1)*t).repeat(samples,1)
            input_data[:, 7+6*i, :] = pump_energy.repeat(num,1).permute(1,0) * np.cos((i+1)*t).repeat(samples,1)
            input_data[:, 8+6*i, :] = pump_energy.repeat(num,1).permute(1,0) * np.sin((i+1)*t).repeat(samples,1)
    
    input_data[:, -1, :] = input_field[:,values]
    
    output_data = output_field[:, values].unsqueeze(1)
    
    if use_fft:
        input_data[:, -1, :] = torch.fft.fft(input_data[:, -1, :], dim=-1)
        output_data = torch.fft.fft(output_data, dim=-1)
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=test_size, random_state=42)

    # Create Dataset objects
    train_dataset = emSHG_Dataset(X_train, y_train)
    test_dataset = emSHG_Dataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



print()
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
device = 'cuda'

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs=1, required=False, help='the seed for the random number generator')
    args = parser.parse_args()
    seed = args.seed[0]
except:
    # Print a specified message
    print("Seed not specified. Using seed=42 by default.\n")
    seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


print_every_n_epochs = 1

use_wandb = False
if use_wandb: 
    wandb.login()

save_model = False
save_every_n_epochs = 10
model_path = "/home/vduruiss/Val/EM/ML/models/FNO_seed_" + str(seed) + ".pt"


# %% ###############################################################
# Load Data

print("Loading data...")
file_path = "/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/emSHG_data_100k.npz"

num_samples = 10000
test_size = 0.1
batch_size = 128

use_truncation = True
truncation_values = [i for i in range(700, 1300)]

use_fft = False
use_embeddings = False
embed_freqs = 14

train_loader, test_loader = create_dataloaders(file_path, 
                                               samples=num_samples, batch_size=batch_size, test_size=test_size, 
                                               use_truncation=use_truncation, values=truncation_values,
                                               use_fft=use_fft, 
                                               use_embeddings=use_embeddings, embed_freqs=embed_freqs)

print("   train_loader:", len(train_loader), "test_loader:", len(test_loader))

# Example data
input_data, output_series = next(iter(train_loader))
print("input_data batch shape:", input_data.shape, "output_series batch shape:", output_series.shape)

# %% ###############################################################
# Create the model

print("Creating the model...")

# Print some information about the loaded data
print(f"Total number of samples: {len(input_data)}")
print(f"Input data shape: {input_data.shape}")
print(f"Output series shape: {output_series.shape}")

# Example of accessing a batch
for batch_input_series, batch_output_series in train_loader:
    print("Batch input series shape:", batch_input_series.shape)
    print("Batch output series shape:", batch_output_series.shape)
    print("Dtype", batch_input_series.dtype, batch_output_series.dtype)
    break  # Just print the first batch and exit the loop
# Usage


data_processor = None
device = 'cuda'

# %%
# We create a tensorized FNO model

model = FNO(n_modes=(64,), in_channels=input_data.shape[1], out_channels=1, hidden_channels=256, projection_channels=64, n_layers=4, complex_data=True, domain_padding=None)  #FNO(n_modes=(1024,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=True)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

#model.load_state_dict(torch.load('/raid/robert/em/model.pt'))

# %%
#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=1e-3, 
                                weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
#scheduler_factor=0.8
#patience=8
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = scheduler_factor, patience=patience)


# %%
# Creating the losses
l4loss = LpLoss(d=1, p=2, reduction='mean')
H1Loss1 = H1Loss(d=1, reduction='mean')

train_loss = l4loss 
eval_losses= {"H1": H1Loss1, "L2": l4loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %% 
epochs = 10010
# Create the trainer
trainer = Trainer(model=model, n_epochs=epochs,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loader,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

torch.save(model.state_dict(), f'/pscratch/sd/r/rgeorge/model.pt')