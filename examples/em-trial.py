# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

class TimeDomainDataset(Dataset):
    def __init__(self, features, targets):
        # make the channel always 1 and the modes as the features and batch size to be the first
        self.features = torch.FloatTensor(features).unsqueeze(1)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract features and target
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)', 'Time (ps)']].values
    targets = df['Pulse Intensity (W)'].values

    return features, targets

def create_dataloaders(features, targets, batch_size=32, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)

    # Create Dataset objects
    train_dataset = TimeDomainDataset(X_train, y_train)
    test_dataset = TimeDomainDataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

file_path = "/raid/robert/em/SHG_example_data/SHG_1.csv"
features, targets = load_and_preprocess_data(file_path)
train_loader, test_loader = create_dataloaders(features, targets)

# Example of accessing data from the DataLoader
for batch_features, batch_targets in train_loader:
    print("Batch features shape:", batch_features.shape)
    print("Batch targets shape:", batch_targets.shape)
    break  # Just print the first batch and exit the loop

print("Data loading and preprocessing completed.")

data_processor = None
device = 'cuda'


# %%
# We create a tensorized FNO model

model = FNO(n_modes=(16,), in_channels=1, out_channels=1, hidden_channels=12, projection_channels=12, n_layers=1)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=10,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
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
