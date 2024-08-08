# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training.callbacks import IncrementalCallback
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

class TimeDomainDataset(Dataset):
    def __init__(self, features, targets):
        # make the channel always 1 and the modes as the features and batch size to be the first
        self.features = torch.FloatTensor(features).unsqueeze(1)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_and_preprocess_data(folder_path):
    all_features = []
    all_targets = []

    for i in range(1, 980):  # Files from SHG_1.csv to SHG_979.csv
        file_name = f"SHG_{i}.csv"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Extract features and target
            features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)', 'Time (ps)']].values
            targets = df['Pulse Intensity (W)'].values

            all_features.append(features)
            all_targets.append(targets)
        else:
            print(f"File not found: {file_path}")

    # Concatenate all features and targets
    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)

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

# Usage
folder_path = "/raid/robert/em/SHG_example_data"
features, targets = load_and_preprocess_data(folder_path)
train_loader, test_loader = create_dataloaders(features, targets)

# Print some information about the loaded data
print(f"Total number of samples: {len(features)}")
print(f"Feature shape: {features.shape}")
print(f"Target shape: {targets.shape}")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

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

data_processor = None
device = 'cuda'


# %%
# We create a tensorized FNO model

class FNOMLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers):
        super(FNOMLP, self).__init__()
        self.main_fno = FNO(n_modes=(16,), max_n_modes=(16, ), in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, n_layers=n_layers)

        #self.mlp = torch.nn.Linear(4, 1)
        
        # instead try a pooling layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        #print("before fno", x.shape)
        x = self.main_fno(x)
        #print("after fno", x.shape)
        x = self.mlp(x)
        return x
    
model = FNOMLP(in_channels=3, out_channels=1, hidden_channels=128, n_layers=4)
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
l2loss = LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)

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

callbacks = [BasicLoggerCallback()]    

# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=100,
                  device=device,
                  callbacks=callbacks,
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
