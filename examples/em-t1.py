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

class SHGTimeSeriesDataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = torch.FloatTensor(input_series)
        self.output_series = torch.FloatTensor(output_series)

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]

def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract features (first 3 columns)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values

    # Function to convert string representation of complex numbers to complex values
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    # Extract and convert input time series (Input_0 to Input_2047)
    input_columns = [f'Input_{i}' for i in range(3)]  # 0 to 2047
    input_series = df[input_columns].applymap(to_complex).values

    # Prepare input data with shape (num_samples, 4, 2048)
    input_data = np.zeros((len(features), 4, 3), dtype=np.complex128)
    
    # Repeat feature values 2048 times for the first 3 channels
    for i in range(3):
        input_data[:, i, :] = np.tile(features[:, i], (3, 1)).T
    
    # Add the input series as the 4th channel
    input_data[:, 3, :] = input_series

    # Extract and convert output time series (Output_0 to Output_2047)
    output_columns = [f'Output_{i}' for i in range(3)]  # 0 to 2047
    output_series = df[output_columns].applymap(to_complex).values

    # Reshape output to (num_samples, 1, 2048)
    output_data = output_series.reshape(-1, 1, 3)
    return input_data, output_data


def create_dataloaders(input_data, output_series, batch_size=32, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_series, test_size=test_size, random_state=42
    )

    # Create Dataset objects
    train_dataset = SHGTimeSeriesDataset(X_train, y_train)
    test_dataset = SHGTimeSeriesDataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Usage
file_path = "/home/robert/repo/neuraloperator/examples/trial.csv"
input_data, output_series = load_and_preprocess_data(file_path)
train_loader, test_loader = create_dataloaders(input_data, output_series)

# Print some information about the loaded data
print(f"Total number of samples: {len(input_data)}")
print(f"Input data shape: {input_data.shape}")
print(f"Output series shape: {output_series.shape}")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# Example of accessing a batch
for batch_input_series, batch_output_series in train_loader:
    print("Batch input series shape:", batch_input_series.shape)
    print("Batch output series shape:", batch_output_series.shape)
    break  # Just print the first batch and exit the loop
# Usage


data_processor = None
device = 'cuda'

# %%
# We create a tensorized FNO model

model = FNO(n_modes=(32,), in_channels=4, out_channels=1, hidden_channels=128, n_layers=4, complex_spatial_data=True)
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
