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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from scipy.fft import fft
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from neuralop.training import AdamW
from neuralop.layers.fourier_continuation import FCLegendre

class SHGTimeSeriesDataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = torch.tensor(input_series, dtype=torch.complex64)
        self.output_series = torch.tensor(output_series, dtype=torch.complex64)

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]

def embed_parameter(value, num_points, d):
    t = np.linspace(0, 2*np.pi, num_points)
    embeddings = []
    for i in range(d):
        embeddings.append(value * np.sin((i+1)*t))
        embeddings.append(value * np.cos((i+1)*t))
    return np.array(embeddings)

def load_and_preprocess_data(file_path, num=2048, samples=1000, use_fft=False, d=3, use_embeddings=False, use_truncation=False, values=None):
    # Load the CSV file
    # Check if file_path is a string (single file) or a list (multiple files)
    if isinstance(file_path, str):
        # If it's a string, check if it's a directory or a single file
        if file_path.endswith('.csv'):
            file_list = [file_path]

    elif isinstance(file_path, list):
        # If it's already a list of files, use it as is
        file_list = file_path
    else:
        raise ValueError("file_path must be a string (file path or directory) or a list of file paths")

    # Initialize an empty list to store DataFrames
    df_list = []

    # Read each CSV file and append to the list
    for file in file_list:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)

    # Extract features (first 3 columns)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values[:samples]

    # Function to convert string representation of complex numbers to complex values
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    if use_truncation:
        input_columns = [f'Input_{i}' for i in values]
        output_columns = [f'Output_{i}' for i in values]
    else:
    # Extract and convert input time series (Input_0 to Input_2047
        input_columns = [f'Input_{i}' for i in range(num)]
        output_columns = [f'Output_{i}' for i in range(num)]
    input_series = df[input_columns].map(to_complex).values[:samples]
    output_series = df[output_columns].map(to_complex).values[:samples]

    # Prepare input data with shape (num_samples, d*3 + 1, num)
    if use_embeddings:
        input_data = np.zeros((samples, d*3*2 + 1, num), dtype=np.complex128)
        # Embed and add the constant parameters
        for i in range(3):
            embeddings = embed_parameter(features[:, i], samples, d)
            input_data[:, i*d*2:(i+1)*d*2, :] = embeddings.T[:, :, np.newaxis]
    else:
        if use_truncation:
            input_data = np.zeros((samples, 4, len(values)), dtype=np.complex128)
            # Add the constant parameters
            for i in range(3):
                input_data[:, i, :] = np.tile(features[:, i], (len(values), 1)).T
        else:
            input_data = np.zeros((samples, 4, num), dtype=np.complex128)
            # Add the constant parameters
            for i in range(3):
                input_data[:, i, :] = np.tile(features[:, i], (num, 1)).T
        
    # Add the input series as the last channel
    input_data[:, -1, :] = input_series

    # Reshape output to (num_samples, 1, num)
    if use_truncation:
        output_data = output_series.reshape(-1, 1, len(values))
    else:
        output_data = output_series.reshape(-1, 1, num)

    if use_fft:
        # Apply FFT to the input series (last channel of input_data)
        input_data[:, -1, :] = fft(input_data[:, -1, :], axis=1)
        
        # Apply FFT to the output data
        output_data = fft(output_data, axis=2)

    return input_data, output_data

def create_dataloaders(input_data, output_series, batch_size=32, test_size=0.10, fc=True, use_fft=True):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_series, test_size=test_size, random_state=42
    )

    # Create Dataset objects
    train_dataset = SHGTimeSeriesDataset(X_train, y_train)
    test_dataset = SHGTimeSeriesDataset(X_test, y_test)

    
    if fc:
        number_additional_points=16
        n = 3
        d= number_additional_points
        fc = FCLegendre(n, d, dtype=torch.complex64)
        input_series1 = []
        output_series1 = []
        input_series2 = []
        output_series2 = []
        for samples in train_dataset:
            input_series, output_series = samples
            input_series = fc.extend_left_right(input_series)
            output_series = fc.extend_left_right(output_series)
            if use_fft:
                #print(input_series.shape, output_series.shape)
                input_series[3:, :] = torch.fft.fft(input_series[3:, :], axis=1)
                #print(input_series.shape, output_series.shape)
                output_series = torch.fft.fft(output_series, axis=1)
            input_series1.append(input_series)
            output_series1.append(output_series)
        # convert list to numpy
        input_series1 = np.array(input_series1)
        output_series1 = np.array(output_series1)
        train_dataset = SHGTimeSeriesDataset(input_series1, output_series1)
        
        for samples in test_dataset:
            input_series, output_series = samples
            input_series = fc.extend_left_right(input_series)
            output_series = fc.extend_left_right(output_series)
            if use_fft:
                input_series[3:, :] = torch.fft.fft(input_series[3:, :], axis=1)
                output_series = torch.fft.fft(output_series, axis=1)
            input_series2.append(input_series)
            output_series2.append(output_series)
        # convert list to numpy
        input_series2 = np.array(input_series2)
        output_series2 = np.array(output_series2)
        test_dataset = SHGTimeSeriesDataset(input_series2, output_series2)        
        
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            
    return train_loader, test_loader

# Usage
file_path = ["/raid/robert/em/SHG_output_final.csv", "/raid/robert/SHG_output_final_more_1000.csv"] #"/home/robert/repo/neuraloperator/examples/trial.csv"
input_data, output_series = load_and_preprocess_data(file_path, num=2048, samples=2000, use_fft=True, d=6, use_embeddings=False, use_truncation=False, values=[i for i in range(600, 1500)])
train_loader, test_loader = create_dataloaders(input_data, output_series, fc=False, use_fft=False)

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

model = FNO(n_modes=(256,), in_channels=4, out_channels=1, hidden_channels=512, projection_channels=256, n_layers=4, complex_spatial_data=True, domain_padding=None)  #FNO(n_modes=(1024,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=True)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

#model.load_state_dict(torch.load('/raid/robert/em/model.pt'))

# %%
#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=8e-4, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)


# %%
# Creating the losses
l4loss = LpLoss(d=1, p=2, reduce_dims=[0,1], reductions=['sum', 'mean'])
H1Loss1 = H1Loss(d=1, reduce_dims=[0,1], reductions=['sum', 'mean'])

train_loss = l4loss #H1Loss1 
eval_losses= {"H1": H1Loss1, "L2": l4loss}


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
epochs = 400
# Create the trainer
trainer = Trainer(model=model, n_epochs=epochs,
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
              eval_losses=eval_losses,
              use_fft=False)

torch.save(model.state_dict(), f'/raid/robert/em/model.pt')