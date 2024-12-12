import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.fft import fft
from neuralop.training import AdamW
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

class SHGTimeSeriesDataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = torch.tensor(input_series, dtype=torch.complex64)
        self.output_series = torch.tensor(output_series, dtype=torch.complex64)

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]

def load_and_preprocess_data(file_path, num=2048, samples=1000, use_fft=False, d=3, use_embeddings=False, use_truncation=False, values=None, l=1):
    if use_truncation:
        num = len(values)
    
    if isinstance(file_path, str):
        if file_path.endswith('.csv'):
            file_list = [file_path]
    elif isinstance(file_path, list):
        file_list = file_path
    else:
        raise ValueError("file_path must be a string (file path or directory) or a list of file paths")

    df_list = []
    for file in file_list:
        df = pd.read_csv(file)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Function to convert string representation of complex numbers to complex values
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', '')).real
    
    # Extract features (first 3 columns)
    if l == 1:
        features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].map(to_complex).values[:samples]
    else:
        features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values[:samples]
    
    if use_truncation:
        input_columns = [f'Input_{i}' for i in values]
        output_columns = [f'Output_{i}' for i in values]
    else:          
        input_columns = [f'Input_{i}' for i in range(num)]
        output_columns = [f'Output_{i}' for i in range(num)]
    
    input_series = df[input_columns].map(to_complex).values[:samples]
    output_series = df[output_columns].map(to_complex).values[:samples]

    input_data = np.zeros((samples, 4, num), dtype=np.complex128)
    for i in range(3):
        input_data[:, i, :] = np.tile(features[:, i], (num, 1)).T
    input_data[:, -1, :] = input_series

    output_data = output_series.reshape(-1, 1, num)

    if use_fft:
        input_data[:, -1, :] = fft(input_data[:, -1, :], axis=1)
        output_data = fft(output_data, axis=2)

    return input_data, output_data

def create_dataloaders(input_data, output_series, batch_size=32, test_size=0.20):
    print(input_data.shape, output_series.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_series, test_size=test_size, random_state=42
    )

    train_dataset = SHGTimeSeriesDataset(X_train, y_train)
    test_dataset = SHGTimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    return train_loader, test_loader

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        real = x.real
        imag = x.imag
        return torch.complex(
            self.conv_real(real) - self.conv_imag(imag),
            self.conv_real(imag) + self.conv_imag(real)
        )

class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ComplexConv1d(in_channels, hidden_channels, kernel_size=3, padding=1))
        for _ in range(num_layers - 2):
            self.layers.append(ComplexConv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        self.layers.append(ComplexConv1d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.relu = ComplexReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

def train_model(model, train_loader, test_loader, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.9)  # Reduce LR by 10% every 25 epochs

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_input, batch_output in train_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs.abs(), batch_output.abs())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_input, batch_output in test_loader:
                batch_input, batch_output = batch_input.to(device), batch_output.to(device)
                outputs = model(batch_input)
                loss = criterion(outputs.abs(), batch_output.abs())
                test_loss += loss.item()

        scheduler.step()  # Step the scheduler

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Test Loss: {test_loss/len(test_loader):.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


class RelativeL2Loss(nn.Module):
    def __init__(self):
        super(RelativeL2Loss, self).__init__()
    
    def forward(self, outputs, targets):
        return torch.norm(outputs - targets, p=2) / torch.norm(targets, p=2)

def train_model1(model, train_loader, test_loader, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = RelativeL2Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR by 10% every 25 epochs

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_input, batch_output in train_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        max_pointwise_error = 0
        with torch.no_grad():
            for batch_input, batch_output in test_loader:
                batch_input, batch_output = batch_input.to(device), batch_output.to(device)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_output)
                test_loss += loss.item()
                max_pointwise_error = max(max_pointwise_error, torch.max(torch.abs(outputs - batch_output)).item())

        scheduler.step()  # Step the scheduler

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss (Rel L2): {train_loss/len(train_loader):.4f}, "
              f"Test Loss (Rel L2): {test_loss/len(test_loader):.4f}, "
              f"Max Pointwise Error: {max_pointwise_error:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        torch.save(model.state_dict(), f'/pscratch/sd/r/rgeorge/model_base.pt')
    return model

def visualize_predictions(model, test_loader, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        for batch_input, batch_output in test_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            predictions = model(batch_input)
            print(batch_input.shape)
            for i in range(num_samples):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f"Sample {i+1}")
                
                # Real part
                axs[0, 0].plot(batch_output[i, 0].real.cpu(), label='Ground Truth')
                axs[0, 0].plot(predictions[i, 0].real.cpu(), label='Prediction')
                axs[0, 0].set_title("Real Part")
                axs[0, 0].legend()
                
                # Imaginary part
                axs[0, 1].plot(batch_output[i, 0].imag.cpu(), label='Ground Truth')
                axs[0, 1].plot(predictions[i, 0].imag.cpu(), label='Prediction')
                axs[0, 1].set_title("Imaginary Part")
                axs[0, 1].legend()
                
                # Magnitude
                axs[1, 0].plot(abs(batch_output[i, 0].cpu())**2, label='Ground Truth')
                axs[1, 0].plot(abs(predictions[i, 0].cpu())**2, label='Prediction')
                axs[1, 0].set_title("Magnitude")
                axs[1, 0].legend()
                
                # Phase
                axs[1, 1].plot(abs(batch_input[i, 0].cpu())**2, label='Ground Truth')
                axs[1, 1].set_title("Input")
                axs[1, 1].legend()
                
                plt.tight_layout()
                plt.savefig('baseline.png')
            
            break  # Only process the first batch

def visualize_error(model, test_loader, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        for batch_input, batch_output in test_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            predictions = model(batch_input)
            
            for i in range(num_samples):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f"Error Analysis - Sample {i+1}")
                
                # Absolute Error
                abs_error = abs(predictions[i, 0] - batch_output[i, 0]).cpu()
                axs[0, 0].plot(abs_error)
                axs[0, 0].set_title("Absolute Error")
                
                # Relative Error
                rel_error = abs_error / (abs(batch_output[i, 0].cpu()) + 1e-10)
                axs[0, 1].plot(rel_error)
                axs[0, 1].set_title("Relative Error")
                
                # Squared Magnitude Error
                sq_mag_error = (abs(predictions[i, 0].cpu())**2 - abs(batch_output[i, 0].cpu())**2)**2
                axs[1, 0].plot(sq_mag_error)
                axs[1, 0].set_title("Squared Magnitude Error")
                
                # Phase Error
                phase_error = np.angle(predictions[i, 0].cpu()) - np.angle(batch_output[i, 0].cpu())
                phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                axs[1, 1].plot(phase_error)
                axs[1, 1].set_title("Phase Error")
                
                plt.tight_layout()
                plt.savefig('base_error.png')
            
            break  # Only process the first batch
            

def visualize_predictions1(model, test_loader, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        for batch_input, batch_output in test_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            predictions = model(batch_input)
            
            # Create a single figure with subplots for each sample
            fig, axs = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
            fig.suptitle("Squared Magnitude (|z|^2) Comparison", fontsize=16)
            
            for i in range(min(num_samples, len(batch_input))):
                # Squared Magnitude (|z|^2)
                print(batch_input[i, 0].shape, batch_output[i, 0].shape, predictions[i, 0].shape)
                axs[i].plot(abs(batch_output[i, 0].squeeze().cpu())**2, label='Ground Truth')
                axs[i].plot(abs(predictions[i, 0].squeeze().cpu())**2, label='Prediction')
                axs[i].set_title(f"Sample {i+1}")
                axs[i].set_xlabel("Position")
                axs[i].set_ylabel("|z|^2")
                axs[i].legend()
            
            plt.tight_layout()
            plt.savefig('base_vis.png')
            
            break  # Only process the first batch
# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = ["/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/emSHG_NewData.csv"]#['/pscratch/sd/r/rgeorge/SHG_output_final-main.csv']
    #file_path = ["/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/simpler_emSHG_9920.csv"]#[f"/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/emSHG_out_{i}.csv" for i in range(1, 21)] #['/pscratch/sd/r/rgeorge/SHG_output_final-main.csv']
    input_data, output_series = load_and_preprocess_data(file_path, num=2048, samples=20000, use_fft=False, use_truncation=False, values=[500, 1300], l=2)

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(input_data, output_series)

    # Print some information about the loaded data
    print(f"Total number of samples: {len(input_data)}")
    print(f"Input data shape: {input_data.shape}")
    print(f"Output series shape: {output_series.shape}")

    # Create and train the model
    in_channels = input_data.shape[1]  # Should be 4
    out_channels = output_series.shape[1]  # Should be 1
    hidden_channels = 2048

    model = ComplexCNN(in_channels, hidden_channels, out_channels, num_layers=6)
    print(model)
    model = train_model(model, train_loader, test_loader, epochs=200)
    torch.save(model.state_dict(), f'/pscratch/sd/r/rgeorge/model_base1.pt')
    model.load_state_dict(torch.load(f'/pscratch/sd/r/rgeorge/model_base1.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    visualize_predictions1(model, test_loader, num_samples=10)
    visualize_predictions(model, test_loader, num_samples=1)
    visualize_error(model, test_loader, num_samples=1)
