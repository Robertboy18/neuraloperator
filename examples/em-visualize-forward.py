import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO
from torch.utils.data import DataLoader
import pandas as pd

def load_pretrained_model(model_path):
    model = FNO(n_modes=(64,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_and_preprocess_data(file_path, num_samples=5):
    df = pd.read_csv(file_path)
    
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))
    
    random_samples = np.random.choice(df.shape[0], num_samples, replace=False)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values[random_samples]
    input_series = df[[f'Input_{i}' for i in range(2048)]].map(to_complex).values[random_samples]
    output_series = df[[f'Output_{i}' for i in range(2048)]].map(to_complex).values[random_samples]
    
    input_data = np.zeros((num_samples, 4, 2048), dtype=np.complex128)
    for i in range(3):
        input_data[:, i, :] = np.tile(features[:, i], (2048, 1)).T
    input_data[:, 3, :] = input_series
    
    return input_data, output_series, features

def predict_and_visualize(model, input_data, actual_output, features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.complex64).to(device)
        predicted_output = model(input_tensor).cpu().numpy()
    
    num_samples = input_data.shape[0]
    fig, axs = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = [axs]
        
    actual_output = np.array(actual_output).reshape(num_samples, 1, 2048)
    
    print(f"Input shape: {input_data.shape}, Output shape: {actual_output.shape}, Predicted shape: {predicted_output.shape}")
    for i in range(num_samples):
        axs[i].plot(np.abs(actual_output[i, 0, :]), label='Actual')
        axs[i].plot(np.abs(predicted_output[i, 0, :]), label='Predicted')
        axs[i].set_title(f"Sample {i+1}: Length={features[i,0]:.2f}, Mismatch={features[i,1]:.2f}, Energy={features[i,2]:.2f}")
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Magnitude')
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig('Forward-problem-em.png')
    plt.show()

def predict_and_visualize_complex(model, input_data, actual_output, features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.complex64).to(device)
        predicted_output = model(input_tensor).cpu().numpy()
    
    num_samples = input_data.shape[0]
    fig, axs = plt.subplots(num_samples, 2, figsize=(20, 10*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, 2)
    
    actual_output = np.array(actual_output).reshape(num_samples, 1, 2048)
    
    print(f"Input shape: {input_data.shape}, Output shape: {actual_output.shape}, Predicted shape: {predicted_output.shape}")
    
    for i in range(num_samples):
        # Plot real part
        axs[i, 0].plot(actual_output[i, 0, :].real, label='Actual (Real)')
        axs[i, 0].plot(predicted_output[i, 0, :].real, label='Predicted (Real)')
        axs[i, 0].set_title(f"Sample {i+1} - Real Part\nLength={features[i,0]:.2f}, Mismatch={features[i,1]:.2f}, Energy={features[i,2]:.2f}")
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('Real Value')
        axs[i, 0].legend()
        
        # Plot imaginary part
        axs[i, 1].plot(actual_output[i, 0, :].imag, label='Actual (Imaginary)')
        axs[i, 1].plot(predicted_output[i, 0, :].imag, label='Predicted (Imaginary)')
        axs[i, 1].set_title(f"Sample {i+1} - Imaginary Part\nLength={features[i,0]:.2f}, Mismatch={features[i,1]:.2f}, Energy={features[i,2]:.2f}")
        axs[i, 1].set_xlabel('Time Step')
        axs[i, 1].set_ylabel('Imaginary Value')
        axs[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('Forward-problem-em-complex.png')
    plt.show()
    
def main():
    model_path = '/home/robert/repo/neuraloperator/examples/model_weights_small/model_20.pth'
    data_path = '/raid/robert/em/SHG_output_final.csv'
    
    model = load_pretrained_model(model_path)
    input_data, actual_output, features = load_and_preprocess_data(data_path, num_samples=10)
    
    predict_and_visualize(model, input_data, actual_output, features)
    predict_and_visualize_complex(model, input_data, actual_output, features)

if __name__ == "__main__":
    main()