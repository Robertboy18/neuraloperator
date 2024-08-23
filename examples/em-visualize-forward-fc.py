import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO
from torch.utils.data import DataLoader
import pandas as pd
from scipy.fft import fft
from neuralop.layers.fourier_continuation import FCLegendre

def load_pretrained_model(model_path):
    model = FNO(n_modes=(64,), in_channels=4, out_channels=1, hidden_channels=512, projection_channels=256, n_layers=4, complex_spatial_data=True, domain_padding=1.0)  #FNO(n_modes=(1024,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_and_preprocess_data(file_path, num=2048, samples=1000, use_fft=False, use_fc=False):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract features (first 3 columns)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values[:samples]

    # Function to convert string representation of complex numbers to complex values
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    # Extract and convert input time series (Input_0 to Input_2047)
    samples1 = np.random.randint(0, df.shape[0], samples)
    input_columns = [f'Input_{i}' for i in range(num)]
    input_series = df[input_columns].map(to_complex).values[samples1]

    # Extract and convert output time series (Output_0 to Output_2047)
    output_columns = [f'Output_{i}' for i in range(num)]
    output_series = df[output_columns].map(to_complex).values[samples1]

    # Prepare input data with shape (num_samples, 4, num)
    input_data = np.zeros((samples, 4, num), dtype=np.complex128)
    
    # Repeat feature values num times for the first 3 channels
    for i in range(3):
        input_data[:, i, :] = np.tile(features[:, i], (num, 1)).T
    
    # Add the input series as the 4th channel
    input_data[:, 3, :] = input_series

    # Reshape output to (num_samples, 1, num)
    output_data = output_series.reshape(-1, 1, num)

    if use_fft:
        # Apply FFT to the input series (4th channel of input_data)
        input_data[:, 3, :] = fft(input_data[:, 3, :], axis=1)
        #print(input_data.shape)
        
        # Apply FFT to the output data
        output_data = fft(output_data, axis=2)
        #print(output_data.shape)

    return input_data, output_data, features

def predict_and_visualize(model, input_data, actual_output, features, use_fc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(input_data.shape, actual_output.shape)
    model = model.to(device)
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.complex64).to(device)
        if use_fc:
            number_additional_points=16
            n = 3
            d= number_additional_points
            fc = FCLegendre(n, d, dtype=torch.complex64)
            input_tensor = fc(torch.tensor(input_data, dtype=torch.complex64))
            actual_output = fc(torch.tensor(actual_output, dtype=torch.complex64))
        predicted_output = model(input_tensor).cpu().numpy()
    
    num_samples = input_data.shape[0]
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = [axs]
        
    actual_output = np.array(actual_output).reshape(num_samples, 1, 2048)
    
    print(f"Input shape: {input_data.shape}, Output shape: {actual_output.shape}, Predicted shape: {predicted_output.shape}")
    for i in range(num_samples):
        axs[i, 1].plot(np.abs(actual_output[i, 0, :])**2, label='Actual')
        axs[i, 1].plot(np.abs(predicted_output[i, 0, :])**2, label='Predicted')
        axs[i, 1].set_title(f"Sample {i+1}: Length={features[i,0]:.2f}, Mismatch={features[i,1]:.2f}, Energy={features[i,2]:.2f}")
        axs[i, 1].set_xlabel('Time Step')
        axs[i, 1].set_ylabel('Magnitude')
        axs[i, 1].legend()
    # now plot the input samples
    for i in range(num_samples):
        axs[i, 0].plot(np.abs(input_data[i, 3, :])**2)
        axs[i, 0].set_title(f"Input Sample {i+1}: Length={features[i,0]:.2f}, Mismatch={features[i,1]:.2f}, Energy={features[i,2]:.2f}")
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.savefig('Forward-problem-em.png')
    plt.show()


def main():
    model_path = '/raid/robert/em/model.pt'
    data_path = '/raid/robert/em/SHG_output_final.csv'
    
    model = load_pretrained_model(model_path)
    input_data, actual_output, features = load_and_preprocess_data(data_path, samples=5, use_fft=True)
    
    predict_and_visualize(model, input_data, actual_output, features)
    #predict_and_visualize_complex(model, input_data, actual_output, features)

if __name__ == "__main__":
    main()