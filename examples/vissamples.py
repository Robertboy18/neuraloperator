import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def load_and_visualize_data(file_path, num_samples=5, num_points=2048):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract features (first 3 columns)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values

    # Function to convert string representation of complex numbers to complex values
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    # Randomly select samples
    samples = np.random.randint(0, df.shape[0], num_samples)

    # Extract and convert input and output time series
    input_columns = [f'Input_{i}' for i in range(num_points)]
    output_columns = [f'Output_{i}' for i in range(num_points)]
    input_series = df[input_columns].map(to_complex).values[samples]
    output_series = df[output_columns].map(to_complex).values[samples]

    # Compute FFT
    input_fft = fft(input_series)
    output_fft = fft(output_series)

    # Visualize the data
    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axs = axs.reshape(1, -1)

    for i in range(num_samples):
        # Plot input time domain
        axs[i, 0].plot(np.abs(input_series[i, :])**2)
        axs[i, 0].set_title(f"Sample {i+1} - Input (Time Domain)")
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('Magnitude Squared')

        # Plot input frequency domain
        axs[i, 1].plot(np.abs(input_fft[i, :])**2)
        axs[i, 1].set_title(f"Sample {i+1} - Input (Frequency Domain)")
        axs[i, 1].set_xlabel('Frequency')
        axs[i, 1].set_ylabel('Magnitude Squared')
        #axs[i, 1].set_xscale('log')
        #axs[i, 1].set_yscale('log')

        # Plot output time domain
        axs[i, 2].plot(np.abs(output_series[i, :])**2)
        axs[i, 2].set_title(f"Sample {i+1} - Output (Time Domain)")
        axs[i, 2].set_xlabel('Time Step')
        axs[i, 2].set_ylabel('Magnitude Squared')

        # Plot output frequency domain
        axs[i, 3].plot(np.abs(output_fft[i, :])**2)
        axs[i, 3].set_title(f"Sample {i+1} - Output (Frequency Domain)")
        axs[i, 3].set_xlabel('Frequency')
        axs[i, 3].set_ylabel('Magnitude Squared')
        #axs[i, 3].set_xscale('log')
        #axs[i, 3].set_yscale('log')

        # Add feature information to the first subplot
        feature_text = f"Length={features[samples[i],0]}, Mismatch={features[samples[i],1]}, Energy={features[samples[i],2]}"
        axs[i, 0].text(0.5, 1.1, feature_text, horizontalalignment='center', verticalalignment='center', transform=axs[i, 0].transAxes)

    plt.tight_layout()
    plt.savefig('Input-Output-FFT-visualization.png')
    plt.show()

def main():
    s = 1
    data_path = "/pscratch/sd/r/rgeorge/SHG_output_final-main.csv" if s== 1 else "/global/cfs/cdirs/m4505/tensorlab/EM_Dataset/emSHG_out_1.csv"
    load_and_visualize_data(data_path, num_samples=20)

if __name__ == "__main__":
    main()