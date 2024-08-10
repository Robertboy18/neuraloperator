import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

def load_pretrained_model(model_path):
    model = FNO(n_modes=(1024,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class SHGTimeSeriesDataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = torch.tensor(input_series, dtype=torch.float32)
        self.output_series = torch.tensor(output_series, dtype=torch.float32)

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]

def load_and_preprocess_data(file_path, num=2048):
    df = pd.read_csv(file_path)
    
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values

    def to_real(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', '')).real

    input_columns = [f'Input_{i}' for i in range(num)]
    input_series = df[input_columns].map(to_real).values

    input_data = np.zeros((len(features), 4, num), dtype=np.float32)
    for i in range(3):
        input_data[:, i, :] = np.tile(features[:, i], (num, 1)).T
    input_data[:, 3, :] = input_series

    output_columns = [f'Output_{i}' for i in range(num)]
    output_series = df[output_columns].map(to_real).values

    output_data = output_series.reshape(-1, 1, num)
    
    return input_data, output_data, features

def create_dataloaders(input_data, output_series, batch_size=32, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, output_series, test_size=test_size, random_state=42
    )

    train_dataset = SHGTimeSeriesDataset(X_train, y_train)
    test_dataset = SHGTimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def predict_and_visualize(model, dataloader, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    fig, axs = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = [axs]

    with torch.no_grad():
        for i, (input_batch, actual_output_batch) in enumerate(dataloader):
            if i >= num_samples:
                break

            input_batch = input_batch.to(device)
            predicted_output = model(input_batch).cpu().numpy()
            
            actual_output = actual_output_batch.numpy()

            axs[i].plot(actual_output[0, 0, :], label='Actual')
            axs[i].plot(predicted_output[0, 0, :], label='Predicted')
            
            features = input_batch[0, :3, 0].cpu().numpy()
            axs[i].set_title(f"Sample {i+1}: Length={features[0]:.2f}, Mismatch={features[1]:.2f}, Energy={features[2]:.2f}")
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Magnitude')
            axs[i].legend()

    plt.tight_layout()
    plt.savefig('Forward-problem-em-real.png')
    plt.show()

def main():
    model_path = '/raid/robert/em/model_weights/model_60.pth'
    data_path = '/raid/robert/em/SHG_output_final.csv'
    
    model = load_pretrained_model(model_path)
    input_data, output_data, features = load_and_preprocess_data(data_path)
    train_loader, test_loader = create_dataloaders(input_data, output_data)
    
    predict_and_visualize(model, test_loader, num_samples=5)

if __name__ == "__main__":
    main()