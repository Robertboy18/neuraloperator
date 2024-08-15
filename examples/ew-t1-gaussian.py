import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.utils import count_model_params
import pandas as pd 


def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y = np.add(y, gaussian(x, params[i], params[i+1], params[i+2]), out=y, casting='unsafe')
    return y

class SHGTimeSeriesDataset(Dataset):
    def __init__(self, input_series, output_series):
        self.input_series = torch.tensor(input_series, dtype=torch.complex64)
        self.output_series = torch.tensor(output_series, dtype=torch.complex64)

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, idx):
        return self.input_series[idx], self.output_series[idx]

def fit_gaussian_and_get_residual(y, num_gaussians=3):
    x = np.arange(len(y))
    initial_guess = [np.max(y), len(y)//2, len(y)//10] * num_gaussians
    try:
        popt, _ = curve_fit(multi_gaussian, x, y, p0=initial_guess, maxfev=5000)
        fitted = multi_gaussian(x, *popt)
        residual = y - fitted
        return fitted, residual
    except RuntimeError:
        print("Gaussian fitting failed. Returning original data.")
        return np.zeros_like(y), y
    
def load_and_preprocess_data(file_path, num=2048, samples=1000):
    df = pd.read_csv(file_path)
    features = df[['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']].values[:samples]
    
    def to_complex(s):
        return complex(s.strip('()').replace('j', 'j').replace(' ', ''))

    input_columns = [f'Input_{i}' for i in range(num)]
    input_series = df[input_columns].applymap(to_complex).values[:samples]

    output_columns = [f'Output_{i}' for i in range(num)]
    output_series = df[output_columns].applymap(to_complex).values[:samples]

    input_data = np.zeros((samples, 4, num), dtype=np.complex128)
    fitted_inputs = np.zeros((samples, num), dtype=np.float64)
    residual_inputs = np.zeros((samples, num), dtype=np.float64)

    for i in range(samples):
        fitted, residual = fit_gaussian_and_get_residual(input_series[i])
        fitted_inputs[i] = fitted
        residual_inputs[i] = residual
        
        for j in range(3):
            input_data[i, j, :] = features[i, j]
        input_data[i, 3, :] = fitted + residual * 1j  # Combine fitted and residual

    return input_data, fitted_inputs, residual_inputs, output_series

def create_dataloaders(input_data, residual_outputs, batch_size=32, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, residual_outputs, test_size=test_size, random_state=42
    )
    train_dataset = SHGTimeSeriesDataset(X_train, y_train)
    test_dataset = SHGTimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_fno_model(train_loader, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FNO(n_modes=(256,), in_channels=4, out_channels=1, hidden_channels=512, 
                projection_channels=256, n_layers=4, complex_spatial_data=True, domain_padding=None)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(model=model, n_epochs=100, device=device)
    trainer.train(train_loader=train_loader, test_loaders=test_loader,
                  optimizer=optimizer, scheduler=scheduler, 
                  training_loss=loss_fn, eval_losses={"L2": loss_fn})

    return model

def predict_and_visualize(model, input_data, fitted_outputs, residual_outputs, num_samples=5):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data[:num_samples], dtype=torch.complex64).to(device)
        predicted_residuals = model(input_tensor).cpu().numpy()

    fig, axs = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = [axs]

    for i in range(num_samples):
        actual_output = np.abs(fitted_outputs[i] + residual_outputs[i])
        predicted_output = np.abs(fitted_outputs[i] + predicted_residuals[i, 0])
        
        axs[i].plot(actual_output, label='Actual')
        axs[i].plot(predicted_output, label='Predicted')
        axs[i].set_title(f"Sample {i+1}: Length={input_data[i,0,0]:.2f}, Mismatch={input_data[i,1,0]:.2f}, Energy={input_data[i,2,0]:.2f}")
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Magnitude')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('FNO-Gaussian-Residual-Predictions.png')
    plt.show()

def main():
    file_path = "/raid/robert/em/SHG_output_final.csv"
    input_data, fitted_outputs, residual_outputs = load_and_preprocess_data(file_path)
    train_loader, test_loader = create_dataloaders(input_data, residual_outputs)

    model = train_fno_model(train_loader, test_loader)
    predict_and_visualize(model, input_data, fitted_outputs, residual_outputs)

if __name__ == "__main__":
    main()