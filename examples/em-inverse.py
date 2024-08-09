import torch
import numpy as np
from neuralop.models import FNO
import torch.optim as optim
import matplotlib.pyplot as plt

def load_pretrained_model(model_path):
    model = FNO(n_modes=(1024,), in_channels=4, out_channels=1, hidden_channels=512, n_layers=4, complex_spatial_data=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inverse_problem_solver(model, target_output, num_iterations=2000, lr=0.01):
    device = 'cuda'
    model = model.to(device)
    
    # Initialize full input randomly
    full_input = torch.randn(1, 4, 2048, dtype=torch.complex64, requires_grad=True, device=device)
    optimizer = optim.Adam([full_input], lr=lr)
    target_output = target_output.to(device)
    
    #print(model.device, target_output.device, full_input.device)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_output = model(full_input.to(device))
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(torch.view_as_real(predicted_output), torch.view_as_real(target_output))
        
        # Add regularization to encourage constant values in first 3 channels
        consistency_loss = torch.var(full_input[:, :3, :], dim=2).sum()
        total_loss = loss + 0.1 * consistency_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update input
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Iteration {i+1}/{num_iterations}, Loss: {loss.item()}, Consistency Loss: {consistency_loss.item()}')
    
    return full_input.detach().cpu().numpy()

def plot_results(original_input, recovered_input):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    channel_names = ['Poling Region Length', 'Poling Period Mismatch', 'Pump Energy', 'Input Signal']
    
    for i in range(4):
        axs[i].plot(original_input[0, i, :].real, label='Original (Real)')
        axs[i].plot(original_input[0, i, :].imag, label='Original (Imag)')
        axs[i].plot(recovered_input[0, i, :].real, label='Recovered (Real)')
        axs[i].plot(recovered_input[0, i, :].imag, label='Recovered (Imag)')
        axs[i].set_title(f'Channel {i+1}: {channel_names[i]}')
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the pretrained model
    model_path = '/home/robert/repo/neuraloperator/examples/model_weights/model_20.pth'
    model = load_pretrained_model(model_path)
    
    # Generate a sample input (you should replace this with your actual data)
    original_input = np.zeros((1, 4, 2048), dtype=np.complex64)
    original_input[0, :3, :] = np.array([1+1j, 2+2j, 3+3j])[:, np.newaxis]  # constant features
    original_input[0, 3, :] = np.random.randn(2048) + 1j * np.random.randn(2048)  # random input signal
    
    # Generate the corresponding output using the forward model
    with torch.no_grad():
        original_output = model(torch.from_numpy(original_input).to(next(model.parameters()).device))
    
    # Solve the inverse problem
    recovered_input = inverse_problem_solver(model, original_output)
    
    # Plot and compare the results
    plot_results(original_input, recovered_input)
    
    print("Recovered features (mean across time steps):")
    for i, name in enumerate(['Poling Region Length (mm)', 'Poling Period Mismatch (nm)', 'Pump Energy (fJ)']):
        print(f"{name}: {recovered_input[0, i, :].mean()}")

if __name__ == "__main__":
    main()