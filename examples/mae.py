import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=(16, 16), hidden_channels=64):
        super().__init__()
        self.fno = FNO(in_channels=in_channels, out_channels=out_channels,
                       n_modes=modes, hidden_channels=hidden_channels)

    def forward(self, x):
        return self.fno(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.fno1 = FNOBlock(in_channels, 256)
        self.fno2 = FNOBlock(256, latent_dim)

    def forward(self, x):
        x = self.fno1(x)
        return self.fno2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.fno1 = FNOBlock(latent_dim, 256)
        self.fno2 = FNOBlock(256, out_channels)

    def forward(self, x):
        x = self.fno1(x)
        return self.fno2(x)

class FNOAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

def apply_mask(x, mask_ratio=0.5):
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H, W, device=x.device) > mask_ratio
    return x * mask

def apply_mask_square(x, mask_ratio=0.3):
    B, C, H, W = x.shape
    mask_size = int(H * mask_ratio)  # Size of the square mask
    mask = torch.ones_like(x)
    mask[:, :, :mask_size, :mask_size] = 0  # Mask out the left corner
    return x * mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=100, batch_size=16, 
        test_resolutions=[16], n_tests=[20],
        test_batch_sizes=[16],
        positional_encoding=True
    )
    data_processor = data_processor.to(device)

    # Create the model
    model = FNOAutoencoder(in_channels=3, out_channels=1, latent_dim=256).to(device)
    print(f'Model parameters: {count_model_params(model)}')

    # Define loss, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    n_epochs = 300
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = data_processor.preprocess(batch)
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)

            # Apply mask
            masked_x = apply_mask_square(x)

            optimizer.zero_grad()
            output = model(masked_x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loaders[16]:
                    batch = data_processor.preprocess(batch)
                    x, y = batch['x'], batch['y']
                    x, y = x.to(device), y.to(device)
                    masked_x = apply_mask_square(x)
                    output = model(masked_x)
                    val_loss += criterion(output, y).item()
            val_loss /= len(test_loaders[16])
            print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}')

    # Visualization
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loaders[16]))
        batch = data_processor.preprocess(batch)
        x, y = batch['x'], batch['y']
        x, y = x.to(device), y.to(device)
        masked_x = apply_mask_square(x)
        output = model(masked_x)

        # Plot results
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        for i in range(4):
            axes[0, i].imshow(masked_x[i, 0].cpu().numpy())
            axes[0, i].set_title('Masked Input')
            axes[1, i].imshow(y[i, 0].cpu().numpy())
            axes[1, i].set_title('Ground Truth')
            axes[2, i].imshow(output[i, 0].cpu().numpy())
            axes[2, i].set_title('Reconstruction')
        
        plt.tight_layout()
        plt.savefig('fno_reconstruction.png')
        plt.close()

    print("Training completed. Visualization saved as 'fno_reconstruction.png'")

if __name__ == "__main__":
    main()