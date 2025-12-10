import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from tools.diffusion_model1 import ConditionalDiffusionModel

# Parameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10000              # Number of epochs: tried on 10, 100 and 1000
TIMESTEPS = 1000            # Number of diffusion steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SubTrajectoryDataset(Dataset):
    def __init__(self, trajectories, weights, min_len=5, max_len=50):
    # def __init__(self, trajectories, weights, min_len=40, max_len=50):
    # tried different min_len (5, 10, 40)
        # Convert to tensors upfront to avoid overhead in __getitem__
        self.trajectories = torch.FloatTensor(trajectories) # (N, 50, 2)
        self.weights = torch.FloatTensor(weights) # (N, 3)
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx] # (50, 2)
        w = self.weights[idx] # (3,)

        # Sample random length
        length = torch.randint(self.min_len, self.max_len + 1, (1,)).item()

        # Create padded trajectory and mask using PyTorch operations
        padded_traj = torch.zeros((self.max_len, 2))
        padded_traj[:length] = traj[:length]

        mask = torch.zeros((self.max_len, 1))
        mask[:length] = 1

        # Combine: (50, 2) + (50, 1) -> (50, 3) -> (3, 50)
        combined = torch.cat([padded_traj, mask], dim=1).transpose(0, 1)

        return combined, w

def main():
    # Data loading & preprocessing
    print("Loading data...")
    try:
        # Load numpy files
        w_data = np.load("data/array_w_10000.npy")               # Shape: (N, 3)
        # w_data = np.load("data/array_w_11.npy")
        traj_data = np.load("data/array_results_angles_10000.npy") # Shape: (N, 50, 2)
        # traj_data = np.load("data/array_results_angles_11.npy")
    except FileNotFoundError:
        print("Error: Files not found in 'data/' directory.")
        exit()

    # Normalizing weights w
    scaler_w = StandardScaler()
    w_data_normalized = scaler_w.fit_transform(w_data)

    # Save the scaler (be careful to use the same scaler for inference later
    with open('scaler_w.pkl', 'wb') as f:
        pickle.dump(scaler_w, f)

    dataset = SubTrajectoryDataset(traj_data, w_data_normalized)
    
    # Optimized DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,      # Parallel loading
        pin_memory=True,    # Faster host-to-device transfer
        persistent_workers=True # Keep workers alive
    )

    # Diffusion schedule setup (Linear)
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    def add_noise(x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    # Model & training
    model = ConditionalDiffusionModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for trajectories, w_real in dataloader:
            trajectories = trajectories.to(DEVICE, non_blocking=True)
            w_real = w_real.to(DEVICE, non_blocking=True)
            batch_current_size = w_real.shape[0]

            t = torch.randint(0, TIMESTEPS, (batch_current_size,), device=DEVICE).long()
            w_noisy, noise = add_noise(w_real, t)
            noise_pred = model(w_noisy, t, trajectories)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.5f}")

    # Save model
    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Model saved to 'diffusion_model.pth'")
    print("Scaler saved to 'scaler_w.pkl'")

if __name__ == "__main__":
    main()
