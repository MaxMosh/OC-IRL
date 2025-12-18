import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import sys

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
from tools.diffusion_model import ConditionalDiffusionModel

# Parameters
BATCH_SIZE = 64 # Si erreur VRAM, passez à 32
LR = 1e-3       # Les transformers aiment parfois des LR plus faibles (ex: 1e-4), mais tentez 1e-3 d'abord.
EPOCHS = 20000
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 5
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2

# Checkpoint settings
CHECKPOINT_DIR = "checkpoints"
SAVE_INTERVAL = 2000

class FullTrajectoryDataset(Dataset):
    def __init__(self, angles, velocities, weights):
        self.angles = torch.FloatTensor(angles)          # (N, 50, 2)
        self.velocities = torch.FloatTensor(velocities)  # (N, 50, 2)
        self.weights = torch.FloatTensor(weights)        # (N, W_DIM)

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        q = self.angles[idx]
        dq = self.velocities[idx]
        w = self.weights[idx]
        
        # Concatenate: (50, 4)
        traj_combined = torch.cat([q, dq], dim=1)
        # Transpose: (4, 50) -> Le Transformer s'attend à recevoir (Batch, Channels, Time)
        # et fera sa propre permutation interne.
        traj_transposed = traj_combined.transpose(0, 1)

        return traj_transposed, w

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    print("Loading data...")
    try:
        w_data = np.load("data/array_w_simplex_21_lim_joint_velocities_800.npy") 
        traj_angles = np.load("data/array_results_angles_simplex_21_lim_joint_velocities_800.npy")
        traj_velocities = np.load("data/array_results_angular_velocities_simplex_21_lim_joint_velocities_800.npy")
        
        if w_data.shape[1] != W_DIM:
            print(f"Alert: data w is shape {w_data.shape[1]} but W_DIM={W_DIM}")
        
        if traj_angles.shape[0] != traj_velocities.shape[0]:
            print("Error: Dataset size mismatch.")
            exit()
            
    except FileNotFoundError as e:
        print(f"Error: Files not found. {e}")
        exit()

    # Dataset & Loader
    dataset = FullTrajectoryDataset(traj_angles, traj_velocities, w_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Diffusion Schedule
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    def add_noise(x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    # --- MODIFICATION 1 : Passage explicite des channels ---
    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE} with Transformer Encoder...")

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

            # --- MODIFICATION 2 : Gradient Clipping ---
            # Indispensable pour la stabilité des Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.5f}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "diffusion_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()