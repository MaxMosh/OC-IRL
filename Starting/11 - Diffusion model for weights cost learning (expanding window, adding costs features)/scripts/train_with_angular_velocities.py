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
from tools.diffusion_model_with_angular_velocities import ConditionalDiffusionModel

# Parameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20000
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 5
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2 (No mask needed anymore)

# Checkpoint settings
CHECKPOINT_DIR = "checkpoints"  # Subfolder name
SAVE_INTERVAL = 2000            # Save every 2000 epochs

class FullTrajectoryDataset(Dataset):
    """
    Dataset that returns the full trajectory of angles and velocities.
    No subsampling, no padding mask.
    """
    def __init__(self, angles, velocities, weights):
        # Convert to tensors
        self.angles = torch.FloatTensor(angles)          # Shape: (N, 50, 2)
        self.velocities = torch.FloatTensor(velocities)  # Shape: (N, 50, 2)
        self.weights = torch.FloatTensor(weights)        # Shape: (N, W_DIM)

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        q = self.angles[idx]      # (50, 2)
        dq = self.velocities[idx] # (50, 2)
        w = self.weights[idx]     # (W_DIM,)
        
        # Concatenate angles and velocities along the feature dimension
        # Result shape: (50, 4)
        traj_combined = torch.cat([q, dq], dim=1)

        # Transpose to match Conv1d expectation: (Sequence_Length, Channels) -> (Channels, Sequence_Length)
        # Result shape: (4, 50)
        traj_transposed = traj_combined.transpose(0, 1)

        return traj_transposed, w

def main():
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    print("Loading data...")
    try:
        w_data = np.load("data/array_w_simplex_21_lim_joint_velocities_800.npy") 
        traj_angles = np.load("data/array_results_angles_simplex_21_lim_joint_velocities_800.npy")
        traj_velocities = np.load("data/array_results_angular_velocities_simplex_21_lim_joint_velocities_800.npy")
        
        # Dimension check
        if w_data.shape[1] != W_DIM:
            print(f"Alert: data w is shape {w_data.shape[1]} but W_DIM={W_DIM}")
        
        # Consistency check
        if traj_angles.shape[0] != traj_velocities.shape[0]:
            print("Error: Number of angle trajectories does not match number of velocity trajectories.")
            exit()
            
    except FileNotFoundError as e:
        print(f"Error: Files not found in 'data/' directory. {e}")
        exit()

    # Normalizing weights w (Commented out as per original file)
    # scaler_w = StandardScaler()
    # w_data_normalized = scaler_w.fit_transform(w_data)

    # with open('scaler_w.pkl', 'wb') as f:
    #     pickle.dump(scaler_w, f)

    # dataset = FullTrajectoryDataset(traj_angles, traj_velocities, w_data_normalized)
    dataset = FullTrajectoryDataset(traj_angles, traj_velocities, w_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    def add_noise(x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    # NOTE: Ensure your ConditionalDiffusionModel accepts the correct input channels if needed
    model = ConditionalDiffusionModel(w_dim=W_DIM).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE} with W_DIM={W_DIM} and Input Channels={INPUT_CHANNELS}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for trajectories, w_real in dataloader:
            trajectories = trajectories.to(DEVICE, non_blocking=True)
            w_real = w_real.to(DEVICE, non_blocking=True)
            batch_current_size = w_real.shape[0]

            t = torch.randint(0, TIMESTEPS, (batch_current_size,), device=DEVICE).long()
            w_noisy, noise = add_noise(w_real, t)
            
            # trajectories shape is now (Batch, 4, 50)
            noise_pred = model(w_noisy, t, trajectories)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.5f}")

        # Periodic Saving
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    # Final Save
    final_path = os.path.join(CHECKPOINT_DIR, "diffusion_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()