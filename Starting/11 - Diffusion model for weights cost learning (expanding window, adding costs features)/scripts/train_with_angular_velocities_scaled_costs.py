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
from tools.diffusion_model_with_angular_velocities_scaled_costs import ConditionalDiffusionModel

# Parameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10000
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 5
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2

# Checkpoint settings
CHECKPOINT_DIR = "checkpoints"
SAVE_INTERVAL = 2000

class FullTrajectoryDataset(Dataset):
    """
    Dataset that returns the full trajectory of angles and velocities.
    Data is expected to be normalized before passed here.
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
        
        # Concatenate angles and velocities along the feature dimension (Result: 50, 4)
        traj_combined = torch.cat([q, dq], dim=1)

        # Transpose to match Conv1d expectation: (Channels, Sequence_Length) -> (4, 50)
        traj_transposed = traj_combined.transpose(0, 1)

        return traj_transposed, w

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    print("Loading data...")
    try:
        # Load Raw Data
        w_data = np.load("data/array_w_simplex_21_lim_joint_velocities_800.npy") 
        traj_angles = np.load("data/array_results_angles_simplex_21_lim_joint_velocities_800.npy")
        traj_velocities = np.load("data/array_results_angular_velocities_simplex_21_lim_joint_velocities_800.npy")
        
        if w_data.shape[1] != W_DIM:
            print(f"Alert: data w is shape {w_data.shape[1]} but W_DIM={W_DIM}")
        
    except FileNotFoundError as e:
        print(f"Error: Files not found. {e}")
        exit()

    # --- DATA NORMALIZATION ---
    print("Normalizing data...")
    
    # 1. Normalize Weights (w)
    # Even though w sum to 1, they are in [0, 1]. Diffusion models prefer N(0,1) or [-1, 1].
    scaler_w = StandardScaler()
    w_data_normalized = scaler_w.fit_transform(w_data)
    
    # 2. Normalize Trajectories (q, dq)
    # We must flatten the time dimension to fit the scaler, then reshape back
    # Input shape: (N, Time, Feat). Reshape -> (N*Time, Feat)
    N_samples, N_time, N_feat_ang = traj_angles.shape
    
    # Angles
    scaler_q = StandardScaler()
    traj_angles_flat = traj_angles.reshape(-1, N_feat_ang)
    traj_angles_norm = scaler_q.fit_transform(traj_angles_flat).reshape(N_samples, N_time, N_feat_ang)
    
    # Velocities
    scaler_dq = StandardScaler()
    traj_velocities_flat = traj_velocities.reshape(-1, N_feat_ang)
    traj_velocities_norm = scaler_dq.fit_transform(traj_velocities_flat).reshape(N_samples, N_time, N_feat_ang)

    # Save Scalers for later inference
    with open('checkpoints/scaler_w.pkl', 'wb') as f:
        pickle.dump(scaler_w, f)
    with open('checkpoints/scaler_q.pkl', 'wb') as f:
        pickle.dump(scaler_q, f)
    with open('checkpoints/scaler_dq.pkl', 'wb') as f:
        pickle.dump(scaler_dq, f)
    
    print("Scalers saved. Creating dataset...")

    # Create dataset with NORMALIZED data
    dataset = FullTrajectoryDataset(traj_angles_norm, traj_velocities_norm, w_data_normalized)
    
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

    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for trajectories, w_real in dataloader:
            trajectories = trajectories.to(DEVICE, non_blocking=True)
            w_real = w_real.to(DEVICE, non_blocking=True) # Normalized w
            
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

        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"diffusion_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "diffusion_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()