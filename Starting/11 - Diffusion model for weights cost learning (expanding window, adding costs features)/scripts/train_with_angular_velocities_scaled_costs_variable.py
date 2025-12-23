import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
import os
import sys
import random

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable import ConditionalDiffusionModel

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64
LR = 1e-3
# EPOCHS = 5000
EPOCHS = 5
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dimensions
# w is now a (5, 3) matrix (5 weights, 3 time phases). We flatten it to 15.
W_RAW_SHAPE = (5, 3) 
W_DIM = 15 
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2 

CHECKPOINT_DIR = "checkpoints"
DATASET_PATH = "data/dataset_parallel_1000_samples.pkl" # Adjust filename if needed based on previous step

class VariableLengthDataset(Dataset):
    """
    Dataset handling variable length trajectories.
    It performs 'progressive' augmentation: for each sample, 
    it returns a random sub-sequence of the trajectory.
    """
    def __init__(self, data_path, random_slice=True, min_len=10):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.w_matrices = raw_data["w_matrices"]
        self.q_trajs = raw_data["q_trajs"]
        self.dq_trajs = raw_data["dq_trajs"]
        
        self.random_slice = random_slice
        self.min_len = min_len

    def __len__(self):
        return len(self.w_matrices)

    def __getitem__(self, idx):
        # 1. Get raw data
        q = self.q_trajs[idx]       # (N, 2)
        dq = self.dq_trajs[idx]     # (N, 2)
        w = self.w_matrices[idx]    # (5, 3)

        total_len = q.shape[0]

        # 2. Random Slicing (Data Augmentation for Progressive Prediction)
        if self.random_slice and total_len > self.min_len:
            # Pick a random end point between min_len and full length
            slice_end = random.randint(self.min_len, total_len)
        else:
            slice_end = total_len
            
        q_slice = q[:slice_end]
        dq_slice = dq[:slice_end]

        # 3. Process inputs
        # Combine -> (Time, 4)
        traj_combined = np.concatenate([q_slice, dq_slice], axis=1)
        
        # Transpose for CNN -> (4, Time)
        traj_tensor = torch.FloatTensor(traj_combined).transpose(0, 1)

        # 4. Process weights
        # Flatten (5, 3) -> (15,)
        w_flat = torch.FloatTensor(w.flatten())
        
        return traj_tensor, w_flat

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences in a batch.
    It pads trajectories with zeros to the max length in the batch.
    """
    trajectories, weights = zip(*batch)
    
    # Pad sequences. 
    # trajectories is a list of tensors (Channels, Time).
    # pad_sequence expects (Time, ...), so we transpose back and forth.
    
    # 1. Transpose to (Time, Channels) for padding
    traj_list_t = [t.transpose(0, 1) for t in trajectories]
    
    # 2. Pad (batch_first=True -> Batch, Time, Channels)
    padded_traj_t = pad_sequence(traj_list_t, batch_first=True, padding_value=0.0)
    
    # 3. Transpose back to (Batch, Channels, Time) for CNN
    padded_traj = padded_traj_t.transpose(1, 2)
    
    # Stack weights
    weights_stacked = torch.stack(weights)
    
    return padded_traj, weights_stacked

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Find the dataset
    # If the specific file doesn't exist, try finding any .pkl in data/
    target_file = DATASET_PATH
    if not os.path.exists(target_file):
        files = [f for f in os.listdir("data") if f.endswith(".pkl") and "dataset" in f]
        if files:
            target_file = os.path.join("data", files[-1])
            print(f"Default file not found, using: {target_file}")
        else:
            print("Error: No dataset .pkl found in data/")
            exit()

    print(f"Loading dataset from {target_file}...")
    dataset = VariableLengthDataset(target_file, random_slice=True, min_len=20)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, # Important for variable lengths
        num_workers=4,
        pin_memory=True
    )

    # 2. Model Setup
    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Diffusion Schedule
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print(f"Starting training on {DEVICE}...")

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for trajectories, w_real in dataloader:
            trajectories = trajectories.to(DEVICE) # (B, 4, Time_Max)
            w_real = w_real.to(DEVICE)             # (B, 15)
            
            # Add Noise
            batch_size = w_real.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            
            noise = torch.randn_like(w_real)
            sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
            w_noisy = sqrt_alpha_hat * w_real + sqrt_one_minus_alpha_hat * noise
            
            # Predict
            noise_pred = model(w_noisy, t, trajectories)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {epoch_loss / len(dataloader):.6f}")

        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"diff_model_variable_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "diff_model_variable_final.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    main()