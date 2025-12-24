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
import time

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- HYPERPARAMETERS ---
# BATCH_SIZE = 512        # Reduced slightly for Transformer memory usage
BATCH_SIZE = 64
LR = 5e-4               # Transformers prefer slightly lower LR often
# EPOCHS = 5000
EPOCHS = 2
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8         # Keep 0 for WSL stability
PREFETCH_FACTOR = None

# Model Config
W_DIM = 15 
INPUT_CHANNELS = 4 
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

CHECKPOINT_DIR = "checkpoints"
DATASET_PATH = "data/dataset_parallel_299862_samples.pkl" # Your generated file

class VariableLengthDataset(Dataset):
    """
    Dataset handling variable length trajectories.
    """
    def __init__(self, data_path, random_slice=True, min_len=20):
        print("Loading dataset into RAM...")
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.w_matrices = raw_data["w_matrices"]
        self.q_trajs = raw_data["q_trajs"]
        self.dq_trajs = raw_data["dq_trajs"]
        
        self.random_slice = random_slice
        self.min_len = min_len
        print(f"Dataset loaded: {len(self.w_matrices)} samples.")

    def __len__(self):
        return len(self.w_matrices)

    def __getitem__(self, idx):
        q = self.q_trajs[idx]       
        dq = self.dq_trajs[idx]     
        w = self.w_matrices[idx]    

        total_len = q.shape[0]

        if self.random_slice and total_len > self.min_len:
            slice_end = random.randint(self.min_len, total_len)
        else:
            slice_end = total_len
            
        q_slice = q[:slice_end]
        dq_slice = dq[:slice_end]

        traj_combined = np.concatenate([q_slice, dq_slice], axis=1)
        
        # Return tensors directly
        return torch.from_numpy(traj_combined).float(), torch.from_numpy(w.flatten()).float()

def collate_fn(batch):
    """
    Custom collate function that handles padding AND creates the Attention Mask.
    """
    trajectories, weights = zip(*batch)
    
    # 1. Get lengths for masking
    lengths = torch.tensor([t.shape[0] for t in trajectories])
    max_len = max(lengths)
    
    # 2. Pad Sequences
    # pad_sequence expects (L, C), output is (B, L, C)
    padded_traj = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
    
    # 3. Create Padding Mask (Boolean)
    # Shape (B, L). True indicates padding (ignore), False indicates data.
    # We use broadcasting to create the mask.
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    
    # 4. Transpose Trajectory for Model Input (CNN/Encoder expects Channels first usually, 
    # but our Transformer Encoder code does the transpose internally. 
    # Let's check model code: Input (B, C, L). So we transpose.
    padded_traj = padded_traj.transpose(1, 2) # (B, 4, L)
    
    weights_stacked = torch.stack(weights)
    
    return padded_traj, weights_stacked, mask

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Find dataset
    target_file = DATASET_PATH
    if not os.path.exists(target_file):
        files = [f for f in os.listdir("data") if f.endswith(".pkl") and "dataset" in f]
        if files:
            target_file = os.path.join("data", sorted(files)[-1])
            print(f"Default file not found, using: {target_file}")
        else:
            print("Error: No dataset .pkl found.")
            exit()

    dataset = VariableLengthDataset(target_file, random_slice=True, min_len=20)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # 2. Model Setup
    model = ConditionalDiffusionModel(
        w_dim=W_DIM, 
        input_channels=INPUT_CHANNELS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    # Optional Compilation
    # model = torch.compile(model) 

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.MSELoss()

    # Diffusion Schedule
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print(f"Starting Transformer Training on {DEVICE}...")

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for trajectories, w_real, mask in dataloader:
            trajectories = trajectories.to(DEVICE, non_blocking=True) 
            w_real = w_real.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                batch_size = w_real.shape[0]
                t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
                
                noise = torch.randn_like(w_real)
                sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
                
                w_noisy = sqrt_alpha_hat * w_real + sqrt_one_minus_alpha_hat * noise
                
                # Predict (Pass the mask!)
                noise_pred = model(w_noisy, t, trajectories, trajectory_mask=mask)
                loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_duration = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Time: {epoch_duration:.2f}s")

        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"diff_model_transformer_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "diff_model_transformer_final.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    main()