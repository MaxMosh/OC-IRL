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
from sklearn.preprocessing import StandardScaler
import joblib 

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
# Assuming you are using the Transformer architecture
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- HYPERPARAMETERS ---
BATCH_SIZE = 512
LR = 5e-4
EPOCHS = 60000
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 # Set to 0 for stability on Windows/WSL

# Model Config
W_DIM = 15 
INPUT_CHANNELS = 4 
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

CHECKPOINT_DIR = "checkpoints_no_scaling"
# Look for the dataset generated without OCP scaling factors
DATASET_PATH = "data/dataset_parallel_NO_SCALING.pkl" 

class VariableLengthDataset(Dataset):
    def __init__(self, data_path, scaler_traj=None, scaler_w=None, random_slice=True, min_len=20):
        print("Loading dataset into RAM...")
        # Fallback search if exact path doesn't exist
        if not os.path.exists(data_path):
             files = [f for f in os.listdir("data") if f.endswith(".pkl") and "NO_SCALING" in f]
             if files:
                 data_path = os.path.join("data", sorted(files)[-1])
                 print(f"Found dataset: {data_path}")
        
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.w_matrices = raw_data["w_matrices"] # List of (5, 3) arrays
        self.q_trajs = raw_data["q_trajs"]       # List of (N, 2) arrays
        self.dq_trajs = raw_data["dq_trajs"]     # List of (N, 2) arrays
        
        self.random_slice = random_slice
        self.min_len = min_len
        self.scaler_traj = scaler_traj
        self.scaler_w = scaler_w

        # --- PRE-COMPUTE NORMALIZED DATA (Faster Training) ---
        print("Normalizing data in memory (StandardScaler)...")
        
        # 1. Normalize Weights
        # Flatten all w to (N_samples, 15) for scaling
        w_flat_list = [w.flatten() for w in self.w_matrices]
        w_flat_array = np.array(w_flat_list)
        
        # Transform
        w_normalized = self.scaler_w.transform(w_flat_array)
        # Store back as list of tensors
        self.w_data = [torch.FloatTensor(w) for w in w_normalized]
        
        # 2. Normalize Trajectories
        # Loop is safer for memory with ragged arrays than massive concatenation
        self.traj_data = []
        for i in range(len(self.q_trajs)):
            q = self.q_trajs[i]
            dq = self.dq_trajs[i]
            # Combine (N, 4)
            combined = np.concatenate([q, dq], axis=1)
            # Scale
            combined_norm = self.scaler_traj.transform(combined)
            # Store
            self.traj_data.append(combined_norm)
            
        print("Normalization complete.")

    def __len__(self):
        return len(self.w_data)

    def __getitem__(self, idx):
        # Retrieve pre-normalized data
        traj_norm = self.traj_data[idx] # (N, 4) numpy
        w_norm = self.w_data[idx]       # (15,) tensor

        total_len = traj_norm.shape[0]

        if self.random_slice and total_len > self.min_len:
            slice_end = random.randint(self.min_len, total_len)
        else:
            slice_end = total_len
            
        traj_slice = traj_norm[:slice_end]
        
        # Convert to Tensor
        return torch.FloatTensor(traj_slice), w_norm

def collate_fn(batch):
    trajectories, weights = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in trajectories])
    max_len = max(lengths)
    
    padded_traj = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
    
    # Create Mask (True = Padding)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    
    # Transpose for Model (B, L, 4) -> (B, 4, L) if model expects channels first
    # Our Transformer Encoder expects (B, 4, L) input which it internally permutes.
    padded_traj = padded_traj.transpose(1, 2) 
    
    weights_stacked = torch.stack(weights)
    
    return padded_traj, weights_stacked, mask

def prepare_scalers(dataset_path):
    """
    Fits scalers on the entire dataset and saves them.
    """
    print("Computing Scalers...")
    
    # Handle dynamic path finding
    if not os.path.exists(dataset_path):
        files = [f for f in os.listdir("data") if f.endswith(".pkl") and "NO_SCALING" in f]
        if files:
            dataset_path = os.path.join("data", sorted(files)[-1])
        else:
            raise FileNotFoundError("No suitable dataset found for scaler fitting.")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. Fit Weight Scaler
    w_list = [w.flatten() for w in data["w_matrices"]]
    w_array = np.array(w_list)
    scaler_w = StandardScaler()
    scaler_w.fit(w_array)
    print(f"Weights Scaler: mean={scaler_w.mean_[:3]}..., scale={scaler_w.scale_[:3]}...")
    
    # 2. Fit Trajectory Scaler
    print("Sampling trajectories for scaler fitting...")
    all_points = []
    # Sample 5% of trajectories for fitting statistics to be fast and accurate enough
    indices = np.random.choice(len(data["q_trajs"]), size=min(10000, len(data["q_trajs"])), replace=False)
    
    for idx in indices:
        q = data["q_trajs"][idx]
        dq = data["dq_trajs"][idx]
        combined = np.concatenate([q, dq], axis=1) # (N, 4)
        all_points.append(combined)
        
    all_points_array = np.concatenate(all_points, axis=0)
    scaler_traj = StandardScaler()
    scaler_traj.fit(all_points_array)
    print(f"Traj Scaler: mean={scaler_traj.mean_}, scale={scaler_traj.scale_}")
    
    # Save
    joblib.dump(scaler_w, os.path.join(CHECKPOINT_DIR, "scaler_w.pkl"))
    joblib.dump(scaler_traj, os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl"))
    
    return scaler_traj, scaler_w

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Get Scalers (Load or Compute)
    scaler_w_path = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
    scaler_traj_path = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")
    
    if os.path.exists(scaler_w_path) and os.path.exists(scaler_traj_path):
        print("Loading existing scalers...")
        scaler_w = joblib.load(scaler_w_path)
        scaler_traj = joblib.load(scaler_traj_path)
    else:
        scaler_traj, scaler_w = prepare_scalers(DATASET_PATH)

    # Load Dataset with Scalers
    dataset = VariableLengthDataset(
        DATASET_PATH, 
        scaler_traj=scaler_traj, 
        scaler_w=scaler_w, 
        random_slice=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ConditionalDiffusionModel(
        w_dim=W_DIM, 
        input_channels=INPUT_CHANNELS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Use standard GradScaler
    scaler_amp = torch.amp.GradScaler('cuda')
    loss_fn = nn.MSELoss()

    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print("Starting Training with Scaled Data (No OCP Factors)...")

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
                
                noise_pred = model(w_noisy, t, trajectories, trajectory_mask=mask)
                loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

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