import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from tools.diffusion_model import ConditionalDiffusionModel

# --- Configuration ---
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10000
TIMESTEPS = 1000 # Number of diffusion steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Data Loading & Preprocessing ---
print("Loading data...")
try:
    # Load numpy files
    w_data = np.load("data/array_w_101.npy")               # Shape: (N, 3)
    traj_data = np.load("data/array_results_angles_101.npy") # Shape: (N, 50, 2)
except FileNotFoundError:
    print("Error: Files not found in 'data/' directory.")
    exit()

# Normalize 'w' (Crucial for diffusion models to work effectively)
# Diffusion models expect data to be roughly Standard Normal N(0,1)
scaler_w = StandardScaler()
w_data_normalized = scaler_w.fit_transform(w_data)

# Save the scaler for inference later
with open('scaler_w.pkl', 'wb') as f:
    pickle.dump(scaler_w, f)

class SubTrajectoryDataset(Dataset):
    def __init__(self, trajectories, weights, min_len=5, max_len=50):
        self.trajectories = trajectories # (N, 50, 2)
        self.weights = weights # (N, 3)
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx] # (50, 2)
        w = self.weights[idx] # (3,)

        # Sample random length
        length = np.random.randint(self.min_len, self.max_len + 1)

        # Slice trajectory
        sub_traj = traj[:length]

        # Pad with zeros
        padded_traj = np.zeros((self.max_len, 2))
        padded_traj[:length] = sub_traj

        # Create mask
        mask = np.zeros((self.max_len, 1))
        mask[:length] = 1

        # Combine padded trajectory and mask
        # Shape: (50, 3) -> (3, 50) for Conv1d
        combined = np.concatenate([padded_traj, mask], axis=1)
        combined_tensor = torch.FloatTensor(combined).transpose(0, 1)

        w_tensor = torch.FloatTensor(w)

        return combined_tensor, w_tensor

dataset = SubTrajectoryDataset(traj_data, w_data_normalized)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. Diffusion Schedule Setup (Linear) ---
# Define beta schedule (variance of noise)
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0) # Cumulative product

def add_noise(x, t):
    """
    Adds Gaussian noise to x at step t using the closed-form formula.
    Returns: noisy_x, noise_added
    """
    noise = torch.randn_like(x)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]

    # x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
    x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
    return x_noisy, noise

# --- 3. Model & Training ---
model = ConditionalDiffusionModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss() # We predict the noise, so MSE is appropriate

print(f"Starting training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for trajectories, w_real in dataloader:
        trajectories = trajectories.to(DEVICE)
        w_real = w_real.to(DEVICE)
        batch_current_size = w_real.shape[0]

        # 1. Sample random timesteps for each sample in batch
        t = torch.randint(0, TIMESTEPS, (batch_current_size,), device=DEVICE).long()

        # 2. Add noise to the real w
        w_noisy, noise = add_noise(w_real, t)

        # 3. Predict the noise using the model
        # The model tries to predict 'noise' given 'w_noisy' and 'trajectories'
        noise_pred = model(w_noisy, t, trajectories)

        # 4. Backpropagation
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Simple logging
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.5f}")

# --- 4. Save Model ---
torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved to 'diffusion_model.pth'")
print("Scaler saved to 'scaler_w.pkl'")