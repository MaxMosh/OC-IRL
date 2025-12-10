import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm  # Library for progress bars
import os

# Ensure the file is named diffusion_model.py in the tools folder
import sys
sys.path.append(os.getcwd())
# sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from tools.diffusion_model import TransformerDiffusionModel

# --- Configuration ---
BATCH_SIZE = 64
LR = 1e-4             # Slightly lower learning rate for Transformers
EPOCHS = 3000         # Can often require fewer epochs than MLP, but 2000 is safe
TIMESTEPS = 1000      # Diffusion steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SequenceDataset(Dataset):
    def __init__(self, trajectories, weights, min_len=5, max_len=50):
        # trajectories: (N, 50, 2)
        # weights: (N, 50, 3)
        self.trajectories = torch.FloatTensor(trajectories)
        self.weights = torch.FloatTensor(weights)
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Retrieve full sequence
        traj_full = self.trajectories[idx] # (50, 2)
        w_full = self.weights[idx]         # (50, 3)

        # Simulate partial observation during movement
        # Choose a random cut-off point t_cut
        t_cut = torch.randint(self.min_len, self.max_len + 1, (1,)).item()

        # Create mask and conditional trajectory
        # Everything after t_cut is masked (set to 0)
        traj_obs = torch.zeros_like(traj_full)
        traj_obs[:t_cut] = traj_full[:t_cut]

        mask = torch.zeros((self.max_len, 1))
        mask[:t_cut] = 1.0

        # Final condition: [q1, q2, mask] -> (50, 3)
        cond = torch.cat([traj_obs, mask], dim=1)

        # Target: The COMPLETE weight sequence (Past + Guessed Future)
        return cond, w_full

def main():
    print(f"Running on {DEVICE}")
    
    # --- 1. Data Loading ---
    print("Loading data...")
    try:
        # Expected shapes: (N, 50, 3) for weights and (N, 50, 2) for trajectories
        # Using the filenames from your uploaded code
        w_data = np.load("data/array_w_variables_w_10000.npy") 
        traj_data = np.load("data/array_results_angles_variables_w_10000.npy")
    except FileNotFoundError:
        print("Error: .npy files not found. Please check data paths.")
        exit()

    N, T, D_w = w_data.shape
    
    # --- 2. Scaling (Tricky with 3D data) ---
    # Flatten to (N * 50, 3) for the scaler, then reshape back
    w_flat = w_data.reshape(-1, D_w)
    scaler_w = StandardScaler()
    w_flat_norm = scaler_w.fit_transform(w_flat)
    w_data_norm = w_flat_norm.reshape(N, T, D_w)

    # Save scaler for inference
    with open('scaler_w.pkl', 'wb') as f:
        pickle.dump(scaler_w, f)

    # Create Dataset
    dataset = SequenceDataset(traj_data, w_data_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 3. Diffusion Schedule ---
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    def add_noise(x, t):
        # x shape: (Batch, 50, 3)
        noise = torch.randn_like(x)
        
        # Reshape alphas to match (Batch, 1, 1) for broadcasting
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None]
        
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    # --- 4. Model Setup ---
    # Instantiate Transformer
    model = TransformerDiffusionModel(
        seq_len=50,
        w_dim=3,
        cond_dim=3, # q1, q2, mask
        d_model=128,
        nhead=4,
        num_layers=4
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR) # AdamW is usually better for Transformers
    loss_fn = nn.MSELoss()

    # Ensure save directory exists
    os.makedirs("trained_models", exist_ok=True)

    print("Starting training...")
    
    # List to store loss values for plotting
    loss_history = []

    # --- 5. Training Loop ---
    # TQDM wrapper for progress bar
    progress_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

    for epoch in progress_bar:
        model.train()
        epoch_loss = 0

        for cond, w_real in dataloader:
            cond = cond.to(DEVICE)     # (Batch, 50, 3)
            w_real = w_real.to(DEVICE) # (Batch, 50, 3)
            
            curr_batch = w_real.shape[0]

            # Sample time t
            t = torch.randint(0, TIMESTEPS, (curr_batch,), device=DEVICE).long()

            # Add noise
            w_noisy, noise = add_noise(w_real, t)

            # Model Prediction
            # The model takes Noisy W AND Condition (Partial Traj)
            noise_pred = model(w_noisy, t, cond)

            # Compute Loss
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Update progress bar description with current loss
        progress_bar.set_postfix({"Loss": f"{avg_loss:.6f}"})
            
        # Checkpoint saving
        if (epoch + 1) % 100 == 0:
             torch.save(model.state_dict(), f"trained_models/diffusion_transformer_{EPOCHS}_epochs_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), f"trained_models/diffusion_transformer_{EPOCHS}_epochs_final.pth")
    print("Training Complete.")

    # --- 6. Plotting Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_loss_plot_{EPOCHS}.png')
    print(f"Loss plot saved to 'training_loss_plot_{EPOCHS}.png'")

if __name__ == "__main__":
    main()