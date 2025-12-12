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
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from tools.dan_model import DAN_WeightEstimator

# --- Configuration ---
BATCH_SIZE = 64
LR = 1e-3             # RNNs can often handle higher LR than Transformers
EPOCHS = 500          # Converges faster than Diffusion
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, weights):
        self.trajectories = torch.FloatTensor(trajectories)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # We return the full sequences. The RNN handles the causality.
        # Input: (50, 2), Target: (50, 3)
        return self.trajectories[idx], self.weights[idx]

def main():
    print(f"Running DAN training on {DEVICE}")
    
    # --- 1. Load Data ---
    print("Loading data...")
    try:
        w_data = np.load("data/array_w_variables_w_10000.npy") 
        traj_data = np.load("data/array_results_angles_variables_w_10000.npy")
    except FileNotFoundError:
        print("Error: .npy files not found.")
        exit()

    N, T, D_w = w_data.shape
    
    # --- 2. Scaling ---
    # Standardize weights to help convergence
    w_flat = w_data.reshape(-1, D_w)
    scaler_w = StandardScaler()
    w_flat_norm = scaler_w.fit_transform(w_flat)
    w_data_norm = w_flat_norm.reshape(N, T, D_w)

    with open('scaler_dan_w.pkl', 'wb') as f:
        pickle.dump(scaler_w, f)

    dataset = TrajectoryDataset(traj_data, w_data_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 3. Model Setup ---
    model = DAN_WeightEstimator(
        input_dim=2, 
        hidden_dim=HIDDEN_DIM, 
        w_dim=3
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    os.makedirs("trained_models_dan", exist_ok=True)
    loss_history = []

    # --- 4. Training Loop ---
    print("Starting training...")
    progress_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

    for epoch in progress_bar:
        model.train()
        epoch_loss = 0

        for traj_batch, w_batch in dataloader:
            traj_batch = traj_batch.to(DEVICE)
            w_batch = w_batch.to(DEVICE)
            
            # Forward pass (Sequential processing happens inside)
            w_pred = model(traj_batch)

            # Loss: Compare predicted sequence vs target sequence
            loss = loss_fn(w_pred, w_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        progress_bar.set_postfix({"Loss": f"{avg_loss:.6f}"})
        
        # Save best/checkpoint
        if (epoch + 1) % 100 == 0:
             torch.save(model.state_dict(), f"trained_models_dan/dan_model_{epoch+1}.pth")

    # Final Save
    torch.save(model.state_dict(), "trained_models_dan/dan_model_final.pth")
    print("Training Complete.")

    # --- 5. Plot Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='DAN Training Loss')
    plt.title('DAN Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log') # Log scale is often useful for regression losses
    plt.grid(True)
    plt.legend()
    plt.savefig('dan_training_loss.png')
    print("Loss plot saved.")

if __name__ == "__main__":
    main()