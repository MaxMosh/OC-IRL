import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from tools.diffusion_model import ConditionalDiffusionModel

# Parameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10000
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 5  # <--- NOUVEAU PARAMÈTRE

class SubTrajectoryDataset(Dataset):
    # ... (Cette classe n'a pas besoin de changement, elle s'adapte à la shape des données) ...
    def __init__(self, trajectories, weights, min_len=5, max_len=50):
        self.trajectories = torch.FloatTensor(trajectories)
        self.weights = torch.FloatTensor(weights) # Sera de shape (N, 5) automatiquement
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        w = self.weights[idx]
        
        length = torch.randint(self.min_len, self.max_len + 1, (1,)).item()
        padded_traj = torch.zeros((self.max_len, 2))
        padded_traj[:length] = traj[:length]
        mask = torch.zeros((self.max_len, 1))
        mask[:length] = 1
        combined = torch.cat([padded_traj, mask], dim=1).transpose(0, 1)

        return combined, w

def main():
    print("Loading data...")
    try:
        # ASSUREZ-VOUS QUE CE FICHIER CONTIENT BIEN DES DONNÉES DE DIMENSION 5
        w_data = np.load("data/array_w_10000.npy") 
        traj_data = np.load("data/array_results_angles_10000.npy")
        
        # Vérification de sécurité
        if w_data.shape[1] != W_DIM:
            print(f"Attention : Les données chargées ont une dimension {w_data.shape[1]} mais W_DIM={W_DIM}")
            # Vous pouvez soit arrêter le script, soit laisser planter plus loin si incompatible
            
    except FileNotFoundError:
        print("Error: Files not found in 'data/' directory.")
        exit()

    # Normalizing weights w
    scaler_w = StandardScaler()
    w_data_normalized = scaler_w.fit_transform(w_data)

    with open('scaler_w.pkl', 'wb') as f:
        pickle.dump(scaler_w, f)

    dataset = SubTrajectoryDataset(traj_data, w_data_normalized)
    
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

    # Initialisation du modèle avec w_dim=5
    model = ConditionalDiffusionModel(w_dim=W_DIM).to(DEVICE) # <--- MODIFICATION ICI
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Starting training on {DEVICE} with W_DIM={W_DIM}...")

    # ... (La boucle d'entraînement reste identique) ...
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

    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Model and Scaler saved.")

if __name__ == "__main__":
    main()
