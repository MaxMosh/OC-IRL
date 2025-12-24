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
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable import ConditionalDiffusionModel

# --- HYPERPARAMETERS OPTIMISÉS ---
BATCH_SIZE = 5096       # AUGMENTÉ (était 64). Essayez 512 ou 2048 si besoin.
LR = 1e-3
EPOCHS = 30
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8         # Augmenté pour utiliser vos coeurs CPU
PREFETCH_FACTOR = 4     # Prépare les batchs à l'avance

# Dimensions
W_DIM = 15 
INPUT_CHANNELS = 4 

CHECKPOINT_DIR = "checkpoints"
DATASET_PATH = "data/dataset_parallel_299862_samples.pkl" # Nom à vérifier

class VariableLengthDataset(Dataset):
    """
    Dataset handling variable length trajectories.
    """
    def __init__(self, data_path, random_slice=True, min_len=10):
        print("Chargement du dataset en RAM...")
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.w_matrices = raw_data["w_matrices"]
        self.q_trajs = raw_data["q_trajs"]
        self.dq_trajs = raw_data["dq_trajs"]
        
        self.random_slice = random_slice
        self.min_len = min_len
        print(f"Dataset chargé : {len(self.w_matrices)} trajectoires.")

    def __len__(self):
        return len(self.w_matrices)

    def __getitem__(self, idx):
        # Optimisation : Conversion Numpy -> Tensor faite ici pour éviter l'overhead plus tard
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
        
        # Retourne des Tensors directement
        return torch.from_numpy(traj_combined).float(), torch.from_numpy(w.flatten()).float()

def collate_fn(batch):
    # batch contient une liste de tuples (traj_tensor, w_tensor)
    trajectories, weights = zip(*batch)
    
    # Pad sequence attend (L, C), nos données sont (L, 4) car pas encore transposées
    # batch_first=True -> (B, L, 4)
    padded_traj = pad_sequence(trajectories, batch_first=True, padding_value=0.0)
    
    # Transpose finale pour le CNN : (B, L, 4) -> (B, 4, L)
    padded_traj = padded_traj.transpose(1, 2)
    
    weights_stacked = torch.stack(weights)
    
    return padded_traj, weights_stacked

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Trouve le dataset
    target_file = DATASET_PATH
    if not os.path.exists(target_file):
        files = [f for f in os.listdir("data") if f.endswith(".pkl") and "dataset" in f]
        if files:
            target_file = os.path.join("data", sorted(files)[-1]) # Prend le dernier
            print(f"Fichier par défaut non trouvé, utilisation de : {target_file}")
        else:
            print("Erreur: Aucun dataset .pkl trouvé dans data/")
            exit()

    dataset = VariableLengthDataset(target_file, random_slice=True, min_len=20)
    
    # OPTIMISATION DATALOADER
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,     # Parallélisation CPU
        pin_memory=True,             # Accélère transfert CPU -> GPU
        prefetch_factor=PREFETCH_FACTOR, # Prépare X batchs d'avance par worker
        persistent_workers=True      # Garde les workers en vie entre les epochs
    )

    # 2. Modèle
    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    
    # OPTIMISATION : Compilation (PyTorch 2.0+)
    # Si cela plante (sur Windows ou vieux GPU), commentez cette ligne
    try:
        model = torch.compile(model)
        print("Modèle compilé avec torch.compile() (Mode optimisé)")
    except Exception as e:
        print(f"Compilation ignorée ({e})")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # OPTIMISATION : Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.MSELoss()

    # Diffusion Schedule
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print(f"Démarrage de l'entraînement sur {DEVICE} avec Batch Size={BATCH_SIZE}...")

    # 3. Boucle d'entraînement
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for trajectories, w_real in dataloader:
            # Transfert asynchrone (non_blocking=True grâce au pin_memory)
            trajectories = trajectories.to(DEVICE, non_blocking=True) 
            w_real = w_real.to(DEVICE, non_blocking=True)             
            
            # --- AMP Context (Mixed Precision) ---
            with torch.cuda.amp.autocast():
                batch_size = w_real.shape[0]
                t = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
                
                noise = torch.randn_like(w_real)
                # Astuce : unsqueeze pour broadcasting propre
                sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
                
                w_noisy = sqrt_alpha_hat * w_real + sqrt_one_minus_alpha_hat * noise
                
                # Predict
                noise_pred = model(w_noisy, t, trajectories)
                loss = loss_fn(noise_pred, noise)

            # Backward avec Scaler (pour gérer float16)
            optimizer.zero_grad(set_to_none=True) # set_to_none est un poil plus rapide
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Logging temps/perf
        epoch_duration = time.time() - start_time
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.6f} | Temps: {epoch_duration:.2f}s")

        if (epoch + 1) % 1000 == 0:
            # Sauvegarder le modèle "dé-compilé" pour compatibilité max (si possible)
            # Sinon on sauvegarde state_dict normalement
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state_dict, os.path.join(CHECKPOINT_DIR, f"diff_model_epoch_{epoch+1}.pth"))

    # Sauvegarde finale
    final_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(final_dict, os.path.join(CHECKPOINT_DIR, "diff_model_final.pth"))
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()