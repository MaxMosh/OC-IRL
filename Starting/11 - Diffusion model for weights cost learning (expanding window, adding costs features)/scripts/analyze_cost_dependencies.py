import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
import sys

# Import de vos modules
from tools.diffusion_model_with_angular_velocities import ConditionalDiffusionModel
from torch.utils.data import DataLoader

# --- Paramètres ---
CHECKPOINT_PATH = "checkpoints/diffusion_model_cnn_encoding/diffusion_model_final.pth" # Ajustez selon votre dernier fichier
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 5
TIMESTEPS = 1000
N_SAMPLES = 500  # Nombre de w qu'on va générer pour UNE trajectoire

# Noms de vos features (basé sur OCP_solving_cpin_new.py)
FEATURE_NAMES = ["dq1^2", "dq2^2", "vx^2", "Torque^2", "Energy"]

def load_model():
    model = ConditionalDiffusionModel(w_dim=W_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

def sample_weights(model, trajectory, n_samples):
    """
    Version robuste avec clamping pour éviter les NaNs.
    """
    # Dupliquer la trajectoire pour le batch
    traj_batch = trajectory.unsqueeze(0).repeat(n_samples, 1, 1).to(DEVICE)
    
    # Paramètres de diffusion
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    model.eval()
    
    # On commence avec du bruit pur
    w_t = torch.randn(n_samples, W_DIM).to(DEVICE)
    
    with torch.no_grad():
        for i in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            
            # 1. Prédiction du bruit
            predicted_noise = model(w_t, t_tensor, traj_batch)
            
            # Check de sécurité : si le réseau sort déjà des NaNs
            if torch.isnan(predicted_noise).any():
                print(f"Warning: NaN detected in model output at step {i}")
                predicted_noise = torch.nan_to_num(predicted_noise)

            # 2. Calcul des coefficients (DDPM standard)
            curr_alpha_hat = alpha_hat[i]
            curr_beta = beta[i]
            
            # Petit epsilon pour éviter la division par zéro dans la racine carrée
            coef1 = 1 / torch.sqrt(1 - curr_beta + 1e-8)
            coef2 = (1 - curr_beta) / torch.sqrt(1 - curr_alpha_hat + 1e-8)
            
            mean = coef1 * (w_t - coef2 * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(w_t)
                sigma = torch.sqrt(curr_beta)
                w_t = mean + sigma * noise
            else:
                w_t = mean
            
            # --- CORRECTION CRITIQUE : CLAMPING ---
            # On empêche les valeurs de partir à l'infini. 
            # Comme les données sont normalisées, elles sont rarement hors de [-5, 5].
            w_t = torch.clamp(w_t, -10.0, 10.0)

    return w_t.cpu().numpy()

def main():
    # 1. Charger les données (pour en prendre une au hasard)
    print("Chargement des données...")
    try:
        # Assurez-vous que les chemins sont corrects par rapport à l'endroit où vous lancez le script
        traj_angles = np.load("data/array_results_angles_simplex_21_lim_joint_velocities_800.npy")
        traj_velocities = np.load("data/array_results_angular_velocities_simplex_21_lim_joint_velocities_800.npy")
    except Exception as e:
        print(f"Erreur chargement données: {e}")
        return

    # Préparer une trajectoire spécifique (ex: l'index 10)
    idx_traj = 10 
    
    # Traitement identique à votre Dataset
    q = torch.FloatTensor(traj_angles[idx_traj])      # (50, 2)
    dq = torch.FloatTensor(traj_velocities[idx_traj]) # (50, 2)
    traj_combined = torch.cat([q, dq], dim=1)         # (50, 4)
    traj_transposed = traj_combined.transpose(0, 1)   # (4, 50) -> C'est ce que le modèle attend

    # 2. Charger le modèle
    model = load_model()
    
    # 3. Échantillonner la distribution P(w | traj)
    print(f"Sampling {N_SAMPLES} w vectors for trajectory {idx_traj}...")
    sampled_w = sample_weights(model, traj_transposed, N_SAMPLES)
    
    # --- ANALYSE DES DÉPENDANCES ---
    
    # A. Matrice de Corrélation
    # Si deux features sont corrélées (ex: 0.9), elles sont redondantes pour expliquer cette trajectoire
    corr_matrix = np.corrcoef(sampled_w, rowvar=False)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Corrélation des poids générés (Ambiguïté)")
    plt.show()
    
    # B. PCA (Principal Component Analysis)
    # Les vecteurs propres associés aux PETITES valeurs propres indiquent les directions contraintes
    # Les vecteurs propres associés aux GRANDES valeurs propres indiquent les directions d'incertitude (Dépendance)
    pca = PCA()
    pca.fit(sampled_w)
    
    explained_variance = pca.explained_variance_ratio_
    components = pca.components_
    
    print("\n--- Analyse PCA (Directions de dépendance) ---")
    print("Variance expliquée par composante :", explained_variance)
    
    # Affichage des composantes principales
    plt.figure(figsize=(10, 5))
    plt.bar(range(W_DIM), explained_variance)
    plt.xlabel("Composante Principale")
    plt.ylabel("Ratio de Variance Expliquée")
    plt.title("Spectre de la variance des poids (Si plat = tout est défini, Si pic = ambiguïté)")
    plt.show()

    # Si la première composante explique beaucoup de variance, c'est la direction d'ambiguïté majeure
    print("\nDirection de plus grande ambiguïté (Composante 0) :")
    for name, val in zip(FEATURE_NAMES, components[0]):
        print(f"{name}: {val:.3f}")
        
    print("\nInterprétation :")
    print("Si 'Energy' et 'Torque' ont des signes opposés et des valeurs fortes dans la Composante 0,")
    print("cela signifie que le modèle hésite entre mettre du poids sur l'un ou sur l'autre.")

if __name__ == "__main__":
    main()