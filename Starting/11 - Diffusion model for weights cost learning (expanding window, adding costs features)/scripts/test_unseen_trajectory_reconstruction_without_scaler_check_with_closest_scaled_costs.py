import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Pour éviter les erreurs d'affichage sur serveur
import matplotlib.pyplot as plt
import pickle
import os
import json
import time
import sys

# AJOUT DU CHEMIN COURANT
sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new import solve_DOC

# --- PARAMETRES ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES = 50      # Nombre de w générés pour faire une moyenne
MAX_LEN = 50        
W_DIM = 5           
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2

# Chemins de fichiers (doivent correspondre à ceux de l'entrainement/génération)
CHECKPOINT_PATH = "checkpoints/diffusion_model_final.pth" 
SCALE_FACTORS_PATH = "data/scale_factors_simplex_21.json" # Adapte le nom si tu as changé la résolution
SCALER_W_PATH = "checkpoints/scaler_w.pkl"
SCALER_Q_PATH = "checkpoints/scaler_q.pkl"
SCALER_DQ_PATH = "checkpoints/scaler_dq.pkl"

# --- 1. CHARGEMENT DES OUTILS (SCALERS & SCALE FACTORS) ---
print("Chargement des outils de normalisation...")

if not os.path.exists(SCALE_FACTORS_PATH):
    print(f"ERREUR CRITIQUE : Fichier {SCALE_FACTORS_PATH} introuvable.")
    print("Veuillez générer les données avec le script 'generate_OCP...' d'abord.")
    sys.exit()

with open(SCALE_FACTORS_PATH, 'r') as f:
    scale_factors = json.load(f)

# Chargement des Scalers (Pickle)
try:
    with open(SCALER_W_PATH, 'rb') as f: scaler_w = pickle.load(f)
    with open(SCALER_Q_PATH, 'rb') as f: scaler_q = pickle.load(f)
    with open(SCALER_DQ_PATH, 'rb') as f: scaler_dq = pickle.load(f)
except FileNotFoundError as e:
    print(f"Erreur de chargement des scalers : {e}")
    sys.exit()

# --- 2. GÉNÉRATION DE LA VÉRITÉ TERRAIN (GROUND TRUTH) ---
print("Génération de la trajectoire Ground Truth...")

# On tire un w aléatoire sur le simplexe
w_true = np.random.rand(W_DIM)
w_true = w_true / np.sum(w_true)

print(f"Poids réels (w_true): {np.round(w_true, 4)}")

try:
    # IMPORTANT : On utilise les scale_factors ici !
    traj_true_q, traj_true_dq = solve_DOC(
        w_true, 
        x_fin=1.9, 
        q_init=[-np.pi/2, np.pi/2], 
        scale_factors=scale_factors
    )
except Exception as e:
    print(f"Erreur lors de la résolution OCP Ground Truth: {e}")
    sys.exit()


# --- 3. PRÉPARATION DE L'ENTRÉE DU MODÈLE (NORMALISATION) ---

# Mise en forme pour le scaler : (Time, Features) -> Standardisation
q_input_norm = scaler_q.transform(traj_true_q)      # (50, 2)
dq_input_norm = scaler_dq.transform(traj_true_dq)   # (50, 2)

# Concaténation q et dq -> (50, 4)
combined_input = np.concatenate([q_input_norm, dq_input_norm], axis=1)

# Transformation en Tensor : (1, 4, 50) pour le Conv1d
traj_tensor = torch.FloatTensor(combined_input).unsqueeze(0).transpose(1, 2).to(DEVICE)


# --- 4. CHARGEMENT DU MODÈLE ET INFÉRENCE ---
model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"Modèle chargé depuis {CHECKPOINT_PATH}")
else:
    print(f"Erreur : Checkpoint introuvable à {CHECKPOINT_PATH}")
    sys.exit()

model.eval()

# Schedule de diffusion (doit correspondre à l'entrainement)
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

def sample_diffusion(model, condition_trajectory, n_samples):
    """ Génère n_samples vecteurs w potentiels """
    model.eval()
    with torch.no_grad():
        # Duplication de la condition pour le batch
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        
        # Bruit initial Gaussien (Normalisé N(0,1))
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            predicted_noise = model(w_current, t, cond_repeated)

            alpha_t = alpha[i]
            alpha_hat_t = alpha_hat[i]
            beta_t = beta[i]

            if i > 0:
                noise = torch.randn_like(w_current)
            else:
                noise = torch.zeros_like(w_current)

            w_current = (1 / torch.sqrt(alpha_t)) * (
                w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    return w_current

print(f"Inférence diffusion ({N_SAMPLES} samples)...")
t_start = time.time()
w_generated_norm = sample_diffusion(model, traj_tensor, N_SAMPLES) # Sortie normalisée
t_end = time.time()
print(f"Temps d'inférence : {t_end - t_start:.4f}s")


# --- 5. POST-TRAITEMENT DES RÉSULTATS ---

# A. Dénormalisation (Inverse Standard Scaler)
w_generated_raw = scaler_w.inverse_transform(w_generated_norm.cpu().numpy())

# B. Moyenne des prédictions (pour avoir un candidat robuste)
w_pred_raw_mean = np.mean(w_generated_raw, axis=0)

# C. Projection sur le Simplexe (Nettoyage physique)
# 1. ReLU : On force les valeurs négatives à 0
w_pred_relu = np.maximum(w_pred_raw_mean, 0)
# 2. Normalisation L1 : Somme à 1
if np.sum(w_pred_relu) > 1e-9:
    w_final = w_pred_relu / np.sum(w_pred_relu)
else:
    # Cas rare ou le modèle sort tout < 0 -> on met un fallback uniforme
    w_final = np.ones(W_DIM) / W_DIM

print(f"Poids prédits (final): {np.round(w_final, 4)}")


# --- 6. RECONSTRUCTION ET COMPARAISON ---
print("Reconstruction de la trajectoire avec les poids prédits...")

try:
    # IMPORTANT : On réutilise les scale_factors pour la validation physique
    traj_rec_q, traj_rec_dq = solve_DOC(
        w_final, 
        x_fin=1.9, 
        q_init=[-np.pi/2, np.pi/2],
        scale_factors=scale_factors
    )
except Exception as e:
    print(f"Erreur reconstruction : {e}")
    traj_rec_q = np.zeros_like(traj_true_q)


# Calcul RMSE (sur les angles en degrés)
traj_true_deg = np.degrees(traj_true_q)
traj_rec_deg = np.degrees(traj_rec_q)
rmse_q1 = np.sqrt(np.mean((traj_true_deg[:, 0] - traj_rec_deg[:, 0])**2))
rmse_q2 = np.sqrt(np.mean((traj_true_deg[:, 1] - traj_rec_deg[:, 1])**2))


# --- 7. PLOTTING ---
fig, (ax_w, ax_q1, ax_q2) = plt.subplots(1, 3, figsize=(18, 6))

# Plot Poids
indices = np.arange(W_DIM)
width = 0.35
ax_w.bar(indices - width/2, w_true, width, label='Vérité Terrain', color='black', alpha=0.7)
ax_w.bar(indices + width/2, w_final, width, label='Prédit (Diff)', color='orange', alpha=0.7)
ax_w.set_title('Comparaison des Poids w')
ax_w.set_ylabel('Valeur')
ax_w.set_xticks(indices)
ax_w.set_xticklabels([f'w{i}' for i in indices])
ax_w.legend()
ax_w.grid(True, alpha=0.3)

# Plot q1
ax_q1.plot(traj_true_deg[:, 0], 'k--', linewidth=2, label='q1 True')
ax_q1.plot(traj_rec_deg[:, 0], 'b-', linewidth=2, alpha=0.8, label='q1 Reconstruit')
ax_q1.set_title(f'Joint q1 (RMSE: {rmse_q1:.2f}°)')
ax_q1.legend()
ax_q1.grid(True, alpha=0.3)

# Plot q2
ax_q2.plot(traj_true_deg[:, 1], 'k--', linewidth=2, label='q2 True')
ax_q2.plot(traj_rec_deg[:, 1], 'r-', linewidth=2, alpha=0.8, label='q2 Reconstruit')
ax_q2.set_title(f'Joint q2 (RMSE: {rmse_q2:.2f}°)')
ax_q2.legend()
ax_q2.grid(True, alpha=0.3)

save_name = "test_result_new_pipeline.png"
plt.tight_layout()
fig.savefig(save_name)
print(f"Graphique sauvegardé sous : {save_name}")