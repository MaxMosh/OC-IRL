import torch
import numpy as np
import matplotlib
# matplotlib.use("Agg") # Décommentez si serveur sans écran
import matplotlib.pyplot as plt
import os
import time
import pickle # N'oubliez pas pickle pour charger le scaler
from tqdm import tqdm

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
import sys
sys.path.append(os.getcwd())

# Import du modèle (Transformer ou standard)
# Assurez-vous que c'est le fichier où vous avez défini la classe avec input_channels=4
from tools.diffusion_model_with_angular_velocities import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new import solve_DOC

# --- Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_INFERENCES = 50   # Nombre d'inférences
MAX_LEN = 50
W_DIM = 5
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2

# --- 1. Génération de la Vérité Terrain (Ground Truth) ---
print("--- 1. Generating Ground Truth ---")

# w_true = np.random.rand(W_DIM)
# w_true = w_true / np.sum(w_true) # Normalisation

# Bad case 1
# w_true = np.array([0.43899002, 0.27423741, 0.18114669, 0.03836526, 0.06726063])

# Bad case 2
w_true = np.array([0.0199002, 0.017423741, 0.08114669, 0.54836526, 0.86726063])
w_true = w_true / np.sum(w_true) # Normalisation


print(f"True Weights: {w_true}")

try:
    # x_fin et q_init selon votre config
    x_fin_val = 1.9
    q_init_val = [-np.pi/2, np.pi/2]
    
    traj_true_q, traj_true_dq = solve_DOC(w_true, x_fin=x_fin_val, q_init=q_init_val)
except Exception as e:
    print(f"Error solving OCP for ground truth: {e}")
    exit()

# --- 2. Chargement du Scaler et du Modèle ---
print("\n--- 2. Loading Scaler and Model ---")

# A. Chargement du Scaler (CRUCIAL pour éviter le biais)
scaler_path = 'scaler_w_new.pkl'
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler_w = pickle.load(f)
    print(f"Loaded scaler from {scaler_path}")
else:
    print(f"Error: Scaler file '{scaler_path}' not found. Did you run train.py with scaler enabled?")
    exit()

# B. Chargement du Modèle
model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)

# Ajustez le chemin ici selon votre dernière sauvegarde
model_path = "checkpoints/diffusion_model_final_with_scaler.pth" 
# model_path = "checkpoints/diffusion_model_epoch_2000.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded model from {model_path}")
else:
    print(f"Error: Model file '{model_path}' not found.")
    exit()

model.eval()

# Paramètres de diffusion
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# --- 3. Fonction d'échantillonnage ---
def sample_diffusion(model, condition_trajectory, n_samples):
    model.eval()
    with torch.no_grad():
        # Répétition de la condition pour le batch
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        
        # Bruit initial (Normal Centré Réduit)
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

# --- 4. Préparation et Inférence ---
print(f"\n--- 3. Running {N_INFERENCES} Inferences ---")

# Construction du tenseur d'entrée
observation_length = MAX_LEN
q_input = traj_true_q[:observation_length]
dq_input = traj_true_dq[:observation_length]
combined = np.concatenate([q_input, dq_input], axis=1) # (50, 4)
traj_tensor = torch.FloatTensor(combined).unsqueeze(0).transpose(1, 2).to(DEVICE)

t_start = time.time()

# Sampling (Sortie : Poids normalisés)
w_generated_normalized = sample_diffusion(model, traj_tensor, N_INFERENCES)

# Inverse Transform (Sortie : Poids réels)
w_generated = scaler_w.inverse_transform(w_generated_normalized.cpu().numpy())

t_end = time.time()
print(f"Inference time: {t_end - t_start:.4f}s")

# --- 5. Reconstruction des trajectoires (OCP Loop) ---
print(f"\n--- 4. Solving OCP for each predicted weight vector ({N_INFERENCES} solves) ---")

reconstructed_trajectories_q = []
valid_weights_indices = []

for i in tqdm(range(N_INFERENCES)):
    w_pred = w_generated[i]
    
    # Optional: Re-normalisation sur le simplexe si nécessaire
    # if np.sum(w_pred) > 0: w_pred = w_pred / np.sum(w_pred)
    
    try:
        q_sol, _ = solve_DOC(w_pred, x_fin=x_fin_val, q_init=q_init_val)
        reconstructed_trajectories_q.append(q_sol)
        valid_weights_indices.append(i)
    except Exception as e:
        pass

reconstructed_trajectories_q = np.array(reconstructed_trajectories_q)
print(f"Successfully reconstructed {len(reconstructed_trajectories_q)}/{N_INFERENCES} trajectories.")

# --- 6. Plotting - Figure 1: Distributions des Poids ---
fig_dist, axes = plt.subplots(1, W_DIM, figsize=(20, 5))
fig_dist.suptitle(f"Distribution of Predicted Weights (N={N_INFERENCES}) - With Scaler", fontsize=16)

colors = ['red', 'green', 'blue', 'purple', 'orange']

for i in range(W_DIM):
    ax = axes[i]
    
    # Histogramme des prédictions
    ax.hist(w_generated[:, i], bins=15, color=colors[i], alpha=0.5, density=True, label='Prediction Dist.')
    
    # Moyenne prédite
    w_mean = np.mean(w_generated[:, i])
    ax.axvline(w_mean, color=colors[i], linestyle='-', linewidth=2, label='Pred. Mean')
    
    # Vérité terrain
    ax.axvline(w_true[i], color='black', linestyle='dashed', linewidth=2, label='Ground Truth')
    
    ax.set_title(f"Weight w{i+1}")
    ax.set_xlabel("Value")
    ax.set_xlim(0, 1)
    
    # Limites dynamiques pour mieux voir (ou fixes si vous préférez 0-1)
    # ax.set_xlim(0, 1) 
    
    ax.legend()

plt.tight_layout()
plt.show()
fig_dist.savefig("distribution_weights_scaled.png")
print("Saved weight distributions to 'distribution_weights_scaled.png'")

# --- 7. Plotting - Figure 2: Trajectory Bundle ---
fig_traj, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(16, 6))
fig_traj.suptitle("Reconstructed Trajectories Bundle", fontsize=16)

traj_true_deg = np.degrees(traj_true_q)

# Plot q1
ax_q1.set_title("Joint Angle q1")
ax_q1.set_xlabel("Time Step")
ax_q1.set_ylabel("Angle (deg)")
ax_q1.grid(True, linestyle='--', alpha=0.3)

# Plot q2
ax_q2.set_title("Joint Angle q2")
ax_q2.set_xlabel("Time Step")
ax_q2.set_ylabel("Angle (deg)")
ax_q2.grid(True, linestyle='--', alpha=0.3)

# Bundle
for traj in reconstructed_trajectories_q:
    traj_deg = np.degrees(traj)
    ax_q1.plot(traj_deg[:, 0], color='blue', alpha=0.15, linewidth=1)
    ax_q2.plot(traj_deg[:, 1], color='red', alpha=0.15, linewidth=1)

# Ground Truth
ax_q1.plot(traj_true_deg[:, 0], color='black', linestyle='--', linewidth=2.5, label='Ground Truth')
ax_q2.plot(traj_true_deg[:, 1], color='black', linestyle='--', linewidth=2.5, label='Ground Truth')

from matplotlib.lines import Line2D
custom_lines_q1 = [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='black', lw=2, linestyle='--')]
ax_q1.legend(custom_lines_q1, ['Reconstructed Bundle', 'Ground Truth'])

custom_lines_q2 = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='black', lw=2, linestyle='--')]
ax_q2.legend(custom_lines_q2, ['Reconstructed Bundle', 'Ground Truth'])

plt.tight_layout()
plt.show()
fig_traj.savefig("trajectory_bundle_scaled.png")
print("Saved trajectory bundle to 'trajectory_bundle_scaled.png'")