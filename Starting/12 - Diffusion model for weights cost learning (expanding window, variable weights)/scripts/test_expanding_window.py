import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import sys
import os

# Ajout des chemins pour importer tes outils
sys.path.append(os.getcwd())
# sys.path.append(os.path.joins(os.path.dirname(__file__), 'tools'))

from tools.diffusion_model import TransformerDiffusionModel
from tools.OCP_solving_cpin import solve_DOC

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000  # Doit matcher l'entrainement
N_SAMPLES = 20    # Nombre de prédictions parallèles pour voir l'incertitude
SEQ_LEN = 50
MIN_OBSERVATION = 5 

# --- 1. Chargement des Ressources ---
print(f"Chargement du modèle sur {DEVICE}...")

# Charger le Scaler
try:
    with open('scaler_w.pkl', 'rb') as f:
        scaler_w = pickle.load(f)
except FileNotFoundError:
    print("Erreur: 'scaler_w.pkl' introuvable. Lancez l'entraînement d'abord.")
    exit()

# Charger le Modèle
model = TransformerDiffusionModel(
    seq_len=50,
    w_dim=3,
    cond_dim=3, # q1, q2, mask
    d_model=128,
    nhead=4,
    num_layers=4
).to(DEVICE)

try:
    # Charge le dernier modèle sauvegardé
    model.load_state_dict(torch.load("trained_models/diffusion_transformer_3000_epochs_1000.pth", map_location=DEVICE))
    # Si tu veux charger un epoch spécifique :
    # model.load_state_dict(torch.load("diffusion_transformer_1000.pth", map_location=DEVICE))
except FileNotFoundError:
    print("Erreur: Modèle .pth introuvable.")
    exit()

model.eval()

# Constantes de Diffusion (Linear Schedule)
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# --- 2. Génération d'une NOUVELLE Donnée Test ---
print("Génération d'une nouvelle trajectoire de test (jamais vue)...")

def generate_random_test_case():
    """Génère un cas unique basé sur ta logique de dataset"""
    while True:
        try:
            # 1. Paramètres aléatoires (Logique de ton script generate_OCP)
            w_base = 0.01
            log_w_max = np.random.uniform(np.log(1), np.log(20)) # Un peu moins extrême pour la visualisation
            w_max = np.exp(log_w_max)
            t_transition = np.random.randint(25, 45)
            k_intensity = np.random.uniform(1, 5)
            
            # 2. Construction de la séquence w(t)
            t = np.linspace(0, 49, 50)
            sigmoid = 1 / (1 + np.exp(-k_intensity * (t - t_transition)))
            w_3 = w_base + (w_max - w_base) * sigmoid
            
            # w1 et w2 constants
            w_1 = np.full(50, 0.01)
            w_2 = np.full(50, 0.01)
            
            w_true = np.column_stack((w_1, w_2, w_3)) # (50, 3)
            
            # 3. Résolution du problème OCP
            # Attention: solve_DOC attend w sous forme (N, 3) ou liste
            # Ton solveur semble accepter w comme tableau numpy (50, 3) grâce à ta modif
            q_res, dq_res = solve_DOC(w_true, x_fin=-1.0, q_init=[0, np.pi/4])
            
            return q_res, w_true
            
        except Exception as e:
            # Si ipopt échoue (ce qui arrive avec des poids extrêmes), on recommence
            # print(f"Solver failed ({e}), retrying...")
            continue

q_test, w_test_true = generate_random_test_case()
print("Trajectoire générée avec succès !")

# --- 3. Fonction d'Inférence (Sampling) ---
def sample_diffusion_sequence(model, condition_tensor, n_samples):
    """
    condition_tensor: (1, 50, 3) -> [q1, q2, mask]
    Retourne: (n_samples, 50, 3) dénormalisé
    """
    model.eval()
    with torch.no_grad():
        # Répéter la condition pour le batch
        cond_batch = condition_tensor.repeat(n_samples, 1, 1) # (N, 50, 3)
        
        # Bruit initial (Gaussian)
        w_current = torch.randn(n_samples, SEQ_LEN, 3).to(DEVICE)
        
        # Boucle de Denoising Inverse
        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            
            # Prédiction du bruit
            predicted_noise = model(w_current, t, cond_batch)
            
            alpha_t = alpha[i]
            alpha_hat_t = alpha_hat[i]
            beta_t = beta[i]
            
            if i > 0:
                noise = torch.randn_like(w_current)
            else:
                noise = torch.zeros_like(w_current)
                
            # Étape de Langevin (Formule DDPM)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
            
            w_current = coef1 * (w_current - coef2 * predicted_noise) + torch.sqrt(beta_t) * noise
            
    return w_current.cpu().numpy()

# --- 4. Animation ---
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

# Plot 1: Trajectoire Robot (Espace Articulaire)
ax_traj = fig.add_subplot(gs[0])
ax_traj.set_title("Observation de la Trajectoire (q1, q2)")
ax_traj.set_xlim(0, 50)
ax_traj.set_ylim(-1.0, 3.5) # Ajuster selon tes angles usuels
ax_traj.grid(True, alpha=0.3)

# Lignes de vérité (transparente)
ax_traj.plot(q_test[:, 0], 'b--', alpha=0.3, label="q1 (Futur Réel)")
ax_traj.plot(q_test[:, 1], 'g--', alpha=0.3, label="q2 (Futur Réel)")

# Lignes observées (animées)
line_q1, = ax_traj.plot([], [], 'b-', linewidth=2, label="q1 (Observé)")
line_q2, = ax_traj.plot([], [], 'g-', linewidth=2, label="q2 (Observé)")
ax_traj.legend(loc='upper right')

# Plot 2: Prédiction des Poids w3 (Séquence Temporelle)
ax_weight = fig.add_subplot(gs[1])
ax_weight.set_title("Estimation de la courbe de coût w3(t) (Pénalité Vitesse X)")
ax_weight.set_xlim(0, 50)
# On fixe Y pour voir la dynamique, adapte selon ton scaler/max w
ymax = np.max(w_test_true[:, 2]) * 1.5 
ax_weight.set_ylim(-0.5, max(10, ymax)) 
ax_weight.grid(True, alpha=0.3)

# Vérité Terrain w3
ax_weight.plot(w_test_true[:, 2], 'k--', linewidth=2, label="w3 (Vérité Terrain)", zorder=10)

# Zone de confiance (remplissage)
fill_area = ax_weight.fill_between([], [], [], color='red', alpha=0.2, label="Incertitude Modèle")
# Ligne moyenne
line_mean, = ax_weight.plot([], [], 'r-', linewidth=2, label="w3 (Prédiction Moyenne)")

ax_weight.legend(loc='upper left')

def update(frame):
    # frame va de MIN_OBSERVATION à 50
    
    # --- A. Préparation de la Condition ---
    # On masque tout ce qui est APRES 'frame'
    current_traj = q_test.copy()
    current_traj[frame:, :] = 0.0 # Masquage futur
    
    mask = np.zeros((SEQ_LEN, 1))
    mask[:frame] = 1.0
    
    # Concaténation [q1, q2, mask] -> (50, 3)
    cond_np = np.concatenate([current_traj, mask], axis=1)
    cond_tensor = torch.FloatTensor(cond_np).unsqueeze(0).to(DEVICE) # (1, 50, 3)
    
    # --- B. Inférence (Generation) ---
    # On génère N_SAMPLES courbes possibles pour w
    w_gen_norm = sample_diffusion_sequence(model, cond_tensor, N_SAMPLES)
    
    # Dénormalisation
    # Scaler attend (N_total, 3), donc on aplatit
    w_gen_flat = w_gen_norm.reshape(-1, 3)
    w_gen_real_flat = scaler_w.inverse_transform(w_gen_flat)
    w_gen_real = w_gen_real_flat.reshape(N_SAMPLES, SEQ_LEN, 3)
    
    # On s'intéresse surtout à w3 (index 2)
    w3_preds = w_gen_real[:, :, 2] # (N_samples, 50)
    
    # Calcul des statistiques par pas de temps
    mean_w3 = np.mean(w3_preds, axis=0)
    std_w3 = np.std(w3_preds, axis=0)
    
    # --- C. Mise à jour Graphique ---
    
    # 1. Trajectoire
    line_q1.set_data(np.arange(frame), q_test[:frame, 0])
    line_q2.set_data(np.arange(frame), q_test[:frame, 1])
    
    # 2. Poids
    line_mean.set_data(np.arange(SEQ_LEN), mean_w3)
    
    # # Update du fill_between (un peu hacky avec matplotlib animation)
    # ax_weight.collections.clear() # On supprime l'ancien fill
    # # On redessine le fill (Mean +/- 2 std)
    # ax_weight.fill_between(np.arange(SEQ_LEN), 
    #                        mean_w3 - 2*std_w3, 
    #                        mean_w3 + 2*std_w3, 
    #                        color='red', alpha=0.2)
    # Suppression des anciens fill_between (PolyCollection)
    for coll in list(ax_weight.collections):
        if isinstance(coll, matplotlib.collections.PolyCollection):
            coll.remove()

    # Nouveau fill_between
    ax_weight.fill_between(
        np.arange(SEQ_LEN),
        mean_w3 - 2 * std_w3,
        mean_w3 + 2 * std_w3,
        color='red', alpha=0.2
    )
    
    # Indicateur visuel du temps présent
    ax_traj.axvline(x=frame, color='gray', alpha=0.1) # Optionnel, peut charger le plot
    
    return line_q1, line_q2, line_mean

print("Génération de l'animation...")
frames = np.arange(MIN_OBSERVATION, SEQ_LEN + 1)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

save_name = "transformer_inference.gif"
ani.save(save_name, writer='pillow', fps=10)
print(f"Animation sauvegardée sous : {save_name}")
plt.show()
