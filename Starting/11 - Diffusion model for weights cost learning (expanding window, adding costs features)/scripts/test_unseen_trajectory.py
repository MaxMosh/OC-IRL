import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from tools.diffusion_model import ConditionalDiffusionModel
from tools.OCP_solving_cpin import solve_DOC

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES = 50
MAX_LEN = 50
MIN_LEN = 5
W_DIM = 5 # <--- NOUVEAU PARAMÈTRE

# Generation of an unseen trajectory
print("Generating unseen trajectory...")
# Generate random w de taille 5
w_unseen = np.random.rand(W_DIM) # <--- MODIFICATION
w_unseen = w_unseen / np.sum(w_unseen)
print(f"Generated w: {w_unseen}")

try:
    # On suppose que solve_DOC accepte un vecteur de taille 5
    results_angles, results_velocities = solve_DOC(w_unseen, x_fin=-1.0, q_init=np.array([0, np.pi/4]))
    traj_sample = results_angles 
except Exception as e:
    print(f"Error solving OCP: {e}")
    exit()

print("Loading model...")
with open('scaler_w.pkl', 'rb') as f:
    scaler_w = pickle.load(f)

# Initialisation avec w_dim=5
model = ConditionalDiffusionModel(w_dim=W_DIM).to(DEVICE) # <--- MODIFICATION
# Assurez-vous de charger un modèle entrainé avec w_dim=5 !
model.load_state_dict(torch.load("diffusion_model.pth", map_location=DEVICE))
model.eval()

beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

def sample_diffusion(model, condition_trajectory, n_samples):
    model.eval()
    with torch.no_grad():
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        # Random noise de taille (N, 5)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE) # <--- MODIFICATION

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

# Animation
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

ax_hist = fig.add_subplot(gs[0])
ax_hist.set_title("Predicted Weight Distributions")
ax_hist.set_xlim(-0.2, 1.2)
ax_hist.set_ylim(0, 50)

ax_traj = fig.add_subplot(gs[1])
# ... (Partie trajectoire identique) ...
ax_traj.set_title("Trajectory (Expanding Window)")
ax_traj.plot(traj_sample[:, 0], label="q1 (Full)", color='lightgray', linestyle='--')
ax_traj.plot(traj_sample[:, 1], label="q2 (Full)", color='lightgray', linestyle='--')
line_q1, = ax_traj.plot([], [], label="q1 (Observed)", color='blue')
line_q2, = ax_traj.plot([], [], label="q2 (Observed)", color='orange')
ax_traj.legend()

def update(length):
    sub_traj = traj_sample[:length]
    padded_traj = np.zeros((MAX_LEN, 2))
    padded_traj[:length] = sub_traj
    mask = np.zeros((MAX_LEN, 1))
    mask[:length] = 1
    
    combined = np.concatenate([padded_traj, mask], axis=1)
    traj_tensor = torch.FloatTensor(combined).unsqueeze(0).transpose(1, 2).to(DEVICE)

    generated_w_normalized = sample_diffusion(model, traj_tensor, N_SAMPLES)
    generated_w = scaler_w.inverse_transform(generated_w_normalized.cpu().numpy())

    ax_hist.clear()
    ax_hist.set_title(f"Predicted Weight Distributions (Window Length: {length})")
    ax_hist.set_xlim(-0.2, 1.2)
    ax_hist.set_ylim(0, 50)
    
    # Palette de couleurs étendue pour 5 dimensions
    colors = ['red', 'green', 'blue', 'purple', 'orange'] # <--- AJOUT DE COULEURS
    labels = [f'w{i+1}' for i in range(W_DIM)]
    
    # Boucle sur W_DIM au lieu de 3
    for i in range(W_DIM): # <--- MODIFICATION
        ax_hist.hist(generated_w[:, i], bins=15, alpha=0.5, color=colors[i], label=f'{labels[i]} Pred', density=True)
        ax_hist.axvline(w_unseen[i], color=colors[i], linestyle='dashed', linewidth=2, label=f'{labels[i]} True')
    
    ax_hist.legend(loc='upper right')

    line_q1.set_data(np.arange(length), traj_sample[:length, 0])
    line_q2.set_data(np.arange(length), traj_sample[:length, 1])
    
    return ax_hist, line_q1, line_q2

frames = np.arange(MIN_LEN, MAX_LEN + 1)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

save_path = "unseen_trajectory_5dim.gif"
print(f"Saving animation to {save_path}...")
ani.save(save_path, writer='pillow', fps=5)
print("Done!")
