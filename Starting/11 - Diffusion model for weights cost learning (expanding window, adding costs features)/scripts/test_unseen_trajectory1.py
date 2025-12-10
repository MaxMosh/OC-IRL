import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from tools.diffusion_model1 import ConditionalDiffusionModel
from tools.OCP_solving_cpin import solve_DOC

# Parameters for the inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES = 50
MAX_LEN = 50
# MIN_LEN = 5
MIN_LEN = 5

# Generation of an unseen trajectory
print("Generating unseen trajectory...")
# Generate random w from a uniforme on [0, 1]^3
w_unseen = np.random.rand(5)
# Then making the sum equal to 1
# w_unseen = w_unseen / np.sum(w_unseen)
# w_unseen = np.array([0.38299481, 0.33713702, 0.27986817])
print(f"Generated w: {w_unseen}")

# Solve OCP for the test weights
try:
    # q_init and x_fin are defaults from generate_OCP_solutions_w.py
    results_angles, results_velocities = solve_DOC(w_unseen, x_fin=-1.0, q_init=np.array([0, np.pi/4]))
    traj_sample = results_angles # (50, 2)
except Exception as e:
    print(f"Error solving OCP: {e}")
    exit()

# Load model & scaler from the training phase
print("Loading model...")
with open('scaler_w.pkl', 'rb') as f:
    scaler_w = pickle.load(f)

model = ConditionalDiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("diffusion_model_epochs_10000_min_len_40.pth", map_location=DEVICE))
model.eval()

# Diffusion constants
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

def sample_diffusion(model, condition_trajectory, n_samples):
    model.eval()
    with torch.no_grad():
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, 5).to(DEVICE)

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

# Subplot 1: Histograms
ax_hist = fig.add_subplot(gs[0])
ax_hist.set_title("Predicted Weight Distributions (Unseen Trajectory)")
ax_hist.set_xlim(-0.2, 1.2)
ax_hist.set_ylim(0, 50)   # AJOUT

# Subplot 2: Associated trajectory
ax_traj = fig.add_subplot(gs[1])
ax_traj.set_title("Trajectory (Expanding Window)")
ax_traj.plot(traj_sample[:, 0], label="q1 (Full)", color='lightgray', linestyle='--')
ax_traj.plot(traj_sample[:, 1], label="q2 (Full)", color='lightgray', linestyle='--')
line_q1, = ax_traj.plot([], [], label="q1 (Observed)", color='blue')
line_q2, = ax_traj.plot([], [], label="q2 (Observed)", color='orange')
ax_traj.legend()

def update(length):
    # Trajectory to put as a condition in the model inference
    sub_traj = traj_sample[:length]
    padded_traj = np.zeros((MAX_LEN, 2))
    padded_traj[:length] = sub_traj
    mask = np.zeros((MAX_LEN, 1))
    mask[:length] = 1
    
    combined = np.concatenate([padded_traj, mask], axis=1) # (50, 3)
    traj_tensor = torch.FloatTensor(combined).unsqueeze(0).transpose(1, 2).to(DEVICE)

    # Inference
    generated_w_normalized = sample_diffusion(model, traj_tensor, N_SAMPLES)
    generated_w = scaler_w.inverse_transform(generated_w_normalized.cpu().numpy())

    # Update plots
    ax_hist.clear()
    ax_hist.set_title(f"Predicted Weight Distributions (Window Length: {length})")
    ax_hist.set_xlim(-0.2, 1.2)
    ax_hist.set_ylim(0, 50)                         # AJOUT
    
    colors = ['red', 'green', 'blue']
    labels = ['w1', 'w2', 'w3', 'w4', 'w5']
    
    for i in range(5):
        ax_hist.hist(generated_w[:, i], bins=15, alpha=0.5, color=colors[i], label=f'{labels[i]} Pred', density=True)
        ax_hist.axvline(w_unseen[i], color=colors[i], linestyle='dashed', linewidth=2, label=f'{labels[i]} True')
    
    ax_hist.legend(loc='upper right')

    # Update trajectory
    line_q1.set_data(np.arange(length), traj_sample[:length, 0])
    line_q2.set_data(np.arange(length), traj_sample[:length, 1])
    
    return ax_hist, line_q1, line_q2

# Create animation
frames = np.arange(MIN_LEN, MAX_LEN + 1)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

# Save the gif animation
save_path = "unseen_trajectory.gif"
print(f"Saving animation to {save_path}...")
ani.save(save_path, writer='pillow', fps=5)
print("Done!")
