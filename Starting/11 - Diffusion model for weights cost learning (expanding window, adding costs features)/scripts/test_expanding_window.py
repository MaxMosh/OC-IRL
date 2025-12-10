import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
from tools.diffusion_model import ConditionalDiffusionModel

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES = 50 # Number of samples for distribution
MAX_LEN = 50
# MIN_LEN = 5
MIN_LEN = 2

# --- 1. Load Resources ---
print("Loading resources...")
try:
    traj_data = np.load("data/array_results_angles_101.npy")
    w_true_data = np.load("data/array_w_101.npy")
except FileNotFoundError:
    print("Error: Data files not found.")
    exit()

with open('scaler_w.pkl', 'rb') as f:
    scaler_w = pickle.load(f)

model = ConditionalDiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("diffusion_model_epochs_10000.pth", map_location=DEVICE))
model.eval()

# Diffusion constants
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
alpha_bar_prev = torch.cat([torch.tensor([1.]).to(DEVICE), alpha_hat[:-1]])

def sample_diffusion(model, condition_trajectory, n_samples):
    """
    Performs the reverse diffusion process: Noise -> Data
    condition_trajectory: Shape (1, 3, 50)
    """
    model.eval()
    with torch.no_grad():
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, 3).to(DEVICE)

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

# --- 2. Setup Animation ---
# Pick a random sample
test_idx = np.random.randint(0, len(traj_data))
traj_sample = traj_data[test_idx] # (50, 2)
w_truth = w_true_data[test_idx]   # (3,)

print(f"Animating sample {test_idx}")
print(f"Ground Truth w: {w_truth}")

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

# Subplot 1: Histograms
ax_hist = fig.add_subplot(gs[0])
ax_hist.set_title("Predicted Weight Distributions")
ax_hist.set_xlim(-0.5, 1.5) # Assuming normalized weights are roughly in this range after inverse transform? 
ax_hist.set_ylim(0, 50)   # AJOUT
# Actually w are in [0, 1] usually, but let's check scaler. 
# Better to set dynamic limits or fixed reasonable ones. 
# Let's assume [0, 1] for now as they are weights, but scaler might shift them.
# We will update limits dynamically or use wide ones.

# Subplot 2: Trajectory
ax_traj = fig.add_subplot(gs[1])
ax_traj.set_title("Trajectory (Expanding Window)")
ax_traj.plot(traj_sample[:, 0], label="q1 (Full)", color='lightgray', linestyle='--')
ax_traj.plot(traj_sample[:, 1], label="q2 (Full)", color='lightgray', linestyle='--')
line_q1, = ax_traj.plot([], [], label="q1 (Observed)", color='blue')
line_q2, = ax_traj.plot([], [], label="q2 (Observed)", color='orange')
ax_traj.legend()

def update(length):
    # 1. Prepare Input
    sub_traj = traj_sample[:length]
    padded_traj = np.zeros((MAX_LEN, 2))
    padded_traj[:length] = sub_traj
    mask = np.zeros((MAX_LEN, 1))
    mask[:length] = 1
    
    combined = np.concatenate([padded_traj, mask], axis=1) # (50, 3)
    traj_tensor = torch.FloatTensor(combined).unsqueeze(0).transpose(1, 2).to(DEVICE)

    # 2. Inference
    generated_w_normalized = sample_diffusion(model, traj_tensor, N_SAMPLES)
    generated_w = scaler_w.inverse_transform(generated_w_normalized.cpu().numpy())

    # 3. Update Plots
    ax_hist.clear()
    ax_hist.set_title(f"Predicted Weight Distributions (Window Length: {length})")
    
    colors = ['red', 'green', 'blue']
    labels = ['w1', 'w2', 'w3']
    
    for i in range(3):
        # Plot histogram/KDE
        # Using hist for simplicity
        ax_hist.hist(generated_w[:, i], bins=15, alpha=0.5, color=colors[i], label=f'{labels[i]} Pred', density=True)
        ax_hist.axvline(w_truth[i], color=colors[i], linestyle='dashed', linewidth=2, label=f'{labels[i]} True')
    
    ax_hist.legend(loc='upper right')
    ax_hist.set_xlim(-0.2, 1.2) # Fixed range for stability
    ax_hist.set_ylim(0, 50)   # AJOUT

    # Update Trajectory
    line_q1.set_data(np.arange(length), traj_sample[:length, 0])
    line_q2.set_data(np.arange(length), traj_sample[:length, 1])
    
    return ax_hist, line_q1, line_q2

# Create Animation
# Frames from MIN_LEN to MAX_LEN
frames = np.arange(MIN_LEN, MAX_LEN + 1)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

# Save
save_path = "expanding_window.gif"
print(f"Saving animation to {save_path}...")
ani.save(save_path, writer='pillow', fps=5)
print("Done!")
