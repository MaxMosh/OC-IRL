import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Backend for server/headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
import sys

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 10  # Reduced for speed in loop
W_DIM = 15 # (5*3 flattened)
INPUT_CHANNELS = 4
MODEL_PATH = "checkpoints/diff_model_final.pth"
DATA_PATH = "data/" # Directory to find pkl

# OCP Parameters
X_FIN_BASE = 1.9 # Used for reconstruction target

def load_dataset():
    # Find the pkl file
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pkl") and "dataset" in f]
    if not files:
        raise FileNotFoundError("No dataset found.")
    path = os.path.join(DATA_PATH, files[-1])
    print(f"Loading data from {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def sample_diffusion(model, condition_trajectory, n_samples):
    """
    Standard diffusion sampling loop.
    """
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    with torch.no_grad():
        # condition_trajectory is (1, 4, len)
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
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

    return w_current.cpu().numpy()

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Using dummy model for code check.")
        # Ensure directory exists if we just want to test logic without trained model
        # return

    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    except:
        print("Warning: Could not load weights, using initialized model.")

    # 2. Get a Test Sample
    data = load_dataset()
    idx = np.random.randint(0, len(data["q_trajs"]))
    
    q_true = data["q_trajs"][idx]      # (N, 2)
    dq_true = data["dq_trajs"][idx]    # (N, 2)
    w_true = data["w_matrices"][idx]   # (5, 3)
    params = data["params"][idx]       # {x_fin, q_init, ...}
    
    total_len = q_true.shape[0]
    print(f"Test sample index: {idx}, Length: {total_len}")

    # Prepare Animation
    fig, (ax_traj, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Store predictions history
    history_w_pred = []
    
    # We will reconstruct the trajectory periodically
    # RECONSTRUCTION_STEP = 5 means we solve OCP every 5 frames
    RECONSTRUCTION_STEP = 10 
    last_reconstructed_q = np.zeros_like(q_true)

    def update(frame):
        current_len = frame + 10 # Start with at least 10 frames
        if current_len > total_len: current_len = total_len
        
        # 1. Prepare Input (Partial Trajectory)
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        
        combined = np.concatenate([q_partial, dq_partial], axis=1) # (len, 4)
        traj_tensor = torch.FloatTensor(combined).transpose(0, 1).unsqueeze(0).to(DEVICE) # (1, 4, len)
        
        # 2. Predict Weights
        # (N_SAMPLES, 15)
        w_pred_samples = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION)
        w_pred_mean_flat = w_pred_samples.mean(axis=0)
        
        # Reshape to (5, 3)
        w_pred_matrix = w_pred_mean_flat.reshape(5, 3)
        
        # 3. Reconstruct Trajectory (Heavy Compute)
        # Only do this every N frames or at the end
        nonlocal last_reconstructed_q
        
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            print(f"Reconstructing at frame {frame}/{total_len}...")
            try:
                # Use the predicted parameters (w) and known boundary conditions
                # We assume x_fin and q_init are known or we use the ones from the dataset
                # Note: In a real blind scenario, q_init is known (start of observation), x_fin might be unknown.
                # Here we assume x_fin is part of the task definition.
                rec_q, rec_dq = solve_DOC(
                    w_matrix=w_pred_matrix, 
                    N_steps=total_len, # Reconstruct full duration
                    x_fin=params["x_fin"], 
                    q_init=params["q_init"],
                    verbose=False
                )
                if rec_q is not None:
                    last_reconstructed_q = rec_q
            except Exception as e:
                print(f"OCP Failed: {e}")

        # --- PLOTTING ---
        ax_traj.clear()
        ax_w.clear()
        
        # Plot Trajectories (q1 only for clarity, or q1/q2)
        time_steps = np.arange(total_len)
        
        # Ground Truth
        ax_traj.plot(time_steps, np.degrees(q_true[:, 0]), 'k--', label='True q1')
        ax_traj.plot(time_steps, np.degrees(q_true[:, 1]), 'k:', label='True q2')
        
        # Observation so far
        ax_traj.plot(np.arange(current_len), np.degrees(q_true[:current_len, 0]), 'g-', linewidth=3, label='Observed q1')
        
        # Reconstruction (Predicted Full Motion)
        if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
            ax_traj.plot(time_steps, np.degrees(last_reconstructed_q[:, 0]), 'r-', alpha=0.7, label='Pred q1')
            ax_traj.plot(time_steps, np.degrees(last_reconstructed_q[:, 1]), 'r--', alpha=0.7, label='Pred q2')
        
        ax_traj.legend(loc='upper right')
        ax_traj.set_title(f"Trajectory Reconstruction (Obs: {current_len}/{total_len})")
        ax_traj.set_ylim(-150, 180)
        ax_traj.grid(True)
        
        # Plot Weights (Matrix 5x3)
        # We plot them as 3 groups of bars
        width = 0.35
        x_indices = np.arange(5)
        
        # Plot True Weights (Phase 1, 2, 3 aggregated or just one phase? Let's plot mean for clarity or all 3)
        # Let's plot the 3 phases side-by-side for each cost component
        
        # Simple Visualization: Flattened comparison or Phase-wise
        # Let's do Phase 1 (Start), Phase 2 (Mid), Phase 3 (End)
        # True
        ax_w.plot(x_indices, w_true[:, 0], 'ko-', alpha=0.3, label='True (Phase 1)')
        ax_w.plot(x_indices, w_true[:, 1], 'ks-', alpha=0.6, label='True (Phase 2)')
        ax_w.plot(x_indices, w_true[:, 2], 'kd-', alpha=1.0, label='True (Phase 3)')
        
        # Pred
        ax_w.plot(x_indices, w_pred_matrix[:, 0], 'ro-', alpha=0.3, label='Pred (Phase 1)')
        ax_w.plot(x_indices, w_pred_matrix[:, 1], 'rs-', alpha=0.6, label='Pred (Phase 2)')
        ax_w.plot(x_indices, w_pred_matrix[:, 2], 'rd-', alpha=1.0, label='Pred (Phase 3)')
        
        ax_w.set_title("Weights Prediction (3 Phases)")
        ax_w.legend(fontsize='x-small')
        ax_w.set_ylim(0, 1.0)
        ax_w.grid(True)

    # Generate Animation
    # Frames: from 0 to total_len-10
    frames = range(0, total_len - 10, 2) # Step of 2 to keep GIF size reasonable
    
    anim = FuncAnimation(fig, update, frames=frames, interval=200)
    
    gif_path = "progressive_prediction.gif"
    anim.save(gif_path, writer='pillow', fps=5)
    print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    main()