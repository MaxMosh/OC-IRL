import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import joblib
from scipy.signal import savgol_filter

sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 50 
W_DIM = 15 
INPUT_CHANNELS = 4

# Paths
MODEL_PATH = "checkpoints_no_scaling/diff_model_transformer_epoch_500.pth"
SCALER_W_PATH = "checkpoints_no_scaling/scaler_w.pkl"
SCALER_TRAJ_PATH = "checkpoints_no_scaling/scaler_traj.pkl"
EXPERIMENTAL_DATA_PATH = "../8 - Diffusion model with conditionning (Berret data)/data/S01/Trial45.csv"

# Model Architecture
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# Robot Constants (for Forward Kinematics x_fin)
L_1 = 1.0
L_2 = 1.0
FREQ = 100.0  # Assumed frequency for dq computation
DT = 1.0 / FREQ

def load_scalers():
    if not os.path.exists(SCALER_W_PATH) or not os.path.exists(SCALER_TRAJ_PATH):
        raise FileNotFoundError("Scalers not found. Run training first.")
    print("Loading scalers...")
    return joblib.load(SCALER_TRAJ_PATH), joblib.load(SCALER_W_PATH)

def process_experimental_data(path):
    print(f"Loading experimental data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load CSV (no header)
    df = pd.read_csv(path, header=None)
    raw_data = df.values # Shape likely (2, N) or (N, 2)
    
    # Check shape and transpose if necessary to get (N, 2)
    if raw_data.shape[0] == 2:
        raw_data = raw_data.T
    
    # 1. Filtering (Smoothing)
    # Using Savitzky-Golay filter to smooth positions before differentiation
    # Window length must be odd and <= len(x). Polyorder typically 2 or 3.
    window_length = min(15, raw_data.shape[0] - 1 if raw_data.shape[0] % 2 == 0 else raw_data.shape[0])
    if window_length % 2 == 0: window_length -= 1
    
    q_smooth = np.zeros_like(raw_data)
    for i in range(2):
        q_smooth[:, i] = savgol_filter(raw_data[:, i], window_length, 3)
        
    # 2. Compute Velocities (Numerical Differentiation)
    # dq = (q[t+1] - q[t-1]) / 2dt
    dq_smooth = np.gradient(q_smooth, DT, axis=0)
    
    # 3. Compute x_fin (Target for OCP)
    # Using Forward Kinematics on the last point
    q_end = q_smooth[-1]
    x_fin = L_1 * np.cos(q_end[0]) + L_2 * np.cos(q_end[0] + q_end[1])
    
    q_init = q_smooth[0]
    total_len = q_smooth.shape[0]
    
    print(f"Data processed. Length: {total_len}, x_fin computed: {x_fin:.4f}")
    
    return q_smooth, dq_smooth, x_fin, q_init, total_len

def sample_diffusion(model, condition_trajectory, n_samples, scaler_w):
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    seq_len = condition_trajectory.shape[2]
    mask = torch.zeros((n_samples, seq_len), dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            predicted_noise = model(w_current, t, cond_repeated, trajectory_mask=mask)
            
            alpha_t = alpha[i]
            alpha_hat_t = alpha_hat[i]
            beta_t = beta[i]
            noise = torch.randn_like(w_current) if i > 0 else torch.zeros_like(w_current)
            w_current = (1 / torch.sqrt(alpha_t)) * (w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    w_pred_numpy = w_current.cpu().numpy()
    w_pred_unscaled = scaler_w.inverse_transform(w_pred_numpy)
    w_pred_unscaled = np.clip(w_pred_unscaled, 0.0, 1.0)
    
    return w_pred_unscaled

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found."); return

    scaler_traj, scaler_w = load_scalers()

    # Load Model
    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Process Data
    try:
        q_ref, dq_ref, x_fin, q_init, total_len = process_experimental_data(EXPERIMENTAL_DATA_PATH)
    except Exception as e:
        print(f"Could not process data: {e}")
        return

    idx_p1 = int(total_len / 3)
    idx_p2 = int(2 * total_len / 3)

    # --- GRAPHICAL LAYOUT ---
    fig = plt.figure(figsize=(18, 14))
    
    # Top: Trajectories
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    gs_top = gs_main[0].subgridspec(2, 2, wspace=0.2, hspace=0.3)
    ax_q1 = fig.add_subplot(gs_top[0, 0])
    ax_q2 = fig.add_subplot(gs_top[0, 1])
    ax_dq1 = fig.add_subplot(gs_top[1, 0])
    ax_dq2 = fig.add_subplot(gs_top[1, 1])
    
    # Bottom: Weights
    gs_weights = gs_main[1].subgridspec(3, 6, width_ratios=[0.2, 1, 1, 1, 1, 1], wspace=0.3, hspace=0.6)
    
    axes_weights = []
    axes_progress = []
    cost_names = ["dq1", "dq2", "V_ee", "Torque", "Energy"]
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(3):
        axes_progress.append(fig.add_subplot(gs_weights[r, 0]))
        row_axes = [fig.add_subplot(gs_weights[r, c+1]) for c in range(5)]
        axes_weights.append(row_axes)

    last_reconstructed_q = np.zeros_like(q_ref)
    last_reconstructed_dq = np.zeros_like(dq_ref)
    RECONSTRUCTION_STEP = 10 

    def update(frame):
        # Allow frame to go slightly beyond if needed, or stick to len
        current_len = frame + 15 
        if current_len > total_len: current_len = total_len
        
        # Prepare Input
        q_partial = q_ref[:current_len]
        dq_partial = dq_ref[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) # (L, 4)
        
        # Scale
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # Inference (50 samples)
        w_pred_samples_flat = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w)
        w_pred_samples = w_pred_samples_flat.reshape(N_SAMPLES_DIFFUSION, 5, 3)
        
        # Normalize sum=1
        sums = w_pred_samples.sum(axis=1, keepdims=True) 
        sums[sums == 0] = 1.0
        w_pred_samples = w_pred_samples / sums
        
        # Mean for Reconstruction
        w_pred_mean = w_pred_samples.mean(axis=0) 
        
        # Reconstruction (OCP)
        nonlocal last_reconstructed_q, last_reconstructed_dq
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            try:
                # Note: x_fin and q_init come from the experimental data processing
                rec_q, rec_dq = solve_DOC(w_pred_mean, total_len, x_fin, q_init, verbose=False)
                if rec_q is not None: 
                    last_reconstructed_q = rec_q
                    last_reconstructed_dq = rec_dq
            except: pass

        # --- PLOTTING TRAJECTORIES ---
        time_steps = np.arange(total_len)
        plots_config = [
            (ax_q1, q_ref[:,0], last_reconstructed_q[:,0], "q1 (deg)", (-150, 120)),
            (ax_q2, q_ref[:,1], last_reconstructed_q[:,1], "q2 (deg)", (-20, 180)),
            (ax_dq1, dq_ref[:,0], last_reconstructed_dq[:,0], "dq1 (deg/s)", (-500, 500)),
            (ax_dq2, dq_ref[:,1], last_reconstructed_dq[:,1], "dq2 (deg/s)", (-500, 500))
        ]

        for ax, data_ref, data_rec, title, lims in plots_config:
            ax.clear()
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=1)
            
            if current_len <= idx_p1: ax.axvspan(0, idx_p1, color='green', alpha=0.05)
            elif current_len <= idx_p2: ax.axvspan(idx_p1, idx_p2, color='green', alpha=0.05)
            else: ax.axvspan(idx_p2, total_len, color='green', alpha=0.05)

            # Plot Experimental Reference
            ax.plot(time_steps, np.degrees(data_ref), 'k--', alpha=0.6, label='Experiment')
            # Plot Observed Part
            ax.plot(np.arange(current_len), np.degrees(data_ref[:current_len]), 'g-', linewidth=3, label='Observed')
            
            # Plot Reconstruction
            if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
                ax.plot(time_steps, np.degrees(data_rec), 'r-', alpha=0.8, label='Reconstruction')
            
            ax.set_title(title)
            # Auto-scale Y or use fixed limits? Experimental data might go out of bounds.
            # Using fixed limits from previous script for consistency, but checking data range is better.
            ax.set_ylim(lims) 
            ax.set_xlabel("Time steps")
            ax.set_ylabel(title.split()[-1]) 
            ax.grid(True, alpha=0.3)
            if title.startswith("q1"): ax.legend(loc='upper right', fontsize='x-small')

        # --- PLOTTING WEIGHTS ---
        prog = [min(1.0, current_len/idx_p1), 
                min(1.0, (current_len-idx_p1)/(idx_p2-idx_p1)) if current_len > idx_p1 else 0,
                min(1.0, (current_len-idx_p2)/(total_len-idx_p2)) if current_len > idx_p2 else 0]

        for r in range(3): 
            # Progress Bar
            ax_p = axes_progress[r]; ax_p.clear()
            p_val = prog[r]
            ax_p.barh([0], [p_val], color='limegreen' if p_val==1 else ('lightgray' if p_val==0 else 'green'), height=0.5)
            ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.5, 0.5); ax_p.axis('off')
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=10, fontweight='bold', transform=ax_p.transData)
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            # Histograms (Predicted Distribution ONLY)
            for c in range(5): 
                ax = axes_weights[r][c]; ax.clear()
                
                # No Ground Truth for experimental data
                val_preds_all = w_pred_samples[:, c, r] 
                
                ax.set_xlim(0, 1.0) 
                ax.set_ylim(0, N_SAMPLES_DIFFUSION) # Fixed Y limit
                ax.tick_params(axis='y', labelsize=6)
                
                if r == 0: ax.set_title(cost_names[c], fontsize=9, fontweight='bold')
                if r == 2: ax.set_xlabel("Value")

                hist_color = 'orange'
                ax.hist(val_preds_all, bins=np.linspace(0, 1, 20), color=hist_color, alpha=0.5, edgecolor='darkorange')
                
                # We do NOT plot the vertical 'True' line here since we don't have it.

    frames = range(0, total_len - 10, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    
    gif_name = "experimental_berret_inference.gif"
    anim.save(gif_name, writer='pillow', fps=5)
    print(f"Animation saved as: {gif_name}")
    
    try:
        mp4_name = "experimental_berret_inference.mp4"
        anim.save(mp4_name, writer='ffmpeg', fps=5, extra_args=['-vcodec', 'libx264'])
        print(f"Animation saved as: {mp4_name}")
    except:
        pass

if __name__ == "__main__":
    main()