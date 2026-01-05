import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import json
import joblib 

sys.path.append(os.getcwd())
# Ensure this matches your file structure
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC, compute_scaling_factors

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 10  # We generate 10 samples per frame to visualize distribution
W_DIM = 15 
INPUT_CHANNELS = 4

# Paths to the NO SCALING checkpoints
MODEL_PATH = "checkpoints_no_scaling/diff_model_transformer_epoch_500.pth"
SCALER_W_PATH = "checkpoints_no_scaling/scaler_w.pkl"
SCALER_TRAJ_PATH = "checkpoints_no_scaling/scaler_traj.pkl"

# Model Architecture
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# Generation Params
Q_INIT_BASES_DEG = [[-90, 90], [-15, 105], [-115, 115]]
Q_INIT_NOISE_STD_DEG = 7.0
X_FIN_BASE = 1.9
X_FIN_NOISE_STD = 0.01

def load_scalers():
    if not os.path.exists(SCALER_W_PATH) or not os.path.exists(SCALER_TRAJ_PATH):
        raise FileNotFoundError("Scalers not found. Did you run the training script?")
    print("Loading scalers...")
    return joblib.load(SCALER_TRAJ_PATH), joblib.load(SCALER_W_PATH)

def generate_random_sample():
    """
    Generates a new random trajectory using the OCP solver WITHOUT scaling factors.
    """
    print("Generating a new random trajectory (Ground Truth)...")
    while True:
        N_steps = np.random.randint(80, 201) 
        base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
        q_init_rad = np.deg2rad(np.array(Q_INIT_BASES_DEG[base_idx]) + np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2))
        x_fin = X_FIN_BASE + np.random.normal(0, X_FIN_NOISE_STD)
        
        # Random weights (Normalized sum=1, applied to raw costs)
        w_matrix = np.random.rand(5, 3)
        w_matrix /= w_matrix.sum(axis=0) 
        
        try:
            # Calling solve_DOC without scale_factors
            q_true, dq_true = solve_DOC(w_matrix, N_steps, x_fin, q_init_rad, verbose=False)
            
            if q_true is not None:
                print(f"Success! Generated trajectory of length {N_steps}.")
                return {
                    "q": q_true, 
                    "dq": dq_true, 
                    "w": w_matrix, 
                    "params": {"x_fin": x_fin, "q_init": q_init_rad, "N": N_steps}
                }
        except: 
            print(".", end="", flush=True)

def sample_diffusion(model, condition_trajectory, n_samples, scaler_w):
    """
    Returns 'n_samples' predictions for the weights.
    Output shape: (n_samples, 15) -> unscaled numpy array
    """
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    seq_len = condition_trajectory.shape[2]
    # Dummy mask (all False = no padding)
    mask = torch.zeros((n_samples, seq_len), dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        # Repeat the single observed trajectory N times to generate N different weight predictions
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            # Pass mask to model
            predicted_noise = model(w_current, t, cond_repeated, trajectory_mask=mask)
            
            alpha_t = alpha[i]; alpha_hat_t = alpha_hat[i]; beta_t = beta[i]
            noise = torch.randn_like(w_current) if i > 0 else torch.zeros_like(w_current)
            w_current = (1 / torch.sqrt(alpha_t)) * (w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    # --- INVERSE TRANSFORM (StandardScaler) ---
    w_pred_numpy = w_current.cpu().numpy()
    w_pred_unscaled = scaler_w.inverse_transform(w_pred_numpy)
    
    # Clip to valid range [0, 1]
    w_pred_unscaled = np.clip(w_pred_unscaled, 0.0, 1.0)
    
    return w_pred_unscaled

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found."); return

    scaler_traj, scaler_w = load_scalers()

    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded.")

    # Generate sample (No Scaling Factors)
    sample = generate_random_sample()
    
    q_true = sample["q"]; dq_true = sample["dq"]; w_true = sample["w"]; params = sample["params"]       
    total_len = q_true.shape[0]
    idx_p1 = int(total_len / 3); idx_p2 = int(2 * total_len / 3)
    
    print(f"Total Length: {total_len}")

    # --- LAYOUT SETUP ---
    fig = plt.figure(figsize=(18, 14))
    
    # Split: Top part (Trajectory 2x2) and Bottom part (Weights)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
    
    # Top Grid: 2 rows x 2 columns for q1, q2, dq1, dq2
    gs_top = gs_main[0].subgridspec(2, 2, wspace=0.2, hspace=0.3)
    ax_q1 = fig.add_subplot(gs_top[0, 0])
    ax_q2 = fig.add_subplot(gs_top[0, 1])
    ax_dq1 = fig.add_subplot(gs_top[1, 0])
    ax_dq2 = fig.add_subplot(gs_top[1, 1])
    
    # Bottom Grid: Weights (3 rows for phases, 6 cols for progress + 5 costs)
    gs_weights = gs_main[1].subgridspec(3, 6, width_ratios=[0.2, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.6)
    
    axes_weights = []; axes_progress = []
    cost_names = ["dq1", "dq2", "V_ee", "Torque", "Energy"]
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(3):
        axes_progress.append(fig.add_subplot(gs_weights[r, 0]))
        row_axes = [fig.add_subplot(gs_weights[r, c+1]) for c in range(5)]
        axes_weights.append(row_axes)

    last_reconstructed_q = np.zeros_like(q_true)
    last_reconstructed_dq = np.zeros_like(dq_true)
    RECONSTRUCTION_STEP = 10 

    def update(frame):
        current_len = frame + 15 
        if current_len > total_len: current_len = total_len
        
        # --- PREPARE & SCALE INPUT ---
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) # (L, 4)
        
        # Apply Trajectory Scaler
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # --- INFERENCE (Multiple Samples) ---
        # Get N_SAMPLES_DIFFUSION predictions
        w_pred_samples_flat = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w) # (10, 15)
        
        # Reshape to (10, 5, 3)
        w_pred_samples = w_pred_samples_flat.reshape(N_SAMPLES_DIFFUSION, 5, 3)
        
        # Normalize each sample so that columns sum to 1
        sums = w_pred_samples.sum(axis=1, keepdims=True) # (10, 1, 3)
        sums[sums == 0] = 1.0
        w_pred_samples = w_pred_samples / sums
        
        # Compute Mean for OCP Reconstruction
        w_pred_mean = w_pred_samples.mean(axis=0) # (5, 3)
        
        # --- RECONSTRUCTION (Using Mean Weights) ---
        nonlocal last_reconstructed_q, last_reconstructed_dq
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            try:
                # OCP Solving (NO Scale Factors)
                rec_q, rec_dq = solve_DOC(w_pred_mean, total_len, params["x_fin"], params["q_init"], verbose=False)
                if rec_q is not None: 
                    last_reconstructed_q = rec_q
                    last_reconstructed_dq = rec_dq
            except: pass

        # --- PLOTTING TRAJECTORIES ---
        time_steps = np.arange(total_len)
        
        # Define limits and data for all 4 plots
        # q1: deg, q2: deg, dq1: deg/s, dq2: deg/s
        plots_config = [
            (ax_q1, q_true[:,0], last_reconstructed_q[:,0], "q1 (deg)", (-150, 120)),
            (ax_q2, q_true[:,1], last_reconstructed_q[:,1], "q2 (deg)", (-20, 180)),
            (ax_dq1, dq_true[:,0], last_reconstructed_dq[:,0], "dq1 (deg/s)", (-500, 500)),
            (ax_dq2, dq_true[:,1], last_reconstructed_dq[:,1], "dq2 (deg/s)", (-500, 500))
        ]

        for ax, data_true, data_pred, title, lims in plots_config:
            ax.clear()
            # Phase lines
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=1)
            
            # Active phase shading
            if current_len <= idx_p1: ax.axvspan(0, idx_p1, color='green', alpha=0.05)
            elif current_len <= idx_p2: ax.axvspan(idx_p1, idx_p2, color='green', alpha=0.05)
            else: ax.axvspan(idx_p2, total_len, color='green', alpha=0.05)

            # Data
            # Convert to degrees for plotting
            ax.plot(time_steps, np.degrees(data_true), 'k--', alpha=0.5, label='True')
            ax.plot(np.arange(current_len), np.degrees(data_true[:current_len]), 'g-', linewidth=3, label='Observed')
            
            if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
                ax.plot(time_steps, np.degrees(data_pred), 'r-', alpha=0.8, label='Pred (Mean)')
            
            ax.set_title(title)
            ax.set_ylim(lims)
            ax.set_xlabel("Time steps")
            ax.set_ylabel(title.split()[-1]) # Extracts "(deg)" or "(deg/s)"
            ax.grid(True, alpha=0.3)
            if title.startswith("q1"): ax.legend(loc='upper right', fontsize='x-small')

        # --- PLOTTING WEIGHTS ---
        # Calculate progress
        prog = [min(1.0, current_len/idx_p1), 
                min(1.0, (current_len-idx_p1)/(idx_p2-idx_p1)) if current_len > idx_p1 else 0,
                min(1.0, (current_len-idx_p2)/(total_len-idx_p2)) if current_len > idx_p2 else 0]

        for r in range(3): # Phases
            # 1. Progress Bar
            ax_p = axes_progress[r]; ax_p.clear()
            p_val = prog[r]
            ax_p.barh([0], [p_val], color='limegreen' if p_val==1 else ('lightgray' if p_val==0 else 'green'), height=0.5)
            ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.5, 0.5); ax_p.axis('off')
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=10, fontweight='bold', transform=ax_p.transData)
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            # 2. Weights (Distributions)
            for c in range(5): # Costs
                ax = axes_weights[r][c]; ax.clear()
                
                val_true = w_true[c, r]
                val_pred_mean = w_pred_mean[c, r]
                
                # Get all 10 predictions for this specific weight
                val_preds_all = w_pred_samples[:, c, r] 
                
                # Colors
                colors = ['black', 'orange'] if prog[r] > 0 else ['black', 'lightgray']
                
                # A. Plot Bars (True vs Predicted Mean)
                ax.bar([0, 1], [val_true, val_pred_mean], color=colors, alpha=0.6)
                
                # B. Plot Distribution Points (Scatter over the predicted bar)
                if prog[r] > 0:
                    # Plot points at x=1 with slight jitter or just straight
                    # Visualizing the 10 samples
                    x_scatter = np.ones(N_SAMPLES_DIFFUSION)
                    ax.scatter(x_scatter, val_preds_all, color='blue', s=10, zorder=3, alpha=0.8, label='Samples')

                ax.set_ylim(0, 1.0); ax.set_xticks([])
                
                if r == 0: ax.set_title(cost_names[c], fontsize=9, fontweight='bold')
                
                # Text labels for mean
                ax.text(0, val_true + 0.05, f"{val_true:.2f}", ha='center', fontsize=7, color='black')
                if prog[r] > 0: 
                    ax.text(1, val_pred_mean + 0.05, f"{val_pred_mean:.2f}", ha='center', fontsize=7, color='darkorange')

    frames = range(0, total_len - 15, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    anim.save("random_unseen_NO_SCALING_distribution_test.gif", writer='pillow', fps=5)
    print(f"Animation saved as: random_unseen_NO_SCALING_distribution_test.gif")

if __name__ == "__main__":
    main()
