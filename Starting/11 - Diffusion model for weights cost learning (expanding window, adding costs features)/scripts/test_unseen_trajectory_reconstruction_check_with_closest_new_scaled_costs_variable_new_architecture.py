import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import json
import joblib # Pour charger les scalers

sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC, compute_scaling_factors

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 10  
W_DIM = 15 
INPUT_CHANNELS = 4
MODEL_PATH = "checkpoints/diff_model_transformer_final.pth"
SCALER_W_PATH = "checkpoints/scaler_w.pkl"
SCALER_TRAJ_PATH = "checkpoints/scaler_traj.pkl"

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
        raise FileNotFoundError("Scalers not found in checkpoints/")
    print("Loading scalers...")
    return joblib.load(SCALER_TRAJ_PATH), joblib.load(SCALER_W_PATH)

def get_scaling_factors():
    json_path = 'data/scale_factors_random.json' 
    if os.path.exists(json_path):
        with open(json_path, 'r') as f: return json.load(f)
    else:
        return compute_scaling_factors(num_samples=5, x_fin=1.9, q_init=[-np.pi/2, np.pi/2])

def generate_random_sample(scale_factors):
    print("Generating a new random trajectory (Ground Truth)...")
    while True:
        N_steps = np.random.randint(80, 201) 
        base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
        q_init_rad = np.deg2rad(np.array(Q_INIT_BASES_DEG[base_idx]) + np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2))
        x_fin = X_FIN_BASE + np.random.normal(0, X_FIN_NOISE_STD)
        w_matrix = np.random.rand(5, 3); w_matrix /= w_matrix.sum(axis=0) 
        
        try:
            q_true, dq_true = solve_DOC(w_matrix, N_steps, x_fin, q_init_rad, scale_factors, verbose=False)
            if q_true is not None:
                return {"q": q_true, "dq": dq_true, "w": w_matrix, "params": {"x_fin": x_fin, "q_init": q_init_rad, "N": N_steps}}
        except: print(".", end="", flush=True)

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
            
            alpha_t = alpha[i]; alpha_hat_t = alpha_hat[i]; beta_t = beta[i]
            noise = torch.randn_like(w_current) if i > 0 else torch.zeros_like(w_current)
            w_current = (1 / torch.sqrt(alpha_t)) * (w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

    # --- INVERSE TRANSFORM THE OUTPUT ---
    w_pred_numpy = w_current.cpu().numpy()
    w_pred_unscaled = scaler_w.inverse_transform(w_pred_numpy)
    
    # Clip to ensure valid weights (approx 0 to 1) 
    # Optional but helps OCP solver stability
    w_pred_unscaled = np.clip(w_pred_unscaled, 0.0, 1.0)
    
    # Re-normalize to sum to 1? OCP does implicit balancing but good to keep sum=1
    # Note: Flattened structure is (15,). We need to reshape (5,3) and normalize columns.
    
    return w_pred_unscaled

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found."); return

    scaler_traj, scaler_w = load_scalers()

    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded.")

    scale_factors_ocp = get_scaling_factors()
    sample = generate_random_sample(scale_factors_ocp)
    
    q_true = sample["q"]; dq_true = sample["dq"]; w_true = sample["w"]; params = sample["params"]       
    total_len = q_true.shape[0]
    idx_p1 = int(total_len / 3); idx_p2 = int(2 * total_len / 3)
    
    print(f"Total Length: {total_len}")

    # Layout Setup
    fig = plt.figure(figsize=(18, 12))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.3)
    gs_top = gs_main[0].subgridspec(1, 2, wspace=0.15)
    ax_q1 = fig.add_subplot(gs_top[0]); ax_q2 = fig.add_subplot(gs_top[1])
    gs_weights = gs_main[1].subgridspec(3, 6, width_ratios=[0.2, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.6)
    
    axes_weights = []; axes_progress = []
    cost_names = ["dq1", "dq2", "V_ee", "Torque", "Energy"]
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    for r in range(3):
        axes_progress.append(fig.add_subplot(gs_weights[r, 0]))
        row_axes = [fig.add_subplot(gs_weights[r, c+1]) for c in range(5)]
        axes_weights.append(row_axes)

    last_reconstructed_q = np.zeros_like(q_true)
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
        
        # --- INFERENCE ---
        w_pred_samples = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w)
        w_pred_mean_flat = w_pred_samples.mean(axis=0) # Average samples
        
        # Reshape & Normalize columns for OCP
        w_pred_matrix = w_pred_mean_flat.reshape(5, 3)
        # Avoid division by zero
        sums = w_pred_matrix.sum(axis=0)
        sums[sums == 0] = 1.0
        w_pred_matrix = w_pred_matrix / sums
        
        # --- RECONSTRUCTION ---
        nonlocal last_reconstructed_q
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            try:
                rec_q, _ = solve_DOC(w_pred_matrix, total_len, params["x_fin"], params["q_init"], scale_factors_ocp, verbose=False)
                if rec_q is not None: last_reconstructed_q = rec_q
            except: pass

        # --- PLOTTING (Same as before) ---
        time_steps = np.arange(total_len)
        q1_lims = (-150, 120); q2_lims = (-20, 180)

        for ax, q_idx, title, lims in zip([ax_q1, ax_q2], [0, 1], ["q1", "q2"], [q1_lims, q2_lims]):
            ax.clear()
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=1)
            
            if current_len <= idx_p1: ax.axvspan(0, idx_p1, color='green', alpha=0.05)
            elif current_len <= idx_p2: ax.axvspan(idx_p1, idx_p2, color='green', alpha=0.05)
            else: ax.axvspan(idx_p2, total_len, color='green', alpha=0.05)

            ax.plot(time_steps, np.degrees(q_true[:, q_idx]), 'k--', alpha=0.5, label='True')
            ax.plot(np.arange(current_len), np.degrees(q_true[:current_len, q_idx]), 'g-', linewidth=3, label='Observed')
            if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
                ax.plot(time_steps, np.degrees(last_reconstructed_q[:, q_idx]), 'r-', alpha=0.8, label='Pred')
            
            ax.set_title(f"Joint {title}"); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
            if q_idx == 0: ax.legend(loc='upper right', fontsize='x-small')

        # Weights Plotting
        prog = [min(1.0, current_len/idx_p1), 
                min(1.0, (current_len-idx_p1)/(idx_p2-idx_p1)) if current_len > idx_p1 else 0,
                min(1.0, (current_len-idx_p2)/(total_len-idx_p2)) if current_len > idx_p2 else 0]

        for r in range(3):
            ax_p = axes_progress[r]; ax_p.clear()
            p_val = prog[r]
            ax_p.barh([0], [p_val], color='limegreen' if p_val==1 else ('lightgray' if p_val==0 else 'green'), height=0.5)
            ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.5, 0.5); ax_p.axis('off')
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=10, fontweight='bold', transform=ax_p.transData)
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            for c in range(5): 
                ax = axes_weights[r][c]; ax.clear()
                val_true = w_true[c, r]; val_pred = w_pred_matrix[c, r]
                colors = ['black', 'orange'] if prog[r] > 0 else ['black', 'lightgray']
                ax.bar([0, 1], [val_true, val_pred], color=colors, alpha=0.7)
                ax.set_ylim(0, 1.0); ax.set_xticks([])
                if r == 0: ax.set_title(cost_names[c], fontsize=9, fontweight='bold')
                ax.text(0, val_true + 0.05, f"{val_true:.2f}", ha='center', fontsize=7, color='black')
                if prog[r] > 0: ax.text(1, val_pred + 0.05, f"{val_pred:.2f}", ha='center', fontsize=7, color='darkorange')

    frames = range(0, total_len - 15, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    anim.save("random_unseen_scaled_test.gif", writer='pillow', fps=5)
    print(f"Animation saved as: random_unseen_scaled_test.gif")

if __name__ == "__main__":
    main()