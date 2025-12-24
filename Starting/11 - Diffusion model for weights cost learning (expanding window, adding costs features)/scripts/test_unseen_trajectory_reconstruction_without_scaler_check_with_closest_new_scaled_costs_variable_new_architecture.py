import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Backend for server/headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import json

# ADDING CURRENT FOLDER TO THE PATH OF PACKAGES
sys.path.append(os.getcwd())
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC, compute_scaling_factors

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 10  
W_DIM = 15 
INPUT_CHANNELS = 4
MODEL_PATH = "checkpoints/diff_model_transformer_final.pth" # Updated path

# Model Architecture Params (Must match training)
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# Generation Parameters
Q_INIT_BASES_DEG = [
    [-90, 90],
    [-15, 105],
    [-115, 115]
]
Q_INIT_NOISE_STD_DEG = 7.0
X_FIN_BASE = 1.9
X_FIN_NOISE_STD = 0.01

def get_scaling_factors():
    """
    Loads scaling factors or computes them if missing.
    """
    json_path = 'data/scale_factors_random.json' 
    if os.path.exists(json_path):
        print(f"Loading scaling factors from {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        print("Scaling factors file not found. Computing them on the fly (quick calibration)...")
        return compute_scaling_factors(num_samples=5, x_fin=1.9, q_init=[-np.pi/2, np.pi/2])

def generate_random_sample(scale_factors):
    """
    Generates a completely new random trajectory (Unseen Data).
    """
    print("Generating a new random trajectory (Ground Truth)...")
    
    while True:
        N_steps = np.random.randint(80, 201) 
        
        base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
        q_base = np.array(Q_INIT_BASES_DEG[base_idx])
        noise_q = np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2)
        q_init_deg = q_base + noise_q
        q_init_rad = np.deg2rad(q_init_deg)
        
        noise_x = np.random.normal(0, X_FIN_NOISE_STD)
        x_fin = X_FIN_BASE + noise_x
        
        w_matrix = np.random.rand(5, 3)
        w_matrix = w_matrix / w_matrix.sum(axis=0) 
        
        try:
            q_true, dq_true = solve_DOC(
                w_matrix, 
                N_steps, 
                x_fin=x_fin, 
                q_init=q_init_rad, 
                scale_factors=scale_factors,
                verbose=False
            )
            
            if q_true is not None:
                print(f"Success! Generated trajectory of length {N_steps}.")
                return {
                    "q": q_true,
                    "dq": dq_true,
                    "w": w_matrix,
                    "params": {"x_fin": x_fin, "q_init": q_init_rad, "N": N_steps}
                }
        except Exception:
            print(".", end="", flush=True) 

def sample_diffusion(model, condition_trajectory, n_samples):
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    # Create Dummy Mask (All False = No Padding)
    # condition_trajectory is (1, 4, Length)
    seq_len = condition_trajectory.shape[2]
    mask = torch.zeros((n_samples, seq_len), dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            # Pass mask to model
            predicted_noise = model(w_current, t, cond_repeated, trajectory_mask=mask)

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
        print(f"Model {MODEL_PATH} not found.")
        # Proceeding strictly might fail, but let's allow instantiation to check code logic
        
    model = ConditionalDiffusionModel(
        w_dim=W_DIM, 
        input_channels=INPUT_CHANNELS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights ({e}).")

    # 2. Generate Random Sample
    scale_factors = get_scaling_factors()
    sample = generate_random_sample(scale_factors)
    
    q_true = sample["q"]      
    dq_true = sample["dq"]    
    w_true = sample["w"]   
    params = sample["params"]       
    
    total_len = q_true.shape[0]
    idx_p1 = int(total_len / 3)
    idx_p2 = int(2 * total_len / 3)
    
    print(f"Total Length: {total_len}")
    print(f"Phases: [0-{idx_p1}], [{idx_p1}-{idx_p2}], [{idx_p2}-{total_len}]")

    # --- GRAPHICAL LAYOUT ---
    fig = plt.figure(figsize=(18, 12))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.3)
    
    gs_top = gs_main[0].subgridspec(1, 2, wspace=0.15)
    ax_q1 = fig.add_subplot(gs_top[0])
    ax_q2 = fig.add_subplot(gs_top[1])
    
    gs_weights = gs_main[1].subgridspec(3, 6, width_ratios=[0.2, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.6)
    
    axes_weights = []
    axes_progress = [] 
    
    cost_names = ["dq1", "dq2", "V_ee", "Torque", "Energy"]
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(3): 
        ax_prog = fig.add_subplot(gs_weights[r, 0])
        axes_progress.append(ax_prog)
        row_axes = []
        for c in range(5): 
            ax = fig.add_subplot(gs_weights[r, c+1]) 
            row_axes.append(ax)
        axes_weights.append(row_axes)

    last_reconstructed_q = np.zeros_like(q_true)
    RECONSTRUCTION_STEP = 10 

    def update(frame):
        current_len = frame + 15 
        if current_len > total_len: current_len = total_len
        
        # Inference
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) 
        traj_tensor = torch.FloatTensor(combined).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        w_pred_samples = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION)
        w_pred_mean_flat = w_pred_samples.mean(axis=0)
        w_pred_matrix = w_pred_mean_flat.reshape(5, 3)
        
        # Reconstruction
        nonlocal last_reconstructed_q
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            try:
                rec_q, _ = solve_DOC(
                    w_matrix=w_pred_matrix, 
                    N_steps=total_len, 
                    x_fin=params["x_fin"], 
                    q_init=params["q_init"],
                    scale_factors=scale_factors, 
                    verbose=False
                )
                if rec_q is not None:
                    last_reconstructed_q = rec_q
            except Exception:
                pass

        # Plotting Trajectories
        time_steps = np.arange(total_len)
        q1_lims = (-150, 120)
        q2_lims = (-20, 180)

        for ax, q_idx, title, lims in zip([ax_q1, ax_q2], [0, 1], ["q1", "q2"], [q1_lims, q2_lims]):
            ax.clear()
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=1)
            
            if current_len <= idx_p1:
                ax.axvspan(0, idx_p1, color='green', alpha=0.05)
            elif current_len <= idx_p2:
                ax.axvspan(idx_p1, idx_p2, color='green', alpha=0.05)
            else:
                ax.axvspan(idx_p2, total_len, color='green', alpha=0.05)

            ax.plot(time_steps, np.degrees(q_true[:, q_idx]), 'k--', alpha=0.5, label='True')
            ax.plot(np.arange(current_len), np.degrees(q_true[:current_len, q_idx]), 'g-', linewidth=3, label='Observed')
            
            if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
                ax.plot(time_steps, np.degrees(last_reconstructed_q[:, q_idx]), 'r-', alpha=0.8, label='Pred')
            
            ax.set_title(f"Joint {title}")
            ax.set_ylim(lims)
            if q_idx == 0: ax.legend(loc='upper right', fontsize='x-small')
            ax.grid(True, alpha=0.3)
            
            y_txt = lims[0] + (lims[1]-lims[0])*0.05
            ax.text(idx_p1/2, y_txt, "Phase 1", ha='center', fontsize=8, color='gray')
            ax.text(idx_p1 + (idx_p2-idx_p1)/2, y_txt, "Phase 2", ha='center', fontsize=8, color='gray')
            ax.text(idx_p2 + (total_len-idx_p2)/2, y_txt, "Phase 3", ha='center', fontsize=8, color='gray')

        # Plotting Weights & Progress
        prog_p1 = min(1.0, current_len / idx_p1)
        prog_p2 = 0.0
        if current_len > idx_p1: prog_p2 = min(1.0, (current_len - idx_p1) / (idx_p2 - idx_p1))
        prog_p3 = 0.0
        if current_len > idx_p2: prog_p3 = min(1.0, (current_len - idx_p2) / (total_len - idx_p2))
        progress_values = [prog_p1, prog_p2, prog_p3]

        for r in range(3):
            # Progress
            ax_p = axes_progress[r]
            ax_p.clear()
            p_val = progress_values[r]
            color_bar = 'lightgray' if p_val == 0 else ('limegreen' if p_val == 1 else 'green')
            ax_p.barh([0], [p_val], color=color_bar, height=0.5)
            ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.5, 0.5); ax_p.axis('off')
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=10, fontweight='bold', transform=ax_p.transData)
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            # Weights
            for c in range(5): 
                ax = axes_weights[r][c]
                ax.clear()
                val_true = w_true[c, r]
                val_pred = w_pred_matrix[c, r]
                x_pos = [0, 1]
                colors = ['black', 'orange']
                if progress_values[r] == 0: colors = ['black', 'lightgray']
                
                ax.bar(x_pos, [val_true, val_pred], color=colors, alpha=0.7)
                ax.set_ylim(0, 1.0); ax.set_xticks([])
                if r == 0: ax.set_title(cost_names[c], fontsize=9, fontweight='bold')
                ax.text(0, val_true + 0.05, f"{val_true:.2f}", ha='center', fontsize=7, color='black')
                if progress_values[r] > 0:
                    ax.text(1, val_pred + 0.05, f"{val_pred:.2f}", ha='center', fontsize=7, color='darkorange')

    frames = range(0, total_len - 15, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    
    gif_name = "random_unseen_transformer_test.gif"
    anim.save(gif_name, writer='pillow', fps=5)
    print(f"Animation saved as: {gif_name}")

if __name__ == "__main__":
    main()