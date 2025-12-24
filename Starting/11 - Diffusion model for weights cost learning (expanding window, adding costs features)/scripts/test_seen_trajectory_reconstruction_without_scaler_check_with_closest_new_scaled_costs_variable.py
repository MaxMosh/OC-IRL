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
N_SAMPLES_DIFFUSION = 10  
W_DIM = 15 
INPUT_CHANNELS = 4
MODEL_PATH = "checkpoints/diff_model_variable_epoch_3000.pth"
DATA_PATH = "data/" 

def load_dataset():
    # Find the last generated pkl file
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pkl") and "dataset" in f]
    if not files:
        raise FileNotFoundError("No dataset found.")
    path = os.path.join(DATA_PATH, sorted(files)[-1])
    print(f"Loading data from {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def sample_diffusion(model, condition_trajectory, n_samples):
    """
    Runs the reverse diffusion process to predict weights.
    """
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    with torch.no_grad():
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
        print(f"Model {MODEL_PATH} not found.")
        return

    model = ConditionalDiffusionModel(w_dim=W_DIM, input_channels=INPUT_CHANNELS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 2. Select Test Sample
    data = load_dataset()
    idx = np.random.randint(0, len(data["q_trajs"]))
    
    q_true = data["q_trajs"][idx]      
    dq_true = data["dq_trajs"][idx]    
    w_true = data["w_matrices"][idx]   
    params = data["params"][idx]       
    
    total_len = q_true.shape[0]
    
    # Define phase boundaries
    idx_p1 = int(total_len / 3)
    idx_p2 = int(2 * total_len / 3)
    
    print(f"Sample index: {idx}, Total Length: {total_len}")
    print(f"Phases: [0-{idx_p1}], [{idx_p1}-{idx_p2}], [{idx_p2}-{total_len}]")

    # --- GRAPHICAL LAYOUT ---
    fig = plt.figure(figsize=(18, 12))
    
    # Main grid: Top (Trajectory) and Bottom (Weights)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.3)
    
    # 1. Trajectory Zone (Top)
    gs_top = gs_main[0].subgridspec(1, 2, wspace=0.15)
    ax_q1 = fig.add_subplot(gs_top[0])
    ax_q2 = fig.add_subplot(gs_top[1])
    
    # 2. Weights Zone (Bottom)
    # 6 Columns: Col 0 = Progress Bar, Cols 1-5 = Weights
    gs_weights = gs_main[1].subgridspec(3, 6, width_ratios=[0.2, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.6)
    
    axes_weights = []
    axes_progress = [] 
    
    cost_names = ["dq1", "dq2", "V_ee", "Torque", "Energy"]
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(3): 
        # Column 0: Progress Bar
        ax_prog = fig.add_subplot(gs_weights[r, 0])
        axes_progress.append(ax_prog)
        
        # Columns 1-5: Weight Bars
        row_axes = []
        for c in range(5): 
            ax = fig.add_subplot(gs_weights[r, c+1]) 
            row_axes.append(ax)
        axes_weights.append(row_axes)

    last_reconstructed_q = np.zeros_like(q_true)
    RECONSTRUCTION_STEP = 10 

    def update(frame):
        # Start observing a bit after 0 to have meaningful input
        current_len = frame + 15 
        if current_len > total_len: current_len = total_len
        
        # --- INFERENCE ---
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) 
        traj_tensor = torch.FloatTensor(combined).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # Predict weights using diffusion
        w_pred_samples = sample_diffusion(model, traj_tensor, N_SAMPLES_DIFFUSION)
        w_pred_mean_flat = w_pred_samples.mean(axis=0)
        w_pred_matrix = w_pred_mean_flat.reshape(5, 3)
        
        # --- OCP RECONSTRUCTION ---
        nonlocal last_reconstructed_q
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            try:
                rec_q, _ = solve_DOC(
                    w_matrix=w_pred_matrix, 
                    N_steps=total_len, 
                    x_fin=params["x_fin"], 
                    q_init=params["q_init"],
                    verbose=False
                )
                if rec_q is not None:
                    last_reconstructed_q = rec_q
            except Exception:
                pass

        # --- PLOTTING TRAJECTORIES ---
        time_steps = np.arange(total_len)
        
        # Limits for fixed Y-axis
        q1_lims = (-150, 120)
        q2_lims = (-20, 180)

        for ax, q_idx, title, lims in zip([ax_q1, ax_q2], [0, 1], ["q1", "q2"], [q1_lims, q2_lims]):
            ax.clear()
            
            # Phase separation lines
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=1)
            
            # Active phase highlighter (light green background)
            if current_len <= idx_p1:
                ax.axvspan(0, idx_p1, color='green', alpha=0.05)
            elif current_len <= idx_p2:
                ax.axvspan(idx_p1, idx_p2, color='green', alpha=0.05)
            else:
                ax.axvspan(idx_p2, total_len, color='green', alpha=0.05)

            # Trajectories
            ax.plot(time_steps, np.degrees(q_true[:, q_idx]), 'k--', alpha=0.5, label='True')
            ax.plot(np.arange(current_len), np.degrees(q_true[:current_len, q_idx]), 'g-', linewidth=3, label='Observed')
            
            if last_reconstructed_q is not None and last_reconstructed_q.shape[0] == total_len:
                ax.plot(time_steps, np.degrees(last_reconstructed_q[:, q_idx]), 'r-', alpha=0.8, label='Pred')
            
            # Axis Setup
            ax.set_title(f"Joint {title}")
            ax.set_ylim(lims) # FIXED LIMITS
            
            if q_idx == 0: ax.legend(loc='upper right', fontsize='x-small')
            ax.grid(True, alpha=0.3)

            # Phase labels inside plot
            y_txt = lims[0] + (lims[1]-lims[0])*0.05
            ax.text(idx_p1/2, y_txt, "Phase 1", ha='center', fontsize=8, color='gray')
            ax.text(idx_p1 + (idx_p2-idx_p1)/2, y_txt, "Phase 2", ha='center', fontsize=8, color='gray')
            ax.text(idx_p2 + (total_len-idx_p2)/2, y_txt, "Phase 3", ha='center', fontsize=8, color='gray')

        # --- PLOTTING WEIGHTS & PROGRESS BARS ---
        
        # Calculate progress per phase
        prog_p1 = min(1.0, current_len / idx_p1)
        
        prog_p2 = 0.0
        if current_len > idx_p1:
            prog_p2 = min(1.0, (current_len - idx_p1) / (idx_p2 - idx_p1))
            
        prog_p3 = 0.0
        if current_len > idx_p2:
            prog_p3 = min(1.0, (current_len - idx_p2) / (total_len - idx_p2))
            
        progress_values = [prog_p1, prog_p2, prog_p3]

        for r in range(3):
            # 1. Progress Bar (Column 0)
            ax_p = axes_progress[r]
            ax_p.clear()
            
            p_val = progress_values[r]
            color_bar = 'lightgray' if p_val == 0 else ('limegreen' if p_val == 1 else 'green')
            
            ax_p.barh([0], [p_val], color=color_bar, height=0.5)
            ax_p.set_xlim(0, 1)
            ax_p.set_ylim(-0.5, 0.5)
            ax_p.axis('off') 
            
            # Label phase
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=10, fontweight='bold', transform=ax_p.transData)
            # Label percentage
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            # 2. Weights (Columns 1-5)
            for c in range(5): 
                ax = axes_weights[r][c]
                ax.clear()
                
                val_true = w_true[c, r]
                val_pred = w_pred_matrix[c, r]
                
                x_pos = [0, 1]
                values = [val_true, val_pred]
                colors = ['black', 'orange']
                
                # Gray out predictions if phase hasn't started
                if progress_values[r] == 0:
                    colors = ['black', 'lightgray']
                
                ax.bar(x_pos, values, color=colors, alpha=0.7)
                ax.set_ylim(0, 1.0) 
                ax.set_xticks([])
                
                if r == 0: ax.set_title(cost_names[c], fontsize=9, fontweight='bold')
                
                # Numeric values
                ax.text(0, val_true + 0.05, f"{val_true:.2f}", ha='center', fontsize=7, color='black')
                if progress_values[r] > 0:
                    ax.text(1, val_pred + 0.05, f"{val_pred:.2f}", ha='center', fontsize=7, color='darkorange')

    # Animation Generation
    frames = range(0, total_len - 15, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    
    gif_name = "progressive_phases_viz.gif"
    anim.save(gif_name, writer='pillow', fps=5)
    print(f"Animation saved as: {gif_name}")

if __name__ == "__main__":
    main()