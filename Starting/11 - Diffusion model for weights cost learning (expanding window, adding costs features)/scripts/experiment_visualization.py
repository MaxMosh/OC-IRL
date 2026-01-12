import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
import sys
import os

# --- IMPORT SOLVER ---
# Ensure the solver script is in the same folder or Python path
sys.path.append(os.getcwd())

try:
    from tools.OCP_solving_cpin_new_variable_corr_acc import solve_DOC
except ImportError:
    print("\n[WARNING] Could not import 'solve_DOC' from 'OCP_solving_cpin_new_variable_corr_acc.py'.")
    print("Please ensure the solver file is in the current directory.")
    solve_DOC = None

# --- CONSTANTS ---
L_1 = 1.0
L_2 = 1.0

# Initial configurations
Q_INIT_BASES_DEG = [[-90, 90], [-15, 105], [-115, 115]]
Q_INIT_NOISE_STD_DEG = 7.0

# Target
X_FIN_BASE = 1.9
X_FIN_NOISE_STD = 0.05 

# --- HELPER FUNCTIONS ---

def forward_kinematics(q1_rad, q2_rad):
    """Computes joint positions."""
    x1 = L_1 * np.cos(q1_rad)
    y1 = L_1 * np.sin(q1_rad)
    x_ee = x1 + L_2 * np.cos(q1_rad + q2_rad)
    y_ee = y1 + L_2 * np.sin(q1_rad + q2_rad)
    return (0, 0), (x1, y1), (x_ee, y_ee)

def gaussian_pdf(x, mu, sigma):
    """Standard Gaussian PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def draw_gradient_wedge(ax, center, radius, mean_angle_deg, std_deg, color, max_alpha=0.6):
    """
    Draws a wedge with a Gaussian gradient opacity to visualize angular uncertainty.
    """
    sigma_range = 3.0
    num_segments = 60 
    
    thetas = np.linspace(mean_angle_deg - sigma_range * std_deg, 
                         mean_angle_deg + sigma_range * std_deg, 
                         num_segments + 1)
    
    mid_thetas = (thetas[:-1] + thetas[1:]) / 2
    pdf_vals = gaussian_pdf(mid_thetas, mean_angle_deg, std_deg)
    
    if np.max(pdf_vals) > 0:
        alphas = pdf_vals / np.max(pdf_vals) * max_alpha
    else:
        alphas = np.zeros_like(pdf_vals)
    
    for i in range(num_segments):
        w = Wedge(center, radius, thetas[i], thetas[i+1], 
                  color=color, alpha=alphas[i], ec=None)
        ax.add_patch(w)

def draw_gradient_band(ax, x_mean, std, y_min, y_max, color, max_alpha=0.6):
    """
    Draws a vertical band with a Gaussian gradient opacity to visualize target position noise.
    """
    sigma_range = 3.0
    num_segments = 100
    
    x_vals = np.linspace(x_mean - sigma_range * std, 
                         x_mean + sigma_range * std, 
                         num_segments + 1)
    
    mid_xs = (x_vals[:-1] + x_vals[1:]) / 2
    pdf_vals = gaussian_pdf(mid_xs, x_mean, std)
    
    if np.max(pdf_vals) > 0:
        alphas = pdf_vals / np.max(pdf_vals) * max_alpha
    else:
        alphas = np.zeros_like(pdf_vals)
        
    for i in range(num_segments):
        width = x_vals[i+1] - x_vals[i]
        # We draw rectangles from y_min to y_max
        rect = Rectangle((x_vals[i], y_min), width, y_max - y_min, 
                         color=color, alpha=alphas[i], ec=None)
        ax.add_patch(rect)

def plot_robot_arm(ax, q1_rad, q2_rad, color='blue', alpha=1.0, lw=2, label=None):
    """Draws the robot links."""
    p0, p1, p2 = forward_kinematics(q1_rad, q2_rad)
    x_vals = [p0[0], p1[0], p2[0]]
    y_vals = [p0[1], p1[1], p2[1]]
    
    line, = ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=lw, alpha=alpha, markersize=5, label=label)
    return line

# --- PLOTTING ROUTINES ---

def plot_initial_conditions():
    """
    Plot 1: Initial configurations with Gaussian gradient noise on joints.
    """
    # Square-ish figure to keep good aspect ratio for a single plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    # Plot Target Line
    ax.axvline(x=X_FIN_BASE, color='purple', linestyle='--', linewidth=2, label=f'Target x = {X_FIN_BASE}m')
    
    for i, (q1_deg, q2_deg) in enumerate(Q_INIT_BASES_DEG):
        q1_rad = np.deg2rad(q1_deg)
        q2_rad = np.deg2rad(q2_deg)
        c = colors[i]
        
        # 1. Gradient Wedge for Joint 1
        draw_gradient_wedge(ax, (0, 0), L_1 * 0.35, q1_deg, Q_INIT_NOISE_STD_DEG, color=c)
        
        # 2. Gradient Wedge for Joint 2
        elbow_x = L_1 * np.cos(q1_rad)
        elbow_y = L_1 * np.sin(q1_rad)
        abs_angle_deg = np.rad2deg(q1_rad + q2_rad)
        draw_gradient_wedge(ax, (elbow_x, elbow_y), L_2 * 0.35, abs_angle_deg, Q_INIT_NOISE_STD_DEG, color=c)
        
        # 3. Draw nominal arm
        plot_robot_arm(ax, q1_rad, q2_rad, color=c, lw=3, label=f'Config {i+1}')

    # Strict aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_title("Robot Initial Configurations with Gaussian Joint Noise", fontsize=14)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("experiment_initial_conditions_gradient.png", dpi=300)
    print("Saved: experiment_initial_conditions_gradient.png")


def plot_terminal_solutions_subplots():
    """
    Plot 2: Three subplots horizontally (1x3), one for each scenario.
    """
    if solve_DOC is None:
        print("Skipping terminal solution plot (Solver not found).")
        return

    print("Solving OCPs for trajectory visualization...")
    
    # Create 3 subplots horizontally
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Define explicit drawing limits for the gradient band (extended to +/- 3)
    y_band_min = -3.0
    y_band_max = 3.0
    
    # Solver Params
    N_steps = 120
    w_matrix = np.ones((5, 3)) 
    w_matrix[2, :] = 2.0  
    w_matrix[3, :] = 0.01 
    w_matrix = w_matrix / w_matrix.sum(axis=0)

    for i, q_init_deg in enumerate(Q_INIT_BASES_DEG):
        ax = axs[i]
        q_init_rad = np.deg2rad(q_init_deg)
        c = colors[i]
        
        print(f"  > Solving for Config {i+1}...")
        res_q, _, _ = solve_DOC(w_matrix, N_steps, x_fin=X_FIN_BASE, q_init=q_init_rad, verbose=False)
        
        # --- Background Elements ---
        # 1. Gradient Band for Target (DRAWN FROM -3 TO 3)
        draw_gradient_band(ax, X_FIN_BASE, X_FIN_NOISE_STD, y_band_min, y_band_max, color='purple')
        ax.axvline(X_FIN_BASE, color='purple', linestyle='--', alpha=0.8)
        
        # 2. Workspace Boundary
        workspace = plt.Circle((0, 0), L_1 + L_2, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(workspace)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("X [m]") 
        
        if i == 0:
            ax.set_ylabel("Y [m]")

        # IMPORTANT: Force aspect ratio to be equal
        ax.set_aspect('equal', adjustable='box')
        
        # View limits (can be tighter than the band drawing)
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.5, 2.0)
        
        if res_q is None:
            ax.text(0, 0, "Optimization Failed", ha='center')
            continue

        # --- Trajectory Ghosts ---
        step_size = 14 
        total_points = res_q.shape[0]
        
        for t in range(0, total_points, step_size):
            prog = t / total_points
            q_t = res_q[t]
            
            # Alpha fading
            alpha_val = 0.15 + 0.5 * prog
            plot_robot_arm(ax, q_t[0], q_t[1], color=c, alpha=alpha_val, lw=1.5)

        # --- Final Position ---
        plot_robot_arm(ax, res_q[-1][0], res_q[-1][1], color='black', alpha=1.0, lw=2.5, label='Final Position')
        
        # Mark End Effector
        _, _, p_ee = forward_kinematics(res_q[-1][0], res_q[-1][1])
        ax.plot(p_ee[0], p_ee[1], marker='*', color=c, markersize=12, markeredgecolor='k')
        
        # Subplot Title
        ax.set_title(f"Scenario {i+1}\nStart [{q_init_deg[0]}°, {q_init_deg[1]}°]", fontsize=11)
        
        # Legend only on first plot
        if i == 0:
            import matplotlib.patches as mpatches
            target_patch = mpatches.Patch(color='purple', alpha=0.4, label='Target Noise')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(target_patch)
            labels.append('Target Noise')
            ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=9)

    fig.suptitle(f"Optimal Trajectories (N={N_steps}) with Target Uncertainty", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    save_path = "experiment_terminal_solutions_subplots_1x3.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists("data"):
        try: os.makedirs("data")
        except OSError: pass

    print("--- Generating Report Plots ---")
    
    plot_initial_conditions()
    plot_terminal_solutions_subplots()
    
    print("Done.")