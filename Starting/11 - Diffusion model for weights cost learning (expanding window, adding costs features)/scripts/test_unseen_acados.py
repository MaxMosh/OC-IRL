import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import copy
import pinocchio as pin
import joblib 

# Add current folder to path
sys.path.append(os.getcwd())

# --- IMPORTS FROM YOUR PROJECT ---
# Helper needed for Acados setup
from utils.reader_parameters import parse_params, convert_to_class
from utils.model_utils_motif import Robot, build_biomechanical_model 
from utils.doc_utils_new_acados_refactor3 import DocHumanMotionGeneration_InvDyn

# Diffusion Model
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_DIFFUSION = 50 

# Model Architecture Config (Must match training)
W_DIM = 12          # 3 Phases * 4 Costs
N_PHASES = 3
N_COSTS = 4         # ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]
INPUT_CHANNELS = 4  # q1, q2, dq1, dq2
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# Paths
CHECKPOINT_DIR = "checkpoints_no_scaling/diff_model_acados"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_50.pth") 
# Try to find the latest model if final doesn't exist, or use a specific epoch
# if not os.path.exists(MODEL_PATH):
#     MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_100.pth")

SCALER_W_PATH = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
SCALER_TRAJ_PATH = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")
PARAM_FILE = "parameters.toml" # Ensure this exists

# Simulation / OCP Params
COST_ORDER = ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]
# RECONSTRUCTION_STEP = 20  # Re-solve OCP every X frames to save time
RECONSTRUCTION_STEP = 1

# --- 1. SETUP ROBOT & ACADOS (From doc_acados_parallel.py) ---
# def setup_robot_and_param():
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     parent_directory = os.path.dirname(script_directory) # Assuming script is in root or subfolder

#     # Load parameters
#     try:
#         dict_param = parse_params(os.path.join(os.getcwd(), PARAM_FILE))
#     except FileNotFoundError:
#         # Fallback if running from a script folder
#         dict_param = parse_params(os.path.join(os.getcwd(), "scripts", PARAM_FILE))
        
#     param = convert_to_class(dict_param) 

#     # Load Robot
#     urdf_name = "human.urdf"
#     urdf_path = os.path.join(os.getcwd(), "model/human_urdf/urdf/", urdf_name)
#     urdf_meshes_path = os.path.join(os.getcwd(), "model")

#     robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer) 
#     model = robot.model
#     model, collision_model, visual_model, param = build_biomechanical_model(robot, param) 
    
#     # Lock joints (similar to parallel script)
#     quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs() 
#     q = np.zeros(model.nq)
#     q[3:7] = quat 
    
#     param.free_flyer = False
#     q_lock = q
#     joints_to_lock = [model.joints[1].id]
    
#     geom_models = [visual_model, collision_model]
#     model_red, geom_models_red = pin.buildReducedModel(
#         model, geom_models, joints_to_lock, q_lock
#     )
    
#     # Return the reduced model and the param class
#     return model_red, param

def setup_robot_and_param():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(script_directory)

    # 1. Chargement TOML
    try:
        dict_param = parse_params(os.path.join(os.getcwd(), PARAM_FILE))
    except FileNotFoundError:
        dict_param = parse_params(os.path.join(os.getcwd(), "scripts", PARAM_FILE))
        
    param = convert_to_class(dict_param) 

    # 2. Robot
    urdf_name = "human.urdf"
    urdf_path = os.path.join(os.getcwd(), "model/human_urdf/urdf/", urdf_name)
    urdf_meshes_path = os.path.join(os.getcwd(), "model")

    robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer) 
    model = robot.model
    model, collision_model, visual_model, param = build_biomechanical_model(robot, param) 
    
    # 3. Lock Joints
    quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs() 
    q = np.zeros(model.nq)
    q[3:7] = quat 
    
    param.free_flyer = False
    q_lock = q
    joints_to_lock = [model.joints[1].id]
    
    geom_models = [visual_model, collision_model]
    model_red, geom_models_red = pin.buildReducedModel(
        model, geom_models, joints_to_lock, q_lock
    )
    
    # 4. FOI (Frames Of Interest)
    param.FOI_to_set = ["right_hand"]
    param.FOI_axes = ["x"]
    
    data_red = model_red.createData()
    q0 = pin.neutral(model_red)
    pin.forwardKinematics(model_red, data_red, q0)
    pin.updateFramePlacements(model_red, data_red)
    
    param.FOI_to_set_Id = []
    param.FOI_position = []
    param.FOI_orientation = []
    
    for name in param.FOI_to_set:
        fid = model_red.getFrameId(name)
        param.FOI_to_set_Id.append(fid)
        param.FOI_position.append(data_red.oMf[fid].translation.copy())
        param.FOI_orientation.append(data_red.oMf[fid].rotation.copy())

    # 5. Poids Généraux
    param.variables_w = True 
    param.nb_w = 3
    param.weights = {}
    
    if hasattr(param, "active_costs"):
        for cost in param.active_costs:
            if cost == "min_joint_torque":
                if hasattr(param, "groups_joint_torques") and param.groups_joint_torques.get("all") == True:
                     param.weights[cost] = 1e-3 * np.ones(param.nb_w)
                else:
                    param.weights[cost] = np.ones(param.nb_w)
            else:
                param.weights[cost] = np.ones(param.nb_w)

    # 6. Poids 'target'
    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    if nb_targets > 0:
        param.weights["target"][0] = 5e4
        
    # 7. --- CORRECTION : AJOUT DE QDF (Final Joint Configuration) ---
    # Valeur par défaut utilisée dans doc_acados_paralell.py
    # qdf = [pi/2, 0] correspond à un bras tendu à l'horizontale (environ)
    param.qdf = np.array([np.pi/2, 0.0])

    return model_red, param

# --- 2. WEIGHT SAMPLER (Dirichlet Mixture) ---
def sample_W_dirichlet_mixture(n_phases=3, n_costs=4, seed=None):
    """
    Generates a SINGLE sample of weights (3, 4) summing to 1 across all elements.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Hyperparams from your provided script
    a_values = np.asarray((0.2, 0.5, 1.0, 2.0), dtype=float)
    a_probs = np.asarray((0.15, 0.25, 0.45, 0.15), dtype=float)
    a_probs = a_probs / a_probs.sum()

    d = n_phases * n_costs
    
    a = rng.choice(a_values, p=a_probs)
    w_flat = rng.dirichlet(np.full(d, a))  # (12,)
    W = w_flat.reshape(n_phases, n_costs)  # (3, 4)

    return W

# --- 3. OCP SOLVER INTERFACE ---
def solve_ocp_interface(W_matrix, model, param_template, q_init, x_fin, nb_samples):
    """
    Configures and solves the OCP using the DocHumanMotionGeneration_InvDyn class.
    """
    # Create a deep copy to avoid modifying the template globally
    param = copy.deepcopy(param_template)
    
    # Update constraints
    param.nb_samples = nb_samples
    param.Tf = param.nb_samples * 1/50 - 1/50
    param.qdi = q_init
    
    # Update Target (FOI)
    # Re-calculate indices just in case
    param.FOI_to_set_Id = []
    for name in param.FOI_to_set:
        param.FOI_to_set_Id.append(model.getFrameId(name))
    
    # We only update the X position of the first FOI (Right Hand)
    # We keep Y and Z from the template or previous config
    # Note: param.FOI_position is a list of arrays
    current_pos = param.FOI_position[0].copy() 
    current_pos[0] = x_fin # Set X
    param.FOI_position[0] = current_pos
    
    param.FOI_sample = [param.nb_samples] # Constrain at end

    # Update Weights
    # W_matrix shape (3, 4) -> (phases, costs)
    # param.weights needs: key -> list of 3 values
    for i, cost_name in enumerate(COST_ORDER):
        param.weights[cost_name] = W_matrix[:, i].tolist()

    # Update OCP Name to avoid conflicts
    rand_id = np.random.randint(0, 100000)
    param.ocp_name = f"test_diffusion_{rand_id}"
    param.build_solver = True 

    # Solve
    try:
        doc = DocHumanMotionGeneration_InvDyn(model, param)
        xs, us, fs = doc.solve_doc_acados(param)
        
        # Extract q, dq
        # xs is (N, state_dim). state_dim usually [q, dq]
        q_sol = xs[:, :model.nq]
        dq_sol = xs[:, model.nq:]
        
        return q_sol, dq_sol, True
    except Exception as e:
        print(f"OCP Solver failed: {e}")
        return None, None, False

# --- 4. DIFFUSION UTILS ---
def load_scalers():
    if not os.path.exists(SCALER_W_PATH) or not os.path.exists(SCALER_TRAJ_PATH):
        raise FileNotFoundError("Scalers not found. Did you run training?")
    return joblib.load(SCALER_TRAJ_PATH), joblib.load(SCALER_W_PATH)

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

    w_pred_numpy = w_current.cpu().numpy()
    w_pred_unscaled = scaler_w.inverse_transform(w_pred_numpy)
    # Clip to avoid negative weights (physically impossible)
    w_pred_unscaled = np.clip(w_pred_unscaled, 1e-6, 1.0)
    
    return w_pred_unscaled

def calculate_rmse(y_true, y_pred):
    # On prend la longueur minimale commune pour éviter le crash de broadcast
    min_len = min(y_true.shape[0], y_pred.shape[0])
    
    # On compare sur la portion commune
    diff = y_true[:min_len] - y_pred[:min_len]
    return np.sqrt(np.mean(diff**2))

# --- MAIN SCRIPT ---
def main():
    print("--- Setting up Robot & Environment ---")
    model, param_template = setup_robot_and_param()
    
    print(f"--- Loading Model from {MODEL_PATH} ---")
    scaler_traj, scaler_w = load_scalers()
    
    diffusion_model = ConditionalDiffusionModel(
        w_dim=W_DIM, 
        input_channels=INPUT_CHANNELS, 
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    diffusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # --- Generate Ground Truth ---
    print("--- Generating Ground Truth Trajectory (Acados) ---")
    
    # 1. Random Weights
    W_true = sample_W_dirichlet_mixture(n_phases=N_PHASES, n_costs=N_COSTS)
    
    # 2. Random Init / Final
    # P2 approx: [90, 90] degrees -> [pi/2, pi/2]
    q_init_deg = np.array([90.0, 90.0])
    noise_deg = np.random.normal(0, 5, size=2) # Slight noise
    q_init_rad = np.deg2rad(q_init_deg + noise_deg)
    
    # Target X
    x_fin = 0.5 + np.random.normal(0, 0.05) # Adjust based on your robot reach
    
    nb_samples = 150 # Duration
    
    print(f"Target X: {x_fin:.3f}, q_init (deg): {np.rad2deg(q_init_rad)}")
    
    q_true, dq_true, success = solve_ocp_interface(
        W_true, model, param_template, q_init_rad, x_fin, nb_samples
    )
    
    if not success:
        print("Failed to generate Ground Truth. Exiting.")
        return

    total_len = q_true.shape[0]
    idx_p1 = int(total_len / 3)
    idx_p2 = int(2 * total_len / 3)

    # --- PREPARE ANIMATION ---
    print("--- Starting Animation Generation ---")
    
    fig = plt.figure(figsize=(16, 12))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    
    # Top: Trajectories (2x2)
    gs_top = gs_main[0].subgridspec(2, 2, wspace=0.25, hspace=0.3)
    ax_q1 = fig.add_subplot(gs_top[0, 0])
    ax_q2 = fig.add_subplot(gs_top[0, 1])
    ax_dq1 = fig.add_subplot(gs_top[1, 0])
    ax_dq2 = fig.add_subplot(gs_top[1, 1])
    
    traj_axes = [ax_q1, ax_q2, ax_dq1, ax_dq2]
    traj_titles = ["Joint Position q1", "Joint Position q2", "Joint Velocity dq1", "Joint Velocity dq2"]
    traj_units = ["[rad]", "[rad]", "[rad/s]", "[rad/s]"]

    # Bottom: Weights Distribution (Rows: Phases, Cols: Costs)
    gs_weights = gs_main[1].subgridspec(N_PHASES, N_COSTS + 1, width_ratios=[0.2] + [1]*N_COSTS, wspace=0.3, hspace=0.5)
    
    axes_weights = []
    axes_progress = []
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(N_PHASES):
        axes_progress.append(fig.add_subplot(gs_weights[r, 0]))
        row_axes = [fig.add_subplot(gs_weights[r, c+1]) for c in range(N_COSTS)]
        axes_weights.append(row_axes)

    # Variables for Reconstruction
    last_reconstructed_q = np.zeros_like(q_true)
    last_reconstructed_dq = np.zeros_like(dq_true)
    
    # Store RMSE history for final print
    rmse_history = {"q": [], "dq": []}

    def update(frame):
        current_len = frame + 10 # Min window size
        if current_len > total_len: current_len = total_len
        
        # 1. Diffusion Inference
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) 
        
        # Scale
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # Sample Weights
        w_pred_samples_flat = sample_diffusion(diffusion_model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w)
        w_pred_samples = w_pred_samples_flat.reshape(N_SAMPLES_DIFFUSION, N_PHASES, N_COSTS)
        
        # Normalize
        sums = w_pred_samples.sum(axis=2, keepdims=True) 
        sums[sums == 0] = 1.0 
        w_pred_samples_norm = w_pred_samples / sums 
        
        # Mean
        w_pred_mean = w_pred_samples_norm.mean(axis=0) 
        
        # 2. Reconstruction
        nonlocal last_reconstructed_q, last_reconstructed_dq
        
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            # On demande (total_len - 1) intervalles pour avoir total_len points
            # Mais par sécurité, on tronquera après
            rec_q, rec_dq, ok = solve_ocp_interface(
                w_pred_mean, model, param_template, q_init_rad, x_fin, nb_samples=total_len - 1
            )
            if ok:
                # --- CORRECTION CRITIQUE ICI ---
                # On force la taille à être exactement celle de q_true (total_len)
                # Si rec_q est trop grand (152), on coupe à 151.
                limit = min(rec_q.shape[0], total_len)
                
                # On réinitialise des buffers de la bonne taille (total_len)
                # remplis avec la dernière valeur connue (ou des zéros) pour éviter les trous
                new_q = np.zeros_like(q_true)
                new_dq = np.zeros_like(dq_true)
                
                new_q[:limit] = rec_q[:limit]
                new_dq[:limit] = rec_dq[:limit]
                
                last_reconstructed_q = new_q
                last_reconstructed_dq = new_dq

        # 3. Calculate RMSE (Safety Net déjà en place)
        rmse_q = calculate_rmse(q_true, last_reconstructed_q)
        rmse_dq = calculate_rmse(dq_true, last_reconstructed_dq)

        # --- PLOTTING ---
        # time_steps a pour longueur total_len (151)
        time_steps = np.arange(total_len)
        
        datas = [
            (q_true[:,0], last_reconstructed_q[:,0]),
            (q_true[:,1], last_reconstructed_q[:,1]),
            (dq_true[:,0], last_reconstructed_dq[:,0]),
            (dq_true[:,1], last_reconstructed_dq[:,1])
        ]
        
        for i, ax in enumerate(traj_axes):
            ax.clear()
            true_d, pred_d = datas[i]
            
            # --- CORRECTION PLOT ---
            # On s'assure que pred_d a la même taille que time_steps pour le plot
            if pred_d.shape[0] > len(time_steps):
                pred_d = pred_d[:len(time_steps)]
            
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=0.8)
            ax.axvspan(0, current_len, color='green', alpha=0.05)

            ax.plot(time_steps, true_d, 'k--', alpha=0.6, label='Ground Truth')
            
            # On ne plot que si on a des données non nulles
            if np.any(last_reconstructed_q):
                ax.plot(time_steps, pred_d, 'r-', linewidth=1.5, alpha=0.9, label='Reconstruction')
            
            ax.set_title(traj_titles[i])
            ax.set_ylabel(traj_units[i])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize='x-small')
                ax.text(0.02, 0.95, f"RMSE Q: {rmse_q:.4f}", transform=ax.transAxes, color='red', fontsize=9, fontweight='bold')
            if i == 2:
                 ax.text(0.02, 0.95, f"RMSE DQ: {rmse_dq:.4f}", transform=ax.transAxes, color='red', fontsize=9, fontweight='bold')

        # --- WEIGHTS PLOTTING ---
        prog = [
            min(1.0, current_len/idx_p1), 
            min(1.0, (current_len-idx_p1)/(idx_p2-idx_p1)) if current_len > idx_p1 else 0,
            min(1.0, (current_len-idx_p2)/(total_len-idx_p2)) if current_len > idx_p2 else 0
        ]

        for r in range(N_PHASES):
            ax_p = axes_progress[r]
            ax_p.clear()
            p_val = prog[r]
            color = 'limegreen' if p_val >= 0.99 else ('lightgray' if p_val <= 0.01 else 'green')
            
            ax_p.barh([0], [p_val], color=color, height=0.5)
            ax_p.set_xlim(0, 1); ax_p.set_ylim(-0.5, 0.5); ax_p.axis('off')
            ax_p.text(-0.1, 0, phase_names[r], ha='right', va='center', fontsize=9, fontweight='bold', transform=ax_p.transData)
            ax_p.text(0.5, 0, f"{int(p_val*100)}%", ha='center', va='center', color='white' if p_val > 0.5 else 'black', fontsize=8)

            for c in range(N_COSTS):
                ax = axes_weights[r][c]
                ax.clear()
                row_sum_true = W_true[r, :].sum()
                val_true = W_true[r, c] / row_sum_true
                val_preds = w_pred_samples_norm[:, r, c]
                
                ax.hist(val_preds, bins=np.linspace(0, 1, 15), color='orange', alpha=0.6, density=True)
                ax.axvline(x=val_true, color='black', linestyle='--', linewidth=2, label='True')
                ax.set_xlim(0, 1.0)
                ax.set_yticks([]) 
                if r == 0: ax.set_title(COST_ORDER[c], fontsize=8, rotation=10)
                if r == N_PHASES - 1: ax.set_xlabel("Weight Value", fontsize=7)

    # Run Animation
    # Skip some frames for speed if needed
    frames = range(0, total_len - 5, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=200)
    
    # Save
    save_path = "test_dirichlet_acados_reconstruction.mp4"
    print(f"--- Saving Animation to {save_path} ---")
    try:
        anim.save(save_path, writer='ffmpeg', fps=10, extra_args=['-vcodec', 'libx264'])
        print("Done.")
    except Exception as e:
        print(f"FFMpeg failed ({e}), trying GIF...")
        anim.save("test_dirichlet_acados_reconstruction.gif", writer='pillow', fps=10)
        print("Done (GIF).")

if __name__ == "__main__":
    main()