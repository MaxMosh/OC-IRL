import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import copy
import pickle
import pinocchio as pin
import joblib 

# Add current folder to path
sys.path.append(os.getcwd())

# --- IMPORTS FROM YOUR PROJECT ---
from utils.reader_parameters import parse_params, convert_to_class
from utils.model_utils_motif import Robot, build_biomechanical_model 
from utils.doc_utils_new_acados_refactor3 import DocHumanMotionGeneration_InvDyn
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000

# Nombre de trajectoires pour l'ensemble (nuage)
N_SAMPLES_DIFFUSION = 20  
RECONSTRUCTION_STEP = 5   

# --- PARAMETRES DE SELECTION ---
DATASET_PATH = "data/dataset_unifie.pkl" 
SAMPLE_IDX = 100 

# Model Architecture
W_DIM = 12          
N_PHASES = 3
N_COSTS = 4         
INPUT_CHANNELS = 4  
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# Paths
CHECKPOINT_DIR = "checkpoints_no_scaling/diff_model_acados"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_250.pth") 

SCALER_W_PATH = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
SCALER_TRAJ_PATH = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")
PARAM_FILE = "parameters.toml"

COST_ORDER = ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]

# --- 1. SETUP ROBOT & ACADOS ---
def setup_robot_and_param():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        dict_param = parse_params(os.path.join(os.getcwd(), PARAM_FILE))
    except FileNotFoundError:
        dict_param = parse_params(os.path.join(os.getcwd(), "scripts", PARAM_FILE))
    param = convert_to_class(dict_param) 

    urdf_name = "human.urdf"
    urdf_path = os.path.join(os.getcwd(), "model/human_urdf/urdf/", urdf_name)
    urdf_meshes_path = os.path.join(os.getcwd(), "model")

    robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer) 
    model = robot.model
    model, collision_model, visual_model, param = build_biomechanical_model(robot, param) 
    
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

    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    if nb_targets > 0:
        param.weights["target"][0] = 5e4
        
    param.qdf = np.array([np.pi/2, 0.0])

    return model_red, param

# --- 2. LOAD DATASET SAMPLE ---
def load_training_sample(dataset_path, idx):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    n_samples = len(data["q_trajs"])
    if idx >= n_samples:
        raise IndexError(f"Index {idx} out of bounds (max {n_samples-1})")
    
    print(f"Extracting sample {idx} / {n_samples}...")
    q_traj = data["q_trajs"][idx]      
    dq_traj = data["dq_trajs"][idx]    
    w_raw = data["w_matrices"][idx]    
    W_matrix = w_raw.T                 
    params = data["params"][idx]
    
    return q_traj, dq_traj, W_matrix, params

# --- 3. OCP SOLVER INTERFACE ---
def solve_ocp_interface(W_matrix, model, param_template, q_init, x_fin, nb_samples):
    param = copy.deepcopy(param_template)
    param.nb_samples = nb_samples
    param.Tf = param.nb_samples * 1/50 - 1/50
    param.qdi = q_init
    
    param.FOI_to_set_Id = []
    for name in param.FOI_to_set:
        param.FOI_to_set_Id.append(model.getFrameId(name))
    
    current_pos = param.FOI_position[0].copy() 
    current_pos[0] = x_fin 
    param.FOI_position[0] = current_pos
    
    param.FOI_sample = [param.nb_samples] 

    for i, cost_name in enumerate(COST_ORDER):
        param.weights[cost_name] = W_matrix[:, i].tolist()

    rand_id = np.random.randint(0, 100000)
    param.ocp_name = f"test_ens_{rand_id}" 
    param.build_solver = True 

    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        doc = DocHumanMotionGeneration_InvDyn(model, param)
        xs, us, fs = doc.solve_doc_acados(param)
        sys.stdout = old_stdout
        
        q_sol = xs[:, :model.nq]
        dq_sol = xs[:, model.nq:]
        return q_sol, dq_sol, True
    except Exception as e:
        sys.stdout = old_stdout
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
    w_pred_unscaled = np.clip(w_pred_unscaled, 1e-6, 1.0)
    return w_pred_unscaled

def calculate_rmse(y_true, y_pred):
    min_len = min(y_true.shape[0], y_pred.shape[0])
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

    # --- LOAD GROUND TRUTH ---
    print(f"--- Loading Train Sample Index {SAMPLE_IDX} ---")
    q_true, dq_true, W_true, params = load_training_sample(DATASET_PATH, SAMPLE_IDX)
    
    q_init_rad = params['q_init']
    x_fin = params['x_fin']
    
    print(f"Sample Params -> Target X: {x_fin:.3f}, q_init (deg): {np.rad2deg(q_init_rad)}")

    total_len = q_true.shape[0]
    idx_p1 = int(total_len / 3)
    idx_p2 = int(2 * total_len / 3)

    # --- PREPARE ANIMATION ---
    print("--- Starting Animation Generation ---")
    
    fig = plt.figure(figsize=(16, 12))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    
    # Trajectories
    gs_top = gs_main[0].subgridspec(2, 2, wspace=0.25, hspace=0.3)
    ax_q1 = fig.add_subplot(gs_top[0, 0])
    ax_q2 = fig.add_subplot(gs_top[0, 1])
    ax_dq1 = fig.add_subplot(gs_top[1, 0])
    ax_dq2 = fig.add_subplot(gs_top[1, 1])
    traj_axes = [ax_q1, ax_q2, ax_dq1, ax_dq2]
    traj_titles = ["Joint Position q1", "Joint Position q2", "Joint Velocity dq1", "Joint Velocity dq2"]
    traj_units = ["[rad]", "[rad]", "[rad/s]", "[rad/s]"]

    # Weights
    gs_weights = gs_main[1].subgridspec(N_PHASES, N_COSTS + 1, width_ratios=[0.2] + [1]*N_COSTS, wspace=0.3, hspace=0.5)
    axes_weights = []
    axes_progress = []
    phase_names = ["Phase 1\n(Start)", "Phase 2\n(Mid)", "Phase 3\n(End)"]
    
    for r in range(N_PHASES):
        axes_progress.append(fig.add_subplot(gs_weights[r, 0]))
        row_axes = [fig.add_subplot(gs_weights[r, c+1]) for c in range(N_COSTS)]
        axes_weights.append(row_axes)

    # STORAGE
    last_reconstructed_q = np.zeros_like(q_true)
    last_reconstructed_dq = np.zeros_like(dq_true)
    last_ensemble_qs = []  
    last_ensemble_dqs = [] 

    def update(frame):
        # --- CORRECTION : Déclaration nonlocal AU DÉBUT ---
        nonlocal last_reconstructed_q, last_reconstructed_dq
        nonlocal last_ensemble_qs, last_ensemble_dqs
        
        current_len = frame + 10 
        if current_len > total_len: current_len = total_len
        
        # Info progression
        n_solved = len(last_ensemble_qs)
        print(f"Frame {current_len}/{total_len} | Ensemble: {n_solved}/{N_SAMPLES_DIFFUSION} solved...", end="\r", flush=True)
        
        # 1. Diffusion Inference
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) 
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        w_pred_samples_flat = sample_diffusion(diffusion_model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w)
        w_pred_samples = w_pred_samples_flat.reshape(N_SAMPLES_DIFFUSION, N_PHASES, N_COSTS)
        sums = w_pred_samples.sum(axis=2, keepdims=True) 
        sums[sums == 0] = 1.0 
        w_pred_samples_norm = w_pred_samples / sums 
        w_pred_mean = w_pred_samples_norm.mean(axis=0) 
        
        # 2. Reconstruction
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            # A. MOYENNE
            rec_q, rec_dq, ok = solve_ocp_interface(
                w_pred_mean, model, param_template, q_init_rad, x_fin, nb_samples=total_len - 1
            )
            if ok:
                limit = min(rec_q.shape[0], total_len)
                new_q = np.zeros_like(q_true); new_dq = np.zeros_like(dq_true)
                new_q[:limit] = rec_q[:limit]; new_dq[:limit] = rec_dq[:limit]
                last_reconstructed_q = new_q
                last_reconstructed_dq = new_dq
            
            # B. ENSEMBLE (Nuage)
            temp_ensemble_qs = []
            temp_ensemble_dqs = []
            
            for idx_w in range(N_SAMPLES_DIFFUSION):
                w_sample = w_pred_samples_norm[idx_w]
                s_q, s_dq, s_ok = solve_ocp_interface(
                    w_sample, model, param_template, q_init_rad, x_fin, nb_samples=total_len - 1
                )
                if s_ok:
                    lim_s = min(s_q.shape[0], total_len)
                    ens_q = np.zeros_like(q_true); ens_dq = np.zeros_like(dq_true)
                    ens_q[:lim_s] = s_q[:lim_s]; ens_dq[:lim_s] = s_dq[:lim_s]
                    temp_ensemble_qs.append(ens_q)
                    temp_ensemble_dqs.append(ens_dq)
            
            last_ensemble_qs = temp_ensemble_qs
            last_ensemble_dqs = temp_ensemble_dqs

        # 3. RMSE
        rmse_q = calculate_rmse(q_true, last_reconstructed_q)
        rmse_dq = calculate_rmse(dq_true, last_reconstructed_dq)

        # --- PLOTTING ---
        time_steps = np.arange(total_len)
        
        datas_mean = [
            (q_true[:,0], last_reconstructed_q[:,0]),
            (q_true[:,1], last_reconstructed_q[:,1]),
            (dq_true[:,0], last_reconstructed_dq[:,0]),
            (dq_true[:,1], last_reconstructed_dq[:,1])
        ]
        
        for i, ax in enumerate(traj_axes):
            ax.clear()
            true_d, pred_mean_d = datas_mean[i]
            
            # Background regions
            ax.axvline(x=idx_p1, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(x=idx_p2, color='gray', linestyle='--', linewidth=0.8)
            ax.axvspan(0, current_len, color='green', alpha=0.05)
            
            # --- 1. PLOT ENSEMBLE (Fond) ---
            # On vérifie si on a des données à plotter
            if len(last_ensemble_qs) == 0 and frame % RECONSTRUCTION_STEP == 0:
                # Debug info (optionnel, n'imprime rien si ça marche)
                pass 

            for j in range(len(last_ensemble_qs)):
                if i == 0: ens_d = last_ensemble_qs[j][:, 0]
                elif i == 1: ens_d = last_ensemble_qs[j][:, 1]
                elif i == 2: ens_d = last_ensemble_dqs[j][:, 0]
                elif i == 3: ens_d = last_ensemble_dqs[j][:, 1]
                
                len_plot = min(len(time_steps), len(ens_d))
                t_plot = time_steps[:len_plot]
                d_plot = ens_d[:len_plot]
                
                if np.any(d_plot):
                    # --- CHANGE: Alpha plus fort (0.4) et ligne plus épaisse ---
                    ax.plot(t_plot, d_plot, color='red', linewidth=1.5, alpha=0.4)

            # --- 2. PLOT TRUTH & MEAN ---
            ax.plot(time_steps, true_d, 'k--', alpha=0.6, label='Train Data (Truth)')
            
            if np.any(last_reconstructed_q):
                ax.plot(time_steps, pred_mean_d, 'r-', linewidth=2.0, alpha=1.0, label='Mean Pred')
            
            ax.set_title(traj_titles[i])
            ax.set_ylabel(traj_units[i])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize='x-small')
                ax.text(0.02, 0.95, f"RMSE Q: {rmse_q:.4f}", transform=ax.transAxes, color='red', fontsize=9, fontweight='bold')

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
                
                ax.hist(val_preds, bins=np.linspace(0, 1, 15), color='blue', alpha=0.6, density=False)
                ax.set_ylim(0, N_SAMPLES_DIFFUSION)
                ax.tick_params(axis='y', labelsize=6)
                ax.axvline(x=val_true, color='black', linestyle='--', linewidth=2, label='Train Val')
                ax.set_xlim(0, 1.0)
                
                if r == 0: ax.set_title(COST_ORDER[c], fontsize=8, rotation=10)
                if r == N_PHASES - 1: ax.set_xlabel("Weight Value", fontsize=7)

    frames = range(0, total_len - 5, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=200)
    
    save_path = f"test_train_sample_{SAMPLE_IDX}_ensemble.gif"
    print(f"--- Saving Animation to {save_path} (GIF ONLY) ---")
    
    # --- COMMENTED OUT MP4 ---
    # try:
    #     anim.save(save_path.replace(".gif", ".mp4"), writer='ffmpeg', fps=10, extra_args=['-vcodec', 'libx264'])
    #     print("Done (MP4).")
    # except Exception as e:
    #     print(f"MP4 failed: {e}")

    try:
        anim.save(save_path, writer='pillow', fps=10)
        print()
        print("Done (GIF).")
    except Exception as e:
        print(f"GIF failed: {e}")

if __name__ == "__main__":
    main()