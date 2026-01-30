import os
import sys
import numpy as np
import torch
import joblib
import pinocchio as pin
import matplotlib
matplotlib.use("Agg") # Headless plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

# --- USER IMPORTS ---
sys.path.append(os.getcwd())

from utils.reader_parameters import parse_params, convert_to_class
from utils.model_utils_motif import Robot, build_biomechanical_model
from utils.doc_utils_new_acados_refactor3 import DocHumanMotionGeneration_InvDyn
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIM = 4
INPUT_CHANNELS = 4
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
TIMESTEPS = 1000

N_SAMPLES_DIFFUSION = 50
UPDATE_INTERVAL = 5 # Update OCP every 5 frames
CHECKPOINT_DIR = "checkpoints_no_scaling/diff_model_dataset_acados_constant"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_600.pth")
SCALER_W_PATH = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
SCALER_TRAJ_PATH = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")

COST_KEYS = ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]

# --- 1. ENVIRONMENT SETUP ---
def setup_environment():
    """
    Initializes the robot model and parameters similarly to doc_acados.py
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_directory) == "scripts":
        script_directory = os.path.dirname(script_directory)
    
    # 1. Load Parameters
    toml_path = os.path.join(script_directory, 'parameters.toml')
    print(f"Loading parameters from {toml_path}")
    dict_param = parse_params(toml_path)
    param = convert_to_class(dict_param) 

    # 2. Load Robot
    urdf_name = "human.urdf"
    urdf_path = os.path.join(script_directory, "model/human_urdf/urdf/", urdf_name)
    urdf_meshes_path = os.path.join(script_directory, "model")

    robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer) 
    model = robot.model
    model, collision_model, visual_model, param = build_biomechanical_model(robot, param) 
    
    # 3. Reduce Model (Lock free-flyer and first joint)
    q = np.zeros(model.nq)
    quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs() 
    q[3:7] = quat 
    
    param.free_flyer = False
    q_lock = q
    joints_to_lock = [model.joints[1].id] 

    geom_models = [visual_model, collision_model]
    model_red, geom_models_red = pin.buildReducedModel(model, geom_models, joints_to_lock, q_lock)
    model = model_red.copy()
    
    # 4. Configure Frames of Interest (FOI)
    param.FOI_to_set = ["right_hand"] 
    param.FOI_axes = ["x"]
    param.FOI_to_set_Id = []
    for name in param.FOI_to_set:
        param.FOI_to_set_Id.append(model.getFrameId(name))
        
    param.FOI_position = [np.zeros(3)] 
    param.FOI_orientation = [np.eye(3)] # Identity matrix for orientation
    
    param.variables_w = True 
    param.nb_w = 1 
    
    return model, param

# --- 2. ACADOS SOLVER WRAPPER ---
def solve_with_acados_weights(model, base_param, w_vector, total_len, q_init, q_final, target_pos_xyz):
    """
    Configures and solves the OCP using Acados.
    Used for both Ground Truth generation and Reconstruction.
    """
    param = copy.deepcopy(base_param)
    
    # Time configuration
    param.nb_samples = total_len - 1
    param.Tf = param.nb_samples * (1/50) # 50 Hz assumption
    param.FOI_sample = [param.nb_samples]
    
    # Boundary Conditions
    param.qdi = np.array(q_init, dtype=float).flatten()
    param.qdf = np.array(q_final, dtype=float).flatten()
    
    # Target Update
    param.FOI_position[0] = np.array(target_pos_xyz, dtype=float)
    
    # Weights configuration
    w_flat = w_vector.flatten()
    # Allow small weights (same logic as generation)
    w_safe = np.maximum(np.abs(w_flat), 1e-6)
    
    param.weights = {}
    for i, key in enumerate(COST_KEYS):
        param.weights[key] = np.array([w_safe[i]]) 
        
    # Target Weight (Must match doc_acados_paralell.py)
    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    param.weights["target"][0] = 5e3 
    
    # Build solver
    param.build_solver = True
    param.ocp_name = f"ocp_gen_{np.random.randint(1e8)}" 
    
    try:
        # Suppress output if needed
        # sys.stdout = open(os.devnull, 'w')
        doc = DocHumanMotionGeneration_InvDyn(model, param)
        xs, us, fs = doc.solve_doc_acados(param)
        # sys.stdout = sys.__stdout__
        
        if xs is None or len(xs) == 0:
            return None, None
            
        return xs[:, :model.nq], xs[:, model.nq:]
        
    except Exception as e:
        print(f"Solver Error: {e}")
        return None, None

# --- 3. GENERATION OF UNSEEN DATA ---
def generate_ground_truth(model, param):
    """
    Generates a new random trajectory following the logic of doc_acados_paralell.py
    (Noise on q0, Noise on Target, Random Weights, Random Length)
    """
    print("Generating new UNSEEN Ground Truth sample...")
    
    # Standard P2 Posture configuration (from doc_acados_paralell.py)
    base_q_init = np.array([np.pi/2, np.pi/2]) # Shoulder 90, Elbow 90
    base_q_final = np.array([np.pi/2, 0])      # Shoulder 90, Elbow 0
    
    # Compute base target position (FK)
    data = model.createData()
    pin.forwardKinematics(model, data, base_q_final)
    pin.updateFramePlacements(model, data)
    hand_id = param.FOI_to_set_Id[0]
    base_target_pos = data.oMf[hand_id].translation.copy()
    
    max_retries = 10
    for attempt in range(max_retries):
        # 1. Randomize Length (between 65 and 140 samples as in doc_acados)
        nb_samples = np.random.randint(65, 140)
        total_len = nb_samples + 1
        
        # 2. Add Noise to Initial State (Sigma = 8 deg)
        sigma = np.deg2rad(8)
        noise_q = np.random.normal(loc=0.0, scale=sigma, size=base_q_init.shape)
        # Clip to limits (simplified limits)
        q_init_noisy = np.clip(base_q_init + noise_q, -np.pi, np.pi)
        
        # 3. Add Noise to Target Position (Scale = 0.03m)
        noise_target = np.random.normal(loc=0.0, scale=0.03, size=1)
        # In doc_acados, noise is added to X axis (index 0) and clipped
        target_pos_noisy = base_target_pos.copy()
        target_pos_noisy[0] = np.clip(target_pos_noisy[0] + noise_target, 
                                      base_target_pos[0]-0.1, 
                                      base_target_pos[0]+0.05)
        
        # 4. Sample Random Weights
        # Simulating a diverse distribution. 
        # Using uniform/log-uniform to explore the space [0, 1] roughly
        w_true = np.random.uniform(0.01, 1.0, 4) 
        
        print(f"Attempt {attempt+1}: Length={total_len}, Weights={np.round(w_true, 3)}")
        
        # 5. Solve to get GT
        q_gt, dq_gt = solve_with_acados_weights(
            model, param, w_true, total_len, q_init_noisy, base_q_final, target_pos_noisy
        )
        
        if q_gt is not None and len(q_gt) == total_len:
            print(">>> Success! Ground Truth generated.")
            return q_gt, dq_gt, w_true, q_init_noisy, target_pos_noisy
        else:
            print(">>> Failed to converge. Retrying...")

    raise RuntimeError("Could not generate a valid trajectory after max retries.")

# --- 4. DIFFUSION UTILS ---
def load_scalers():
    if not os.path.exists(SCALER_W_PATH): raise FileNotFoundError("Scalers not found")
    return joblib.load(SCALER_TRAJ_PATH), joblib.load(SCALER_W_PATH)

def predict_weights(model, condition_traj, n_samples, scaler_w):
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    seq_len = condition_traj.shape[2]
    mask = torch.zeros((n_samples, seq_len), dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        cond_repeated = condition_traj.repeat(n_samples, 1, 1)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            predicted_noise = model(w_current, t, cond_repeated, trajectory_mask=mask)
            
            alpha_t, alpha_hat_t, beta_t = alpha[i], alpha_hat[i], beta[i]
            noise = torch.randn_like(w_current) if i > 0 else 0
            
            w_current = (1 / torch.sqrt(alpha_t)) * (
                w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    w_np = w_current.cpu().numpy()
    w_unscaled = scaler_w.inverse_transform(w_np)
    return np.abs(w_unscaled)

# --- 5. MAIN ---
def main():
    # 1. Setup
    model_pin, param = setup_environment()
    scaler_traj, scaler_w = load_scalers()
    
    diff_model = ConditionalDiffusionModel(W_DIM, INPUT_CHANNELS, D_MODEL, NHEAD, NUM_LAYERS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        diff_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("Model not found."); return

    # 2. Generate UNSEEN Data
    # We ignore the pickle file and generate fresh data on the fly
    q_gt, dq_gt, w_true, q_init, target_pos = generate_ground_truth(model_pin, param)
    
    # 3. Animation Setup
    total_len = len(q_gt)
    # Use q_final from generation as q_final for reconstruction (it's mainly for initialization in Acados)
    q_final_ref = np.array([np.pi/2, 0]) 

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1])
    
    ax_q1, ax_q2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax_dq1, ax_dq2 = fig.add_subplot(gs[0,2]), fig.add_subplot(gs[0,3])
    hist_axes = [fig.add_subplot(gs[1,i]) for i in range(4)]
    
    last_rec_q = np.full_like(q_gt, np.nan)
    last_rec_dq = np.full_like(dq_gt, np.nan)
    
    def update(frame_idx):
        nonlocal last_rec_q, last_rec_dq
        
        curr_len = min(frame_idx + 10, total_len)
        
        # A. Inference
        combined = np.concatenate([q_gt[:curr_len], dq_gt[:curr_len]], axis=1)
        scaled = scaler_traj.transform(combined)
        tensor = torch.FloatTensor(scaled).transpose(0,1).unsqueeze(0).to(DEVICE)
        
        w_preds = predict_weights(diff_model, tensor, N_SAMPLES_DIFFUSION, scaler_w)
        w_mean = np.mean(w_preds, axis=0)
        
        # B. Solve OCP (Reconstruction)
        print(f"Frame {curr_len}/{total_len} | Solving OCP...", end="\r")
        rec_q, rec_dq = solve_with_acados_weights(
            model_pin, param, w_mean, total_len, q_init, q_final_ref, target_pos
        )
        
        if rec_q is not None and len(rec_q) >= total_len:
            last_rec_q = rec_q[:total_len]
            last_rec_dq = rec_dq[:total_len]
        
        # C. Plotting
        t = np.arange(total_len)
        
        # Plot Trajectories
        # Axes Limits: q -> [-70, 15], dq -> [-2, 1]
        for ax, data_gt, data_rec, title in zip(
            [ax_q1, ax_q2, ax_dq1, ax_dq2],
            [q_gt[:,0], q_gt[:,1], dq_gt[:,0], dq_gt[:,1]],
            [last_rec_q[:,0], last_rec_q[:,1], last_rec_dq[:,0], last_rec_dq[:,1]],
            ["Joint 1 Position (deg)", "Joint 2 Position (deg)", "Joint 1 Velocity (deg/s)", "Joint 2 Velocity (deg/s)"]
        ):
            ax.clear(); ax.set_title(title)
            
            if "Velocity" in title:
                # Limits for velocity
                # ax.set_ylim(np.rad2deg(-2, 1))
                ax.set_ylim(np.rad2deg((-2, 1)))
                ax.plot(t, np.rad2deg(data_gt), 'k--', alpha=0.5, label="Ground Truth")
                ax.plot(np.arange(curr_len), np.rad2deg(data_gt[:curr_len]), 'g-', lw=2, label="Observation")
                ax.plot(t, np.rad2deg(data_rec), 'r-', alpha=0.8, label="Prediction")
            else:
                # Limits for position (deg)
                ax.set_ylim(-70, 120)
                ax.plot(t, np.rad2deg(data_gt), 'k--', alpha=0.5, label="Ground Truth")
                ax.plot(np.arange(curr_len), np.rad2deg(data_gt[:curr_len]), 'g-', lw=2, label="Observation")
                ax.plot(t, np.rad2deg(data_rec), 'r-', alpha=0.8, label="Prediction")

            if "Joint 1 Position" in title: ax.legend(loc='lower left', fontsize='x-small')

        # Plot Weight Histograms
        # Axes Limits: x -> [0, 1], y -> [0, N_SAMPLES]
        for i, ax in enumerate(hist_axes):
            ax.clear(); ax.set_title(COST_KEYS[i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, N_SAMPLES_DIFFUSION)
            
            bins = np.linspace(0, 1, 15)
            ax.hist(w_preds[:,i], bins=bins, color='orange', alpha=0.7, edgecolor='white')
            
            ax.axvline(w_true[i], color='blue', lw=2, ls='--', label='True Weight')
            ax.axvline(w_mean[i], color='red', lw=2, label='Mean Pred')
            
            if i == 0: ax.legend(fontsize='x-small', loc='upper right')

    # Run Animation
    anim = FuncAnimation(fig, update, frames=range(5, total_len, UPDATE_INTERVAL), interval=200)
    
    save_path = "test_unseen_acados_robust.mp4"
    print(f"\nSaving animation to {save_path}...")
    try:
        anim.save(save_path, writer='ffmpeg', fps=5)
    except:
        print("FFmpeg not found/error. Saving as GIF.")
        anim.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=5)
    print("Done.")

if __name__ == "__main__":
    main()
