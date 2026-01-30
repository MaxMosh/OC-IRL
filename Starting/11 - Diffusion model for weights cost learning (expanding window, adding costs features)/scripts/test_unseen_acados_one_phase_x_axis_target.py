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
UPDATE_INTERVAL = 5 
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
    
    # 3. Reduce Model
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
    param.FOI_orientation = [np.eye(3)] 
    
    param.variables_w = True 
    param.nb_w = 1 
    
    return model, param

# --- 2. ACADOS SOLVER WRAPPER ---
def solve_with_acados_weights(model, base_param, w_vector, total_len, q_init, q_final_guess, target_pos_xyz):
    """
    Configures and solves the OCP using Acados.
    q_final_guess: Used for initialization (qdf), but the HARD constraint is target_pos_xyz.
    """
    param = copy.deepcopy(base_param)
    
    # Time configuration
    param.nb_samples = total_len - 1
    param.Tf = param.nb_samples * (1/50) 
    param.FOI_sample = [param.nb_samples]
    
    # Boundary Conditions
    param.qdi = np.array(q_init, dtype=float).flatten()
    param.qdf = np.array(q_final_guess, dtype=float).flatten() # Reference only
    
    # Target Update (The real constraint)
    param.FOI_position[0] = np.array(target_pos_xyz, dtype=float)
    
    # Weights configuration
    w_flat = w_vector.flatten()
    w_safe = np.maximum(np.abs(w_flat), 1e-6)
    
    param.weights = {}
    for i, key in enumerate(COST_KEYS):
        param.weights[key] = np.array([w_safe[i]]) 
        
    # Target Weight
    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    param.weights["target"][0] = 5e3 # High weight -> Drive hand to target
    
    # Build solver
    param.build_solver = True
    param.ocp_name = f"ocp_gen_{np.random.randint(1e8)}" 
    
    try:
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

# --- 3. GENERATION OF UNSEEN DATA (UPDATED) ---
def generate_ground_truth_cartesian(model, param):
    """
    Generates a GT trajectory based on Cartesian Reaching (90% arm length).
    """
    print("Generating new UNSEEN Ground Truth sample...")
    
    data = model.createData()
    hand_id = param.FOI_to_set_Id[0]
    
    # 1. Compute Total Arm Length
    # We set q=0 (arm extended) to measure max reach
    q_extended = np.zeros(model.nq)
    pin.forwardKinematics(model, data, q_extended)
    pin.updateFramePlacements(model, data)
    
    # Vector from Shoulder (Root) to Hand
    # Assuming the robot root is at (0,0,0) or close enough in this reduced model
    hand_pos_extended = data.oMf[hand_id].translation
    # If the base isn't at 0, strictly we should subtract base pos, 
    # but for berret_2dof, calculating norm of hand pos usually gives reach radius.
    arm_length = np.linalg.norm(hand_pos_extended)
    print(f"Computed Arm Length: {arm_length:.3f} m")

    # 2. Define Target based on Length (90% reach)
    # The target is on X axis (as per param.FOI_axes=["x"])
    target_distance = 0.9 * arm_length
    
    # Base target position [X, Y, Z]
    # For a planar robot (X-Y or X-Z), Y is usually 0 or fixed.
    # We take the extended position and scale it.
    base_target_pos = hand_pos_extended * 0.9
    
    # Override with specific logic if needed (e.g. strict X reach)
    # base_target_pos = np.array([target_distance, 0, 0]) 
    
    # Reference Final Q (Just for initialization, e.g., pointing forward)
    # [0, 0] is fully extended. [pi/2, 0] is up.
    # We leave it as [pi/2, 0] or [0, 0] as a "guess".
    base_q_final_guess = np.array([np.pi/2, 0]) 

    max_retries = 10
    for attempt in range(max_retries):
        # A. Randomize Length
        nb_samples = np.random.randint(65, 140)
        total_len = nb_samples + 1
        
        # B. Noise on Initial State
        base_q_init = np.array([np.pi/2, np.pi/2])
        sigma = np.deg2rad(8)
        noise_q = np.random.normal(loc=0.0, scale=sigma, size=base_q_init.shape)
        q_init_noisy = np.clip(base_q_init + noise_q, -np.pi, np.pi)
        
        # C. Noise on Target
        # Add noise mainly on the reaching axis (X usually)
        noise_target = np.random.normal(loc=0.0, scale=0.03, size=1)
        target_pos_noisy = base_target_pos.copy()
        
        # We assume X is the main reaching axis (index 0)
        # We verify FOI_axes from param just in case
        if "x" in param.FOI_axes:
            target_pos_noisy[0] += noise_target
        
        # D. Random Weights
        w_true = np.random.uniform(0.01, 1.0, 4) 
        
        print(f"Attempt {attempt+1}: Length={total_len}, TargetX={target_pos_noisy[0]:.3f}")
        
        # E. Solve
        q_gt, dq_gt = solve_with_acados_weights(
            model, param, w_true, total_len, q_init_noisy, base_q_final_guess, target_pos_noisy
        )
        
        if q_gt is not None and len(q_gt) == total_len:
            print(">>> Success! Ground Truth generated.")
            return q_gt, dq_gt, w_true, q_init_noisy, target_pos_noisy, base_q_final_guess
        else:
            print(">>> Failed to converge. Retrying...")

    raise RuntimeError("Could not generate a valid trajectory.")

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

    # 2. Generate UNSEEN Data (Cartesian Logic)
    # We get q_final_guess (e.g. pi/2,0) but the solver used target_pos
    q_gt, dq_gt, w_true, q_init, target_pos, q_final_guess = generate_ground_truth_cartesian(model_pin, param)
    
    # 3. Animation Setup
    total_len = len(q_gt)
    
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
        # Ensure we pass the SAME target_pos as generated
        rec_q, rec_dq = solve_with_acados_weights(
            model_pin, param, w_mean, total_len, q_init, q_final_guess, target_pos
        )
        
        if rec_q is not None and len(rec_q) >= total_len:
            last_rec_q = rec_q[:total_len]
            last_rec_dq = rec_dq[:total_len]
        
        # C. Plotting
        t = np.arange(total_len)
        
        for ax, data_gt, data_rec, title in zip(
            [ax_q1, ax_q2, ax_dq1, ax_dq2],
            [q_gt[:,0], q_gt[:,1], dq_gt[:,0], dq_gt[:,1]],
            [last_rec_q[:,0], last_rec_q[:,1], last_rec_dq[:,0], last_rec_dq[:,1]],
            ["Joint 1 Position (deg)", "Joint 2 Position (deg)", "Joint 1 Velocity (rad/s)", "Joint 2 Velocity (rad/s)"]
        ):
            ax.clear(); ax.set_title(title)
            
            if "Velocity" in title:
                # ax.set_ylim(np.rad2deg((-2, 1)))
                ax.set_ylim(np.rad2deg((-np.pi, np.pi)))
                ax.plot(t, np.rad2deg(data_gt), 'k--', alpha=0.5, label="Ground Truth")
                ax.plot(np.arange(curr_len), np.rad2deg(data_gt[:curr_len]), 'g-', lw=2, label="Observation")
                ax.plot(t, np.rad2deg(data_rec), 'r-', alpha=0.8, label="Prediction")
            else:
                # ax.set_ylim(-70, 120)
                ax.set_ylim(-90, 180)
                ax.plot(t, np.rad2deg(data_gt), 'k--', alpha=0.5, label="Ground Truth")
                ax.plot(np.arange(curr_len), np.rad2deg(data_gt[:curr_len]), 'g-', lw=2, label="Observation")
                ax.plot(t, np.rad2deg(data_rec), 'r-', alpha=0.8, label="Prediction")

            if "Joint 1 Position" in title: ax.legend(loc='lower left', fontsize='x-small')

        for i, ax in enumerate(hist_axes):
            ax.clear(); ax.set_title(COST_KEYS[i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, N_SAMPLES_DIFFUSION)
            
            bins = np.linspace(0, 1, 15)
            ax.hist(w_preds[:,i], bins=bins, color='orange', alpha=0.7, edgecolor='white')
            
            ax.axvline(w_true[i], color='blue', lw=2, ls='--', label='True Weight')
            ax.axvline(w_mean[i], color='red', lw=2, label='Mean Pred')
            
            if i == 0: ax.legend(fontsize='x-small', loc='upper right')

    anim = FuncAnimation(fig, update, frames=range(5, total_len, UPDATE_INTERVAL), interval=200)
    
    save_path = "test_unseen_acados_cartesian.mp4"
    print(f"\nSaving animation to {save_path}...")
    try:
        anim.save(save_path, writer='ffmpeg', fps=5)
    except:
        anim.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=5)
    print("Done.")

if __name__ == "__main__":
    main()