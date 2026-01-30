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

# --- PARAMETRES ---
N_SAMPLES_HIST = 50     # Pour les histogrammes
#N_SAMPLES_HIST = 5
RECONSTRUCTION_STEP = 2 # Fréquence de résolution DOC
SAMPLE_IDX = 100        # Index à tester dans le dataset

# --- ARCHITECTURE (1 Phase) ---
W_DIM = 4           
N_PHASES = 1        
N_COSTS = 4         
INPUT_CHANNELS = 4  
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6

# --- PATHS ---
DATASET_PATH = "data/dataset_unifie_one_phase_cost.pkl" 
CHECKPOINT_DIR = "checkpoints_no_scaling/diff_model_dataset_acados_constant" # /diff_model_dataset_acados_constant"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_600.pth") 
# Fallback si final n'existe pas
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_1.pth")

SCALER_W_PATH = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
SCALER_TRAJ_PATH = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")
PARAM_FILE = "parameters.toml"

COST_ORDER = ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]

# -----------------------------------------------------------------------------
# 1. SETUP EXACT (Basé sur doc_acados_paralell.py)
# -----------------------------------------------------------------------------
def setup_robot_and_param():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        dict_param = parse_params(os.path.join(os.getcwd(), PARAM_FILE))
    except FileNotFoundError:
        dict_param = parse_params(os.path.join(os.getcwd(), "scripts", PARAM_FILE))
    param = convert_to_class(dict_param) 

    # Load Robot
    urdf_name = "human.urdf"
    urdf_path = os.path.join(os.getcwd(), "model/human_urdf/urdf/", urdf_name)
    urdf_meshes_path = os.path.join(os.getcwd(), "model")

    robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer) 
    model = robot.model
    model, collision_model, visual_model, param = build_biomechanical_model(robot, param) 
    
    # --- LOCK JOINTS (Comme dans la génération) ---
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
    
    # --- FRAMES OF INTEREST ---
    param.FOI_to_set = ["right_hand"]
    param.FOI_axes = ["x"]
    
    # Calcul cinématique initiale pour initialiser les structures
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

    # --- COST WEIGHTS CONFIG ---
    param.variables_w = True 
    param.nb_w = 1 # ONE PHASE
    param.weights = {}
    
    # Initialisation basique des poids pour éviter KeyError
    for cost in param.active_costs:
        if cost == "min_joint_torque":
            if hasattr(param, "groups_joint_torques") and param.groups_joint_torques.get("all") == True:
                 param.weights[cost] = 1e-3 * np.ones(param.nb_w)
            else:
                param.weights[cost] = np.ones(param.nb_w)
        else:
            param.weights[cost] = np.ones(param.nb_w)

    # Initialisation target (sera écrasée par la suite mais nécessaire pour la structure)
    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    # Valeur par défaut vue dans doc_acados_parallel.py
    param.weights["target"][0] = 5e3 
    #param.weights["target"][0] = 1e1

    return model_red, param

# -----------------------------------------------------------------------------
# 2. INTERFACE ACADOS (Robustifiée)
# -----------------------------------------------------------------------------
def solve_ocp_interface(W_matrix, model, param_template, q_init, x_fin, nb_samples):
    """
    Adapte les paramètres pour une nouvelle trajectoire et résout.
    """
    # Copie propre pour ne pas modifier le template global
    param = copy.deepcopy(param_template)
    
    # --- MISE A JOUR CRITIQUE DES PARAMETRES TEMPORELS ---
    param.nb_samples = nb_samples
    # Tf doit être cohérent avec le sampling rate (50Hz généralement utilisé dans vos scripts)
    param.Tf = param.nb_samples * (1/50) - (1/50) 
    
    # --- MISE A JOUR ETAT INITIAL / FINAL ---
    param.qdi = q_init
    # On met qdf à 0 ou à la fin de la trajectoire si on la connait, 
    # mais pour la génération pure souvent on laisse libre ou on guide via la Target.
    # Ici on met une valeur neutre car c'est le coût 'target' qui pilote la fin.
    param.qdf = np.array([np.pi/2, 0.0]) 

    # --- MISE A JOUR CIBLE (TARGET) ---
    # On récupère la position actuelle stockée dans param pour garder Y et Z constants
    # et on modifie juste X selon x_fin
    current_pos = param.FOI_position[0].copy() 
    current_pos[0] = x_fin 
    param.FOI_position[0] = current_pos
    
    # La contrainte s'applique au dernier échantillon
    param.FOI_sample = [param.nb_samples] 

    # --- MISE A JOUR DES POIDS ---
    # W_matrix est (1, 4). On extrait les valeurs scalaires.
    for i, cost_name in enumerate(COST_ORDER):
        val = W_matrix[:, i].item()
        param.weights[cost_name] = [val] # Doit être une liste

    # Poids de la cible (HARDCODED comme dans doc_acados_parallel.py)
    param.weights["target"] = [5e3]
    #param.weights["target"] = [1e1] 

    # Nom unique pour éviter les conflits de fichiers JSON générés par Acados
    rand_id = np.random.randint(0, 1000000)
    param.ocp_name = f"test_run_{rand_id}" 
    
    # On force la reconstruction du solver car la taille (nb_samples) change souvent
    param.build_solver = True 

    # Suppression des prints
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        doc = DocHumanMotionGeneration_InvDyn(model, param)
        xs, us, fs = doc.solve_doc_acados(param)
        sys.stdout = old_stdout
        
        # Extraction résultat
        q_sol = xs[:, :model.nq]
        dq_sol = xs[:, model.nq:]
        
        return q_sol, dq_sol, True
    except Exception as e:
        sys.stdout = old_stdout
        # print(f"Solver Error: {e}") 
        return None, None, False

# -----------------------------------------------------------------------------
# 3. UTILS
# -----------------------------------------------------------------------------
def load_training_sample(dataset_path, idx):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    q_traj = data["q_trajs"][idx]      
    dq_traj = data["dq_trajs"][idx]
    w_raw = data["w_matrices"][idx] 
    W_matrix = w_raw.T  # (1, 4)
    params = data["params"][idx]
    
    return q_traj, dq_traj, W_matrix, params

def load_scalers():
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
    # Clip pour éviter les valeurs négatives qui font planter Acados
    w_pred_unscaled = np.clip(w_pred_unscaled, 1e-4, 1.0)
    return w_pred_unscaled

def calculate_rmse(y_true, y_pred):
    min_len = min(y_true.shape[0], y_pred.shape[0])
    diff = y_true[:min_len] - y_pred[:min_len]
    return np.sqrt(np.mean(diff**2))

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
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
    
    try:
        diffusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"\nERREUR CHARGEMENT MODELE: {e}")
        return

    # --- LOAD DATA ---
    print(f"--- Loading Sample {SAMPLE_IDX} ---")
    q_true, dq_true, W_true, params = load_training_sample(DATASET_PATH, SAMPLE_IDX)
    
    # Paramètres extraits du sample pour configurer l'OCP
    q_init_rad = params['q_init']
    x_fin = params['x_fin']
    total_len = q_true.shape[0] # N points = nb_samples + 1 normalement, ou nb_samples
    
    # Acados nb_samples = intervalles. Si q_true a 100 points, il y a 99 intervalles.
    nb_samples_ocp = total_len - 1
    
    print(f"Target X: {x_fin:.3f}, q_init: {np.rad2deg(q_init_rad)}, Samples: {nb_samples_ocp}")

    # --- SETUP VISUALIZATION ---
    fig = plt.figure(figsize=(16, 10))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)
    
    # Trajectories Plots
    gs_top = gs_main[0].subgridspec(2, 2, wspace=0.25, hspace=0.3)
    ax_q1 = fig.add_subplot(gs_top[0, 0])
    ax_q2 = fig.add_subplot(gs_top[0, 1])
    ax_dq1 = fig.add_subplot(gs_top[1, 0])
    ax_dq2 = fig.add_subplot(gs_top[1, 1])
    traj_axes = [ax_q1, ax_q2, ax_dq1, ax_dq2]
    traj_titles = ["q1 (Shoulder)", "q2 (Elbow)", "dq1", "dq2"]
    
    # Weights Plots
    gs_weights = gs_main[1].subgridspec(1, 5, width_ratios=[0.2, 1, 1, 1, 1], wspace=0.3)
    ax_progress = fig.add_subplot(gs_weights[0, 0])
    axes_hist = [fig.add_subplot(gs_weights[0, i+1]) for i in range(4)]

    # Storage
    last_reconstructed_q = np.zeros_like(q_true)
    last_reconstructed_dq = np.zeros_like(dq_true)

    def update(frame):
        nonlocal last_reconstructed_q, last_reconstructed_dq
        
        # On avance par pas plus grands pour que l'animation ne soit pas éternelle
        # frame est l'index dans frames
        current_len = frame + 15
        if current_len > total_len: current_len = total_len
        
        # 1. Prepare Input Condition (Observed Trajectory)
        q_partial = q_true[:current_len]
        dq_partial = dq_true[:current_len]
        combined = np.concatenate([q_partial, dq_partial], axis=1) 
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # 2. Diffusion Inference (Get 50 samples)
        w_pred_samples_flat = sample_diffusion(diffusion_model, traj_tensor, N_SAMPLES_HIST, scaler_w)
        w_pred_samples = w_pred_samples_flat.reshape(N_SAMPLES_HIST, 1, N_COSTS)
        
        # Normalize weights (Sum = 1)
        sums = w_pred_samples.sum(axis=2, keepdims=True) 
        sums[sums == 0] = 1.0 
        w_pred_samples_norm = w_pred_samples / sums 
        
        # Compute Mean Weights for Reconstruction
        w_pred_mean = w_pred_samples_norm.mean(axis=0) # Shape (1, 4)
        
        # 3. OCP Reconstruction (Only Mean)
        status = "."
        if frame % RECONSTRUCTION_STEP == 0 or current_len == total_len:
            # Appel Acados avec les paramètres du sample actuel
            rec_q, rec_dq, ok = solve_ocp_interface(
                w_pred_mean, 
                model, 
                param_template, 
                q_init_rad, 
                x_fin, 
                nb_samples_ocp
            )
            
            if ok:
                status = "OK"
                # On met à jour la courbe reconstruite. 
                # Attention aux dimensions: rec_q peut avoir nb_samples + 1 points.
                L = min(rec_q.shape[0], total_len)
                
                # Reset buffers
                last_reconstructed_q = np.zeros_like(q_true)
                last_reconstructed_dq = np.zeros_like(dq_true)
                
                last_reconstructed_q[:L] = rec_q[:L]
                last_reconstructed_dq[:L] = rec_dq[:L]
            else:
                status = "FAIL"

        print(f"Observed: {current_len}/{total_len} | Solver: {status}", end="\r", flush=True)

        # 4. RMSE
        rmse_q = calculate_rmse(q_true, last_reconstructed_q)

        # --- PLOTTING ---
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
            
            # Zone observée en vert
            ax.axvspan(0, current_len, color='green', alpha=0.1)
            
            # Ground Truth (Noir pointillé)
            ax.plot(time_steps, true_d, 'k--', linewidth=1.5, alpha=0.7, label='Truth')
            
            # Observation (Vert plein)
            ax.plot(time_steps[:current_len], true_d[:current_len], 'g-', linewidth=2, label='Observed')
            
            # Prédiction Moyenne (Rouge plein)
            if np.any(pred_d):
                # On coupe si pred_d est plus long (sécurité)
                L = min(len(time_steps), len(pred_d))
                ax.plot(time_steps[:L], pred_d[:L], 'r-', linewidth=2, label='Mean Pred')
            
            ax.set_title(traj_titles[i])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize='x-small')
                ax.text(0.02, 0.95, f"RMSE: {rmse_q:.4f}", transform=ax.transAxes, color='red', fontweight='bold')

        # --- WEIGHTS PLOTTING ---
        # Progress Bar
        ax_progress.clear()
        p = min(1.0, current_len/total_len)
        ax_progress.barh([0], [p], color='green', height=0.5)
        ax_progress.set_xlim(0, 1); ax_progress.axis('off')
        ax_progress.text(0.5, 0, f"{int(p*100)}%", ha='center', va='center', color='white', fontweight='bold')
        
        # Histograms
        for c in range(N_COSTS):
            ax = axes_hist[c]; ax.clear()
            
            # Vraie valeur (normalisée)
            true_val = W_true[0, c] / W_true[0, :].sum()
            # Valeurs prédites (normalisées)
            pred_vals = w_pred_samples_norm[:, 0, c]
            
            ax.hist(pred_vals, bins=np.linspace(0, 1, 15), color='orange', alpha=0.7)
            ax.axvline(x=true_val, color='black', linestyle='--', linewidth=2, label='True')
            
            ax.set_title(COST_ORDER[c], fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, N_SAMPLES_HIST)
            ax.tick_params(axis='both', which='major', labelsize=7)

    # Lancement de l'animation
    # On génère une frame tous les 2 pas pour aller plus vite
    frames = range(0, total_len - 5, 2) 
    anim = FuncAnimation(fig, update, frames=frames, interval=150)
    
    save_path = f"test_rebuilt_{SAMPLE_IDX}.gif"
    print(f"\n--- Saving to {save_path} ---")
    try:
        anim.save(save_path, writer='pillow', fps=10)
        print("\nDone.")
    except Exception as e:
        print(f"GIF failed: {e}")

if __name__ == "__main__":
    main()
