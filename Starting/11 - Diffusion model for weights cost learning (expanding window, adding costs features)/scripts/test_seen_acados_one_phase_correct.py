import os
import sys
import numpy as np
import torch
import joblib
import pinocchio as pin
import matplotlib
matplotlib.use("Agg") # Pour sauvegarder sans fenêtre
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

import pickle

# --- IMPORTS UTILISATEUR ---
# On suppose que le script est à la racine, comme doc_acados_paralell.py
sys.path.append(os.getcwd())

# Imports Acados / Pinocchio utils
from utils.reader_parameters import parse_params, convert_to_class
from utils.model_utils_motif import Robot, build_biomechanical_model 
from utils.doc_utils_new_acados_refactor3 import DocHumanMotionGeneration_InvDyn  

# Import du modèle de diffusion
from tools.diffusion_model_with_angular_velocities_scaled_costs_variable_new_architecture import ConditionalDiffusionModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Paramètres du modèle (doivent matcher l'entraînement)
W_DIM = 4              # 4 poids
INPUT_CHANNELS = 4     # q1, q2, dq1, dq2
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
TIMESTEPS = 1000       # Diffusion steps

# Paramètres de test
# N_SAMPLES_DIFFUSION = 50
N_SAMPLES_DIFFUSION = 20
# UPDATE_INTERVAL = 2    # Prédire tous les 2 samples
UPDATE_INTERVAL = 5
CHECKPOINT_DIR = "checkpoints_no_scaling/diff_model_dataset_acados_constant"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "diff_model_transformer_epoch_50.pth") 
# Ou un epoch précis ex: "diff_model_transformer_epoch_500.pth"

SCALER_W_PATH = os.path.join(CHECKPOINT_DIR, "scaler_w.pkl")
SCALER_TRAJ_PATH = os.path.join(CHECKPOINT_DIR, "scaler_traj.pkl")

# Ordre des coûts (Doit être STRICTEMENT le même que lors de la création du dataset)
# D'après doc_acados_paralell.py :
COST_KEYS = ["min_joint_torque", "min_torque_change", "min_joint_acc", "min_joint_vel"]


# --- 1. SETUP INITIAL (Robots & Params) ---
def setup_robot_and_params():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Gestion du chemin
    if os.path.basename(script_directory) == "scripts":
        parent_directory = os.path.dirname(script_directory)
    else:
        parent_directory = os.path.dirname(os.path.dirname(script_directory))
    
    # 1. Charger le TOML
    toml_path = os.path.join(os.getcwd(), 'parameters.toml') 
    dict_param = parse_params(toml_path)
    param = convert_to_class(dict_param)
    
    # 2. Charger le Robot
    urdf_name = "human.urdf"
    urdf_path = os.path.join(os.getcwd(), "model/human_urdf/urdf/", urdf_name)
    urdf_meshes_path = os.path.join(os.getcwd(), "model")
    
    robot = Robot(urdf_path, urdf_meshes_path, param.free_flyer)
    model, collision_model, visual_model, param = build_biomechanical_model(robot, param)
    
    # 3. Réduction du modèle
    q = np.zeros(model.nq)
    quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs()
    q[3:7] = quat
    
    param.free_flyer = False
    q_lock = q
    joints_to_lock = [model.joints[1].id]
    
    geom_models = [visual_model, collision_model]
    model_red, geom_models_red = pin.buildReducedModel(model, geom_models, joints_to_lock, q_lock)
    
    # --- AJOUTS ET CORRECTIONS ---
    
    # A. Configurer qdi et qdf
    param.qdi = np.array([np.pi/2, np.pi/2]) 
    param.qdf = np.array([np.pi/2, 0])     

    # B. Configurer les Frames (C'est ici que ça bloquait)
    param.FOI_to_set = ["right_hand"]
    param.FOI_axes = ["x"]
    
    # --- CORRECTION : AJOUT DES IDs ---
    param.FOI_to_set_Id = []
    for name in param.FOI_to_set:
        # On récupère l'ID numérique du frame dans le modèle réduit
        param.FOI_to_set_Id.append(model_red.getFrameId(name))

    # C. Calculer la position cible (FOI_position) via cinématique directe
    data_red = model_red.createData()
    pin.forwardKinematics(model_red, data_red, param.qdf)
    pin.updateFramePlacements(model_red, data_red)
    
    param.FOI_position = []
    param.FOI_orientation = [] 
    
    for i in range(len(param.FOI_to_set)):
        frame_id = param.FOI_to_set_Id[i] # On utilise l'ID qu'on vient de trouver
        translation = data_red.oMf[frame_id].translation.copy()
        param.FOI_position.append(translation)
        # On met une orientation dummy pour éviter d'autres erreurs si le code l'appelle
        param.FOI_orientation.append(data_red.oMf[frame_id].rotation.copy())

    # D. Autres paramètres requis par le solveur
    param.nb_samples = 100 
    param.FOI_sample = [param.nb_samples]
    
    # Paramètres liés aux poids variables (même si on ne les utilise pas ici, le solveur peut vérifier leur existence)
    param.variables_w = False
    param.nb_w = 1

    return model_red, param

# --- 2. WRAPPER SOLVEUR ACADOS ---
def solve_ocp_acados(model, base_param, w_vector, total_len, q_init, q_final, foi_pos):
    """
    Configure et résout le DOC Acados de manière robuste.
    """
    param = copy.deepcopy(base_param)
    
    # 1. Configuration Temporelle
    # Acados définit nb_samples comme le nombre d'intervalles
    param.nb_samples = total_len - 1
    param.Tf = param.nb_samples * (1/50)
    
    # 2. Conditions Limites (Assurance type float)
    param.qdi = np.array(q_init, dtype=float).flatten()
    param.qdf = np.array(q_final, dtype=float).flatten()
    
    # Mise à jour de la cible
    param.FOI_position[0] = np.array(foi_pos, dtype=float)
    param.FOI_sample = [param.nb_samples]
    
    # 3. Sécurisation des Poids (CORRECTION MAJEURE)
    w_flat = w_vector.flatten()
    param.weights = {}
    
    # On force les poids prédits à être au moins 1e-2. 
    # Des poids trop faibles (1e-6) rendent le problème mal conditionné pour Acados.
    # w_safe = np.maximum(np.abs(w_flat), 1e-2)
    w_safe = np.maximum(np.abs(w_flat), 1e-6)
    
    param.nb_w = 1
    for i, key in enumerate(COST_KEYS):
        param.weights[key] = np.array([w_safe[i]]) 
    
    # 4. Poids de la Cible
    # On réduit légèrement le poids de la cible (1e3 au lieu de 5e3) pour éviter
    # les gradients explosifs à l'initialisation, tout en gardant la précision.
    nb_targets = len(param.FOI_to_set)
    param.weights["target"] = np.zeros(nb_targets)
    # param.weights["target"][0] = 1000.0 
    param.weights["target"][0] = 5e3
    
    # 5. Compilation
    param.build_solver = True
    # param.build_solver = False
    # Nom unique pour forcer une recompilation propre
    param.ocp_name = f"ocp_{np.random.randint(1000000)}" 
    
    try:
        # On tente la résolution
        doc = DocHumanMotionGeneration_InvDyn(model, param)
        xs, us, fs = doc.solve_doc_acados(param)
        
        # Vérification de la sortie
        if xs is None or len(xs) == 0:
            return None, None
            
        return xs[:, :model.nq], xs[:, model.nq:] 
        
    except Exception as e:
        # En cas d'erreur, on l'affiche mais on permet au script de continuer
        print(f"Solver Exception: {e}")
        return None, None

# --- 3. FONCTIONS DIFFUSION ---
def load_scalers():
    print("Loading scalers...")
    if not os.path.exists(SCALER_W_PATH):
        raise FileNotFoundError("Scaler W not found")
    scaler_w = joblib.load(SCALER_W_PATH)
    scaler_traj = joblib.load(SCALER_TRAJ_PATH)
    return scaler_traj, scaler_w

def sample_diffusion_batch(model, condition_traj, n_samples, scaler_w):
    """
    Génère n_samples vecteurs de poids.
    condition_traj: Tensor (1, 4, Length)
    """
    model.eval()
    beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    seq_len = condition_traj.shape[2]
    # Mask: False partout (pas de padding car batch unique répliqué)
    mask = torch.zeros((n_samples, seq_len), dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        # Dupliquer la condition pour le batch
        cond_repeated = condition_traj.repeat(n_samples, 1, 1)
        
        # Bruit initial Gaussien (Normalisé)
        w_current = torch.randn(n_samples, W_DIM).to(DEVICE)

        # Boucle de diffusion inverse
        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)
            
            predicted_noise = model(w_current, t, cond_repeated, trajectory_mask=mask)
            
            alpha_t = alpha[i]
            alpha_hat_t = alpha_hat[i]
            beta_t = beta[i]
            
            if i > 0:
                noise = torch.randn_like(w_current)
            else:
                noise = torch.zeros_like(w_current)
                
            # Équation standard DDPM
            w_current = (1 / torch.sqrt(alpha_t)) * (
                w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    w_pred_numpy = w_current.cpu().numpy() # (N_samples, 4)
    
    # Inverse Scaling (retour vers l'espace réel)
    w_pred_unscaled = scaler_w.inverse_transform(w_pred_numpy)
    
    # Les poids doivent être positifs
    w_pred_unscaled = np.abs(w_pred_unscaled) # ou np.clip(w_pred_unscaled, 1e-6, None)
    
    return w_pred_unscaled

# --- 4. MAIN ---
def main():
    # 1. Init
    model_pin, param = setup_robot_and_params()
    # On crée data_pin pour pouvoir calculer la target du mouvement chargé
    data_pin = model_pin.createData() 
    
    scaler_traj, scaler_w = load_scalers()
    
    # 2. Load Network
    diff_model = ConditionalDiffusionModel(
        w_dim=W_DIM, 
        input_channels=INPUT_CHANNELS, 
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        diff_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded form {MODEL_PATH}")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # --- MODIFICATION ICI : CHARGEMENT DU DATASET ---
    dataset_path = "data/dataset_unifie_one_phase_cost.pkl"
    print(f"Loading Ground Truth from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Structure supposée : data['q_trajs'], data['dq_trajs'], data['w_matrices']
    # On tire un index au hasard
    idx = np.random.randint(0, len(data['q_trajs']))
    print(f"Selected sample index: {idx}")

    # Récupération des données
    q_gt = data['q_trajs'][idx]       # (N, 2)
    dq_gt = data['dq_trajs'][idx]     # (N, 2)
    w_true = data['w_matrices'][idx]  # (4,) ou (1, 4)
    w_true = np.array(w_true).flatten() # Sécurité pour avoir (4,)

    # --- CRUCIAL : RETROUVER LA CIBLE (TARGET) ---
    # Le solveur a besoin de savoir où aller pour la reconstruction.
    # On suppose que la cible est la position de la main à la fin de la trajectoire chargée.
    q_final_gt = q_gt[-1]
    
    # Calcul FK
    pin.forwardKinematics(model_pin, data_pin, q_final_gt)
    pin.updateFramePlacements(model_pin, data_pin)
    
    # Récupération de l'ID de la main (défini dans setup_robot_and_params)
    hand_frame_id = param.FOI_to_set_Id[0] 
    target_pos = data_pin.oMf[hand_frame_id].translation.copy()
    
    print(f"Ground Truth Loaded. Length: {len(q_gt)}")
    print(f"True Weights: {w_true}")
    print(f"Inferred Target Pos: {target_pos}")
    
    # Paramètres initiaux pour le solveur (début du mouvement)
    q_init = q_gt[0] 
    q_final = q_final_gt # Configuration finale articulaire (optionnelle pour Acados si target défini, mais bonne pratique)

    # -----------------------------------------------
    
    # 4. Préparation Graphique
    total_len = len(q_gt)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1]) 
    
    # Plots Trajectoires (Haut)
    ax_q1 = fig.add_subplot(gs[0, 0])
    ax_q2 = fig.add_subplot(gs[0, 1])
    ax_dq1 = fig.add_subplot(gs[0, 2])
    ax_dq2 = fig.add_subplot(gs[0, 3])
    traj_axes = [ax_q1, ax_q2, ax_dq1, ax_dq2]
    titles = ["q1 (deg)", "q2 (deg)", "dq1 (rad/s)", "dq2 (rad/s)"]
    
    # Plots Histogrammes (Bas)
    hist_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    
    # Initialiser avec des NaN pour voir clairement si ça ne marche pas (le plot sera vide au lieu de 0)
    last_rec_q = np.full_like(q_gt, np.nan) 
    last_rec_dq = np.full_like(dq_gt, np.nan)
    last_w_preds = np.zeros((N_SAMPLES_DIFFUSION, 4))
    
    def update(frame_idx):
        nonlocal last_rec_q, last_rec_dq, last_w_preds
        
        current_len = frame_idx + 10 
        if current_len >= total_len: current_len = total_len
        
        # --- A. INFERENCE (Identique) ---
        q_obs = q_gt[:current_len]
        dq_obs = dq_gt[:current_len]
        combined = np.concatenate([q_obs, dq_obs], axis=1) 
        combined_scaled = scaler_traj.transform(combined)
        traj_tensor = torch.FloatTensor(combined_scaled).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        w_preds = sample_diffusion_batch(diff_model, traj_tensor, N_SAMPLES_DIFFUSION, scaler_w)
        last_w_preds = w_preds 
        w_mean = np.mean(w_preds, axis=0)
        
        # --- B. RESOLUTION OCP (Corrigée) ---
        print(f"Solving OCP at {current_len}/{total_len}...", end="\r")
        
        # Appel avec total_len (le solveur calculera nb_samples = total_len - 1)
        rec_q, rec_dq = solve_ocp_acados(model_pin, param, w_mean, total_len, q_init, q_final, target_pos)
        
        if rec_q is not None:
            # Vérification de sécurité sur la taille
            if len(rec_q) != total_len:
                # Si Acados retourne N+1 points et qu'on a un décalage de 1 frame
                # On tronque ou on pad selon le besoin. 
                # Habituellement, si param.nb_samples = total_len - 1, on a total_len points.
                # Si jamais ça diffère, on force la taille :
                min_len = min(len(rec_q), total_len)
                last_rec_q[:min_len] = rec_q[:min_len]
                last_rec_dq[:min_len] = rec_dq[:min_len]
            else:
                last_rec_q = rec_q
                last_rec_dq = rec_dq
        else:
            print(f"\nSkipping frame {current_len}: Solver returned None")
        
        # --- C. PLOTTING ---
        time_vec = np.arange(total_len)
        
        # 1. Trajectoires
        # Q1
        ax_q1.clear(); ax_q1.set_title(titles[0])
        ax_q1.plot(time_vec, np.rad2deg(q_gt[:,0]), 'k--', alpha=0.6, label="GT")
        ax_q1.plot(np.arange(current_len), np.rad2deg(q_gt[:current_len,0]), 'g-', lw=2, label="Obs")
        ax_q1.plot(time_vec, np.rad2deg(last_rec_q[:,0]), 'r-', alpha=0.8, label="Pred")
        ax_q1.set_ylim(-70, 15)
        ax_q1.legend()
        
        # Q2
        ax_q2.clear(); ax_q2.set_title(titles[1])
        ax_q2.plot(time_vec, np.rad2deg(q_gt[:,1]), 'k--', alpha=0.6)
        ax_q2.plot(np.arange(current_len), np.rad2deg(q_gt[:current_len,1]), 'g-', lw=2)
        ax_q2.plot(time_vec, np.rad2deg(last_rec_q[:,1]), 'r-', alpha=0.8)
        ax_q2.set_ylim(-70, 15)

        # DQ1
        ax_dq1.clear(); ax_dq1.set_title(titles[2])
        ax_dq1.plot(time_vec, dq_gt[:,0], 'k--', alpha=0.6)
        ax_dq1.plot(np.arange(current_len), dq_gt[:current_len,0], 'g-', lw=2)
        ax_dq1.plot(time_vec, last_rec_dq[:,0], 'r-', alpha=0.8)
        ax_dq1.set_ylim(-2, 1)
        
        # DQ2
        ax_dq2.clear(); ax_dq2.set_title(titles[3])
        ax_dq2.plot(time_vec, dq_gt[:,1], 'k--', alpha=0.6)
        ax_dq2.plot(np.arange(current_len), dq_gt[:current_len,1], 'g-', lw=2)
        ax_dq2.plot(time_vec, last_rec_dq[:,1], 'r-', alpha=0.8)
        ax_dq2.set_ylim(-2, 1)
        
        # 2. Histogrammes Poids
        for i, ax in enumerate(hist_axes):
            ax.clear()
            cost_name = COST_KEYS[i]
            vals = w_preds[:, i]
            true_val = w_true[i]
            
            ax.hist(vals, bins=15, color='orange', alpha=0.7, edgecolor='k')
            ax.axvline(true_val, color='blue', lw=3, ls='--', label='True')
            ax.axvline(np.mean(vals), color='red', lw=2, label='Mean')

            ax.set_xlim(0, 1) # Poids toujours entre 0 et 1
            ax.set_ylim(0, N_SAMPLES_DIFFUSION) # Y max = nombre total d'échantillons
            
            ax.set_title(cost_name, fontsize=8)
            if i == 0: ax.legend(fontsize='x-small')

    # Animation
    frames = range(5, total_len, UPDATE_INTERVAL)
    
    anim = FuncAnimation(fig, update, frames=frames, interval=200)
    
    save_name = "test_acados_loaded_traj.mp4"
    print(f"\nSaving animation to {save_name}...")
    try:
        anim.save(save_name, writer='ffmpeg', fps=5, dpi=100)
        print("Done.")
    except Exception as e:
        print(f"Error saving MP4: {e}. Trying GIF...")
        anim.save(save_name.replace(".mp4", ".gif"), writer='pillow', fps=5)

if __name__ == "__main__":
    main()