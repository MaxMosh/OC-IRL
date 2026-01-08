import numpy as np
import json
import multiprocessing
import os
import pickle
import sys

# Import des outils mis à jour
# Assurez-vous que le fichier OCP_solving_cpin_new_variable.py contient bien la version modifiée avec ddq
sys.path.append(os.getcwd())
from tools.OCP_solving_cpin_new_variable_corr_acc import solve_DOC
from tools.OCP_solving_cpin_new_variable_corr_acc import plot_trajectory_q1, plot_trajectory_q2, plot_trajectory_ee

# --- CONFIGURATION ---
NUM_SAMPLES = 300000
FREQ = 100.0
NUM_CORES = multiprocessing.cpu_count() - 2 

# Constantes de génération
Q_INIT_BASES_DEG = [[-90, 90], [-15, 105], [-115, 115]]
Q_INIT_NOISE_STD_DEG = 7.0
X_FIN_BASE = 1.9
X_FIN_NOISE_STD = 0.01

# --- FONCTION WORKER ---
def generate_single_sample(seed):
    """
    Génération d'un échantillon.
    Mise à jour : Récupération de l'accélération (ddq)
    """
    np.random.seed(seed)
    
    # Randomisation
    N_steps = np.random.randint(80, 201) 
    base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
    q_init_rad = np.deg2rad(np.array(Q_INIT_BASES_DEG[base_idx]) + np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2))
    x_fin = X_FIN_BASE + np.random.normal(0, X_FIN_NOISE_STD)
    
    # Poids (Normalisés pour sommer à 1)
    w_matrix = np.random.rand(5, 3)
    w_matrix = w_matrix / w_matrix.sum(axis=0) 
    
    try:
        # Appel du solveur (Récupération de 3 variables maintenant)
        # res_ddq contient l'accélération optimale
        res_q, res_dq, res_ddq = solve_DOC(w_matrix, N_steps, x_fin=x_fin, q_init=q_init_rad, verbose=False)
        
        if res_q is not None:
            return {
                "status": "success",
                "w_matrix": w_matrix,
                "q": res_q,
                "dq": res_dq,
                "ddq": res_ddq,  # Ajout de l'accélération dans le résultat
                "params": {"N": N_steps, "q_init": q_init_rad, "x_fin": x_fin}
            }
        else:
            return {"status": "failed"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    
    seeds = [np.random.randint(0, 1000000) for _ in range(NUM_SAMPLES)]
    
    print(f"\nLancement de la génération (AVEC ACCELERATIONS) sur {NUM_CORES} coeurs.")
    print(f"Objectif : {NUM_SAMPLES} échantillons.")

    list_results_angles = []
    list_results_angular_velocities = []
    list_results_accelerations = []  # Nouvelle liste pour stocker ddq
    list_w_matrices = []
    list_parameters = []
    valid_count = 0

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        try:
            from tqdm import tqdm
            iterator = tqdm(pool.imap_unordered(generate_single_sample, seeds), total=NUM_SAMPLES)
        except ImportError:
            print("Installez tqdm pour une barre de progression.")
            iterator = pool.imap_unordered(generate_single_sample, seeds)

        for result in iterator:
            if result["status"] == "success":
                list_results_angles.append(result["q"])
                list_results_angular_velocities.append(result["dq"])
                list_results_accelerations.append(result["ddq"]) # Stockage
                list_w_matrices.append(result["w_matrix"])
                list_parameters.append(result["params"])
                valid_count += 1
            
            if 'tqdm' not in locals() and valid_count % 100 == 0:
                print(f"\rSuccès : {valid_count}/{NUM_SAMPLES}", end="")

    print(f"\n\nGénération terminée. {valid_count}/{NUM_SAMPLES} valides.")

    # Construction du dictionnaire final
    data_dict = {
        "w_matrices": list_w_matrices,
        "q_trajs": list_results_angles,
        "dq_trajs": list_results_angular_velocities,
        "ddq_trajs": list_results_accelerations, # Ajout au dataset
        "params": list_parameters
    }

    suffix = f"parallel_{valid_count}_samples_WITH_ACC"
    filepath = f'data/dataset_{suffix}.pkl'
    
    # Création du dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dataset sauvegardé dans : {filepath}")

    if valid_count > 0:
        subset_q = list_results_angles[:10]
        subset_dq = list_results_angular_velocities[:10]
        subset_ddq = list_results_accelerations[:10] # Nouvelle liste
        
        # Passage du nouvel argument
        plot_trajectory_q1(subset_q, subset_dq, subset_ddq)
        plot_trajectory_q2(subset_q, subset_dq, subset_ddq)
        plot_trajectory_ee(subset_q, x_fin_target=X_FIN_BASE)