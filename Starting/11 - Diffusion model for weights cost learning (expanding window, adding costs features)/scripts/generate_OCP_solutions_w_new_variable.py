import numpy as np
import json
import multiprocessing
import os
import pickle
import sys

# Import des outils mis à jour
sys.path.append(os.getcwd())
from tools.OCP_solving_cpin_new_variable import solve_DOC
from tools.OCP_solving_cpin_new_variable import plot_trajectory_q1, plot_trajectory_q2, plot_trajectory_ee

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
    Plus d'argument scale_factors ici.
    """
    np.random.seed(seed)
    
    # Randomisation
    N_steps = np.random.randint(80, 201) 
    base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
    q_init_rad = np.deg2rad(np.array(Q_INIT_BASES_DEG[base_idx]) + np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2))
    x_fin = X_FIN_BASE + np.random.normal(0, X_FIN_NOISE_STD)
    
    # Poids (Normalisés pour sommer à 1, mais appliqués directement aux coûts bruts)
    w_matrix = np.random.rand(5, 3)
    w_matrix = w_matrix / w_matrix.sum(axis=0) 
    
    try:
        # Appel sans scale_factors
        res_q, res_dq = solve_DOC(w_matrix, N_steps, x_fin=x_fin, q_init=q_init_rad, verbose=False)
        
        if res_q is not None:
            return {
                "status": "success",
                "w_matrix": w_matrix,
                "q": res_q,
                "dq": res_dq,
                "params": {"N": N_steps, "q_init": q_init_rad, "x_fin": x_fin}
            }
        else:
            return {"status": "failed"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# --- MAIN ---
if __name__ == "__main__":
    # Pas de calibration ici
    
    seeds = [np.random.randint(0, 1000000) for _ in range(NUM_SAMPLES)]
    
    print(f"\nLancement de la génération (SANS SCALING FACTORS) sur {NUM_CORES} coeurs.")
    print(f"Objectif : {NUM_SAMPLES} échantillons.")

    list_results_angles = []
    list_results_angular_velocities = []
    list_w_matrices = []
    list_parameters = []
    valid_count = 0

    # Plus besoin de partial car un seul argument variable
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
                list_w_matrices.append(result["w_matrix"])
                list_parameters.append(result["params"])
                valid_count += 1
            
            if 'tqdm' not in locals() and valid_count % 100 == 0:
                print(f"\rSuccès : {valid_count}/{NUM_SAMPLES}", end="")

    print(f"\n\nGénération terminée. {valid_count}/{NUM_SAMPLES} valides.")

    data_dict = {
        "w_matrices": list_w_matrices,
        "q_trajs": list_results_angles,
        "dq_trajs": list_results_angular_velocities,
        "params": list_parameters
    }

    suffix = f"parallel_{valid_count}_samples_NO_SCALING"
    filepath = f'data/dataset_{suffix}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dataset sauvegardé dans : {filepath}")

    if valid_count > 0:
        subset_q = list_results_angles[:10]
        subset_dq = list_results_angular_velocities[:10]
        plot_trajectory_q1(subset_q, subset_dq)
        plot_trajectory_q2(subset_q, subset_dq)
        plot_trajectory_ee(subset_q, x_fin_target=X_FIN_BASE)