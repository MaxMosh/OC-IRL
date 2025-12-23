import numpy as np
import json
import multiprocessing
from functools import partial
import os
import pickle

# Import des outils
# Assurez-vous que le fichier s'appelle bien OCP_solving_cpin_new_scaled_costs.py
# et qu'il est dans le dossier tools
from tools.OCP_solving_cpin_new_scaled_costs_variables import solve_DOC
from tools.OCP_solving_cpin_new_scaled_costs_variables import compute_scaling_factors
from tools.OCP_solving_cpin_new_scaled_costs_variables import plot_trajectory_q1, plot_trajectory_q2, plot_trajectory_ee

# --- CONFIGURATION ---
NUM_SAMPLES = 1000  # Taille du dataset
FREQ = 100.0
NUM_CORES = multiprocessing.cpu_count() - 2  # On laisse 2 coeurs libres pour le système

# Constantes de génération
Q_INIT_BASES_DEG = [
    [-90, 90],
    [-15, 105],
    [-115, 115]
]
Q_INIT_NOISE_STD_DEG = 7.0
X_FIN_BASE = 1.9
X_FIN_NOISE_STD = 0.01

# --- FONCTION WORKER (Exécutée sur chaque coeur) ---
def generate_single_sample(seed, scale_factors):
    """
    Cette fonction tourne dans un processus séparé.
    Elle doit être autonome.
    """
    # 1. Initialiser l'aléatoire avec un seed unique pour ce processus
    np.random.seed(seed)
    
    # 2. Randomiser N (Durée)
    # 0.8s à 1.0s (attention, vous aviez dit 2.0s avant, ici je remets 1.0s max selon votre texte "0.8s et 1s")
    # Si vous voulez 0.8 à 1.0s à 100Hz -> 80 à 100 points
    # Si vous voulez 0.8 à 2.0s -> 80 à 200 points
    N_steps = np.random.randint(80, 201) 
    
    # 3. Randomiser q_init
    base_idx = np.random.randint(0, len(Q_INIT_BASES_DEG))
    q_base = np.array(Q_INIT_BASES_DEG[base_idx])
    noise_q = np.random.normal(0, Q_INIT_NOISE_STD_DEG, size=2)
    q_init_deg = q_base + noise_q
    q_init_rad = np.deg2rad(q_init_deg)
    
    # 4. Randomiser x_fin
    noise_x = np.random.normal(0, X_FIN_NOISE_STD)
    x_fin = X_FIN_BASE + noise_x
    
    # 5. Randomiser Poids (3 sets par trajectoire)
    w_matrix = np.random.rand(5, 3)
    w_matrix = w_matrix / w_matrix.sum(axis=0) # Normalisation colonnes
    
    # 6. Résolution
    # Note : Le try/except est déjà géré un peu dans solve_DOC mais on sécurise ici
    try:
        res_q, res_dq = solve_DOC(
            w_matrix, 
            N_steps, 
            x_fin=x_fin, 
            q_init=q_init_rad, 
            scale_factors=scale_factors
        )
        
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
    # Important : Le code multiprocessing doit être sous "if __name__ == '__main__':"
    
    # 1. Calibrer les facteurs d'échelle (Séquentiel, rapide)
    print("Calibration des facteurs d'échelle...")
    q_init_calib = np.deg2rad(Q_INIT_BASES_DEG[0])
    scale_factors = compute_scaling_factors(num_samples=20, x_fin=X_FIN_BASE, q_init=q_init_calib)
    
    with open(f'data/scale_factors_parallel.json', 'w') as f:
        json.dump(scale_factors, f)

    # 2. Préparer les arguments pour la parallélisation
    # On crée une liste de seeds uniques
    seeds = [np.random.randint(0, 1000000) for _ in range(NUM_SAMPLES)]
    
    print(f"\nLancement de la génération parallélisée sur {NUM_CORES} coeurs.")
    print(f"Objectif : {NUM_SAMPLES} échantillons.")

    # Listes pour stocker les résultats
    list_results_angles = []
    list_results_angular_velocities = []
    list_w_matrices = []
    list_parameters = []
    valid_count = 0

    # 3. Exécution Parallèle
    # On utilise Pool.imap_unordered pour récupérer les résultats dès qu'ils finissent
    # partial permet de fixer l'argument scale_factors pour tous les workers
    worker_func = partial(generate_single_sample, scale_factors=scale_factors)

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        # Utilisation de tqdm si disponible pour la barre de progression, sinon print simple
        try:
            from tqdm import tqdm
            iterator = tqdm(pool.imap_unordered(worker_func, seeds), total=NUM_SAMPLES)
        except ImportError:
            print("Astuce: installez 'tqdm' (pip install tqdm) pour une barre de progression jolie.")
            iterator = pool.imap_unordered(worker_func, seeds)

        for result in iterator:
            if result["status"] == "success":
                list_results_angles.append(result["q"])
                list_results_angular_velocities.append(result["dq"])
                list_w_matrices.append(result["w_matrix"])
                list_parameters.append(result["params"])
                valid_count += 1
            
            # Feedback minimal si pas de tqdm
            if 'tqdm' not in locals() and valid_count % 10 == 0:
                print(f"\rSuccès : {valid_count}/{NUM_SAMPLES}", end="")

    print(f"\n\nGénération terminée. {valid_count}/{NUM_SAMPLES} valides.")

    # --- SAUVEGARDE ---
    data_dict = {
        "w_matrices": list_w_matrices,
        "q_trajs": list_results_angles,
        "dq_trajs": list_results_angular_velocities,
        "params": list_parameters
    }

    suffix = f"parallel_{valid_count}_samples"
    filepath = f'data/dataset_{suffix}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dataset sauvegardé dans : {filepath}")

    # --- VISUALISATION ---
    if valid_count > 0:
        print("Affichage des 10 premiers résultats...")
        subset_q = list_results_angles[:10]
        subset_dq = list_results_angular_velocities[:10]
        
        plot_trajectory_q1(subset_q, subset_dq)
        plot_trajectory_q2(subset_q, subset_dq)
        plot_trajectory_ee(subset_q, x_fin_target=X_FIN_BASE)