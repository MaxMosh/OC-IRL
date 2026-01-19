import pickle
import numpy as np
import os
import sys

# Ensure the plotting tools are accessible
sys.path.append(os.getcwd())
from tools.OCP_solving_cpin_new_variable_corr_acc_EN import plot_trajectory_q1, plot_trajectory_q2, plot_trajectory_ee

def plot_from_dataset(filepath, num_samples=20):
    """
    Loads a dataset and plots a subset of trajectories.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    print(f"Loading dataset: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    q_trajs = data["q_trajs"]
    dq_trajs = data["dq_trajs"]
    ddq_trajs = data.get("ddq_trajs", None)
    
    total_available = len(q_trajs)
    print(f"Dataset contains {total_available} samples.")

    # Select random indices
    indices = np.random.choice(total_available, min(num_samples, total_available), replace=False)
    
    subset_q = [q_trajs[i] for i in indices]
    subset_dq = [dq_trajs[i] for i in indices]
    # subset_ddq = [ddq_trajs[i] for i in indices] if ddq_trajs else None
    subset_ddq = None

    # Plotting
    print(f"Plotting {len(indices)} trajectories...")
    
    # Save results in a subfolder to avoid overwriting generation plots
    plot_trajectory_q1(subset_q, subset_dq, subset_ddq, save_path='data/viz_dataset_q1.png')
    plot_trajectory_q2(subset_q, subset_dq, subset_ddq, save_path='data/viz_dataset_q2.png')
    
    # Target X might vary per sample, we use the base target 1.9 for visual reference
    plot_trajectory_ee(subset_q, x_fin_target=1.9, save_path='data/viz_dataset_ee.png')

if __name__ == "__main__":
    # Update this path with your actual generated filename
    DATASET_PATH = 'data/dataset_parallel_299999_samples_WITH_ACC.pkl' 
    
    plot_from_dataset(DATASET_PATH, num_samples=20)