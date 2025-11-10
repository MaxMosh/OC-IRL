import os
import sys

sys.path.append(os.path.abspath("Starting/5 - Diffusion model"))

from tools.steps_two_arms_robot_cpin import generate_DOC_solutions, w_true
import numpy as np

# Generating DOC solutions using different values of Px_f
x_fin_values = np.linspace(0.0, -1.0, 101)
x_fin_values_test = np.array([np.random.uniform(0.0, -1.0)])  
# vec_limit_1 = np.array([np.pi/8, np.pi/4])
# vec_limit_2 = np.array([-np.pi/8, -np.pi/4])
# q_init_values = np.linspace(vec_limit_1, vec_limit_2, 101)
q_init_values = np.array([0, np.pi/4])
results_angles, results_velocities = generate_DOC_solutions(w_true, x_fin_values, list_q_init=q_init_values)
results_angles_test, results_velocities_test = generate_DOC_solutions(w_true, x_fin_values_test, list_q_init=q_init_values)

# Building the train dataset
data_train = []
for q in results_angles:
    q = np.array(q)  # shape (N, 2)
    N = len(q)
    # Subsequences of different sizes
    for k in range(10, N-10, 5):  # keeping a minimum of 10 time steps, evolving 5 time steps per 5 time steps
        prefix = q[:k]
        future = q[k:]
        data_train.append({
            "prefix": prefix,
            "future": future
        })

np.save("Starting/5 - Diffusion model/data/trajectories_dataset_train.npy", data_train, allow_pickle=True)


# Building the test dataset
data_test = []
for q in results_angles_test:
    q = np.array(q)  # shape (N, 2)
    N = len(q)
    # Subsequences of different sizes
    for k in range(10, N-10, 5):  # keeping a minimum of 10 time steps, evolving 5 time steps per 5 time steps
        prefix = q[:k]
        future = q[k:]
        data_test.append({
            "prefix": prefix,
            "future": future
        })

np.save("Starting/5 - Diffusion model/data/trajectories_dataset_test.npy", data_test, allow_pickle=True)