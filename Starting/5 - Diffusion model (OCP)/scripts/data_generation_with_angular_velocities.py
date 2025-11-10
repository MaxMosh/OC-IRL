import os
import sys

sys.path.append(os.path.abspath("Starting/5 - Diffusion model"))

from tools.steps_two_arms_robot_cpin import generate_DOC_solutions, w_true, plot_trajectory_q1, plot_trajectory_q2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generating DOC solutions using different values of Px_f
x_fin_values = np.linspace(0.0, -1.0, 101)                                                  # MODIFIED: INITIALLY 11 FOR THE 3RD ARGUMENT
x_fin_values_test = np.array([np.random.uniform(0.0, -1.0)])   
# vec_limit_1 = np.array([np.pi/8, np.pi/4])
# vec_limit_2 = np.array([-np.pi/8, -np.pi/4])
# q_init_values = np.linspace(vec_limit_1, vec_limit_2, 11)
q_init_values = np.array([0, np.pi/4])
results_angles, results_velocities = generate_DOC_solutions(w_true, x_fin_values, list_q_init=q_init_values)
results_angles_test, results_velocities_test = generate_DOC_solutions(w_true, x_fin_values_test, list_q_init=q_init_values)

plot_trajectory_q1(results_angles, results_velocities)
plot_trajectory_q1(results_angles_test, results_velocities_test, linestyle="--")
plt.show()

plot_trajectory_q2(results_angles, results_velocities)
plot_trajectory_q2(results_angles_test, results_velocities_test, linestyle="--")
plt.show()


print(f"Number of initial trajectories generated: {len(results_angles)}")
print(f"Shape of angles input: {np.array(results_angles[0]).shape}")
print(f"Shape of angular velocities input: {np.array(results_velocities[0]).shape}")

# Building the train dataset
data = []
for q, dq in zip(results_angles, results_velocities):
    q = np.array(q)    # shape (N, 2) - angles [q1, q2]
    dq = np.array(dq)  # shape (N, 2) - angular velocities [dq1, dq2]
    
    # Concatenate angles and angular velocities: shape (N, 4) - [q1, q2, dq1, dq2]
    state = np.concatenate([q, dq], axis=1)
    
    N = len(state)
    
    # Subsequences of different sizes
    for k in range(10, N-10, 5):  # keeping a minimum of 10 time steps, evolving 5 time steps per 5 time steps
        prefix = state[:k]
        future = state[k:]
        data.append({
            "prefix": prefix,
            "future": future
        })

print(f"\nTotal number of subsenquences in the whole dataset: {len(data)}")


# Building the test dataset
# data_test = []
# for q, dq in zip(results_angles_test, results_velocities_test):
#     q = np.array(q)    # shape (N, 2) - angles [q1, q2]
#     dq = np.array(dq)  # shape (N, 2) - angular velocities [dq1, dq2]
    
#     # Concatenate angles and angular velocities: shape (N, 4) - [q1, q2, dq1, dq2]
#     state = np.concatenate([q, dq], axis=1)
    
#     N = len(state)
    
#     # Subsequences of different sizes
#     for k in range(10, N-10, 5):  # keeping a minimum of 10 time steps, evolving 5 time steps per 5 time steps
#         prefix = state[:k]
#         future = state[k:]
#         data_test.append({
#             "prefix": prefix,
#             "future": future
#         })

# print(f"\nTotal number of subsenquences in the test dataset: {len(data_test)}")


np.save("Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities.npy", 
        data, allow_pickle=True)

# np.save("Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_test.npy", 
#         data_test, allow_pickle=True)


# Splitting into a train and a test dataset
# TODO: CHANGE THE train_test_split, TRAJECTORIES HAVE TO BE INDEPENDENT BETWEEN THE TWO SETS
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Savings train and val dataset
np.save("Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_train.npy", 
        train_data, allow_pickle=True)
np.save("Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_val.npy", 
        val_data, allow_pickle=True)