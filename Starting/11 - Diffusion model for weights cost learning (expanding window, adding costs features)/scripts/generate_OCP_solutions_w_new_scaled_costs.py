import numpy as np
import matplotlib.pyplot as plt
import json
from simplex_grid import simplex_grid

# Import from the updated solver file
from tools.OCP_solving_cpin_new_scaled_costs import dq_max_lim_deg_par_s
from tools.OCP_solving_cpin_new_scaled_costs import generate_DOC_solutions_list_w
from tools.OCP_solving_cpin_new_scaled_costs import compute_scaling_factors
from tools.OCP_solving_cpin_new_scaled_costs import plot_trajectory_q1, plot_trajectory_q2
from tools.OCP_solving_cpin_new_scaled_costs import plot_trajectory_dq1, plot_trajectory_dq2

# Constants
x_fin = 1.9
q_init = np.array([-np.pi/2, np.pi/2])
# N_grid_w = 21 

# Generate weight vectors for 5-objective optimization using Simplex
m = 5  # number of objectives
r = 21  # resolution (higher = more points)

list_w = simplex_grid(m=m, r=r)
array_w = np.array(list_w)

# --- CRITICAL STEP: COMPUTE SCALING FACTORS ---
# This ensures that physical units (Torque vs Velocity) don't bias the optimization
# We perform this calibration once before generating the dataset.
scale_factors = compute_scaling_factors(num_samples=20, x_fin=x_fin, q_init=q_init)

# Save scale factors for later reference (optional but recommended)
with open(f'data/scale_factors_simplex_{r}.json', 'w') as f:
    json.dump(scale_factors, f)

# Checking the sum of w components (should equal to 1 for each combination)
card_w = array_w.shape[0]
print(f"Checking simplex validity: First element sum = {array_w[0,:].sum()}")


# Generate Solutions using the computed Scale Factors
list_results_angles, list_results_angular_velocities, array_w = generate_DOC_solutions_list_w(
    array_w, 
    x_fin, 
    q_init,
    scale_factors=scale_factors # Passing the normalization factors
)

# Plotting
plot_trajectory_q1(list_results_angles, list_results_angular_velocities, linestyle = '-')
plot_trajectory_q2(list_results_angles, list_results_angular_velocities, linestyle = '-')
plot_trajectory_dq1(list_results_angles, list_results_angular_velocities, linestyle = '-')
plot_trajectory_dq2(list_results_angles, list_results_angular_velocities, linestyle = '-')

array_results_angles = np.array(list_results_angles)
array_results_angular_velocities = np.array(list_results_angular_velocities)

print(f"w array shape: {array_w.shape}")
print(f"Angles array shape: {array_results_angles.shape}")

# Saving trajectories and associated w in npy files
print(array_w.shape[0])
print(array_results_angles.shape[0])
if (array_w.shape[0] == array_results_angles.shape[0]):
    suffix = f"simplex_{r}_lim_joint_velocities_{dq_max_lim_deg_par_s}_scaled_costs"
    np.save(f'data/array_w_{suffix}.npy', array_w)
    np.save(f'data/array_results_angles_{suffix}.npy', array_results_angles)
    np.save(f'data/array_results_angular_velocities_{suffix}.npy', array_results_angular_velocities)
    print("Data saved successfully.")
else:
    print("Error: Mismatch in data shapes, saving aborted.")