import numpy as np
import matplotlib.pyplot as plt
from tools.OCP_solving_cpin import solve_DOC
from tools.OCP_solving_cpin import generate_DOC_solutions_list_w, plot_trajectory_q1, plot_trajectory_q2


# Constants
x_fin = -1.0
q_init = np.array([0, np.pi/4])
# N_grid_w = 21  # Reduced from 101 to keep total samples reasonable for 5D simplex
# N_grid_w = 31
#N_trajectories = 10000

# Creating of a list of w (all should have a norm 1 equal to 1)
# list_w = []
# for i1 in range(N_grid_w):
#     if ((i1 > N_grid_w/20) and (i1 < N_grid_w - 1)):
#         temp_range_i2 = N_grid_w - 1 - i1
#         for i2 in range(temp_range_i2):
#             if i2 > N_grid_w/20:
#                 temp_range_i3 = N_grid_w - 1 - i1 - i2
#                 for i3 in range(temp_range_i3):
#                     temp_range_i4 = N_grid_w - 1 - i1 - i2 - i3
#                     for i4 in range(temp_range_i4):
#                         i5 = N_grid_w - 1 - i1 - i2 - i3 - i4
#                         w_current = [
#                             (1/(N_grid_w-1))*i1, 
#                             (1/(N_grid_w-1))*i2, 
#                             (1/(N_grid_w-1))*i3,
#                             (1/(N_grid_w-1))*i4,
#                             (1/(N_grid_w-1))*i5
#                         ]
#                         # NOTE: AS MENTIONED BELOW, 0 COST WEIGHTS OVER ANGULAR VELOCITIES GIVE A BAD OUTPUT
#                         if not(i1 == 0 and i2 == 0):
#                             list_w.append(w_current)
# array_w = np.array(list_w)
# for i1 in range(N_grid_w):
#     i2 = N_grid_w - i1
#     w_current = [(1/(N_grid_w-1))*i1, (1/(N_grid_w-1))*i2]
#     if (i1 > N_grid_w/20 and i2 > N_grid_w/20):
#         list_w.append(w_current)
# array_w = np.array(list_w)
# list_w = []
# for i in range(N_trajectories):
#     w_non_normalized = np.random.random(5)
#     w_normalized = w_non_normalized/w_non_normalized.sum()
#     list_w.append(w_normalized)
# array_w = np.array(list_w)
# for i in range(N_trajectories):
#     w_non_normalized = np.random.random(5)
#     # w_normalized = w_non_normalized/w_non_normalized.sum()
#     list_w.append(w_non_normalized)
# array_w = np.array(list_w)
# NOTE: constitency issue with the for loops


# From Becanovic simplex implementation
# from simplex_grid import simplex_grid
# import numpy as np

# Generate weight vectors for 3-objective optimization
# m = 5  # number of objectives
# r = 11  # resolution (higher = more points)

# list_w = simplex_grid(m=m, r=r)
# array_w = np.array(list_w)

nb_w = 10000
list_w = []
w_base = 0.01
constant_w_1 = 0.01
constant_w_2 = 0.01
low_log_uniform = 0                 # ln(1)
high_log_uniform = np.log(100)
t = np.linspace(0,49,50)
for ind_w in range(nb_w):
    log_w_max = np.random.uniform(low_log_uniform, high_log_uniform)
    w_max = np.exp(log_w_max)                               # asymptotic value of the sigmoid
    t_transition = np.random.randint(30,45)                 # slowing moment
    k_intensity = np.random.uniform(1,5)                    # slowing intensity
    num = w_max - w_base
    denom = 1 + np.exp(-k_intensity*(t - t_transition))
    w_3 = w_base + num/denom
    # print(type(constant_w_1))
    # print(type(constant_w_2))
    # print(w_3.shape)
    w = np.array([constant_w_1*np.ones_like(w_3), constant_w_2*np.ones_like(w_3), w_3]).T
    list_w.append(w)
array_w = np.array(list_w)
print(f"Shape of array of w: {array_w.shape}")

# Figuring out bad combination of w coordinates (uncomment to check a bad solution, when there's no cost weights over angular velocities)
# results_angles_cpin, results_vitesses_angulaires_cpin = solve_DOC([0.0, 0.0, 1.0], x_fin = -1.0, q_init= np.array([0, np.pi/4]))
# plt.plot(results_angles_cpin[:,0])
# plt.show()
# plt.plot(results_angles_cpin[:,1])
# plt.show()


# Checking the sum of w components (should equal to 1 for each combination)
# card_w = array_w.shape[0]
# for i in range(card_w):
#     print(array_w[i,:].sum())


# Checking how to iterate on list_w
# for w in list_w:
#     print(w)


list_results_angles, list_results_angular_velocities, array_w = generate_DOC_solutions_list_w(array_w, x_fin, q_init)

plot_trajectory_q1(list_results_angles, list_results_angular_velocities, linestyle = '-')
plot_trajectory_q2(list_results_angles, list_results_angular_velocities, linestyle = '-')

array_results_angles = np.array(list_results_angles)
array_results_angular_velocities = np.array(list_results_angular_velocities)

print(f"w array shape: {array_w.shape}")
print(f"Angles array shape: {array_results_angles.shape}")

# Saving trajectories and associated w in npy files
if (array_w.shape[0] == array_results_angles.shape[0]):
    np.save(f'data/array_w_variables_w_{nb_w}.npy', array_w)
    np.save(f'data/array_results_angles_variables_w_{nb_w}.npy', array_results_angles)