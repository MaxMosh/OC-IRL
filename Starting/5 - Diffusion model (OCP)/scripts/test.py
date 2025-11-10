import sys
import os

# Add workin directory path
sys.path.append(os.path.abspath("Starting/5 - Diffusion model"))

import tools.steps_two_arms_robot_cpin

import numpy as np
import matplotlib.pyplot as plt




w = tools.steps_two_arms_robot_cpin.w_true



nb_x_fin = 11
x_fin_values = np.linspace(0.0, -1.0, nb_x_fin)

results_angles, results_angles_velocities = tools.steps_two_arms_robot_cpin.generate_DOC_solutions(w, x_fin_values)

for i in range(nb_x_fin):
    tools.steps_two_arms_robot_cpin.plot_trajectory(results_angles[i], results_angles_velocities[i])
    # P_x = tools.steps_two_arms_robot_cpin.L_1*np.cos()
    # print(f"Valeur de P_x : {}")
plt.show()