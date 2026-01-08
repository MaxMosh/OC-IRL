import pinocchio as pin
import math
import time
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
import eigenpy
import numpy as np

import casadi as ca
from pinocchio import casadi as cpin

import matplotlib.pyplot as plt

from tools import irl_utils
from tools import steps_two_arms_robot_cpin


results_angle_casadi, results_vitesses_angulaires_casadi = steps_two_arms_robot_cpin.solve_DOC(w = steps_two_arms_robot_cpin.w_true)

q_op = np.array([results_angle_casadi[:,0], results_angle_casadi[:,1]]).T
dq_op = np.array([results_vitesses_angulaires_casadi[:,0], results_vitesses_angulaires_casadi[:,1]]).T
# print(dq_op.shape)
dPx_op = np.zeros((1,steps_two_arms_robot_cpin.N)).T
# print(dPx_op.shape)

# for i in range(steps_two_arms_robot_cpin.N):
#     dPx_op[i,:] = steps_two_arms_robot_cpin.vx_fun(q_op[i,:], dq_op[i,:])


nb_nopt = 100

param = {"nb_samples": 100,
         "nb_joints": 2,
         "nb_nopt": nb_nopt,
         "noise_std": 1e-3,

         "nb_cost": 5,
         "nb_traj" : 1,

         "nb_w" : 0,
         "variables_w": 0
         }

ddq_fic = np.zeros(q_op.shape)

q_nopts, dq_nopts = irl_utils.generate_non_optimal_traj(q_op, dq_op, ddq_fic, param)
# q_nopts, dq_nopts, dPx_nopts = irl_utils.generate_non_optimal_traj(q_op, dq_op, ddq_fic, dPx_op, param)

q_nopts = np.transpose(q_nopts, (1, 0, 2))
dq_nopts = np.transpose(dq_nopts, (1, 0, 2))
# dPx_nopts = np.transpose(dPx_nopts, (1, 0, 2))


J_opt = steps_two_arms_robot_cpin.cost_function_value(q_op.T, dq_op.T)    #, dPx_op.T)
J_nopt = np.zeros((5, nb_nopt))
# J_nopt = []
for i in range(nb_nopt):
    J_nopt[:,i] = steps_two_arms_robot_cpin.cost_function_value(q_nopts[:,:,i], dq_nopts[:,:,i])      #, dPx_nopts[:, :, i])
candidates_costs_true = steps_two_arms_robot_cpin.w_true.T @ J_nopt
candidates_and_expert_costs_true = np.append(candidates_costs_true, steps_two_arms_robot_cpin.w_true.T @ J_opt)


temperature = max(candidates_costs_true)
non_normalized_distribution = np.exp(-candidates_costs_true/temperature)
normalized_distribution = non_normalized_distribution/non_normalized_distribution.sum()


nb_iter_IRL = 20
# nb_iter_IRL = 1

J_nopt_augmented = J_nopt.copy()
# J_expert_trajectories_current = J_expert_trajectories.copy()
J_expert_trajectories_current = J_opt.copy()

# candidates_costs_true_current = steps_two_arms_robot_casadi.w_true.T @ J_nopt_augmented

results_angles_list = np.zeros((2, 100, nb_iter_IRL))
results_angles_velocity_list = np.zeros((2, 100, nb_iter_IRL))

rmse_candidate = 1e9
ind_cand = -1

w_est_list = []

for i in range(nb_iter_IRL):
    print(i)

    # Estimation de w par IRL
    w_est_current = steps_two_arms_robot_cpin.IRL_step(J_expert_trajectories_current, J_nopt_augmented)
    print(w_est_current)

    # Calcul de la trajectoire optimale avec w_est
    results_angles_casadi_current, results_vitesses_angulaires_casadi_current = steps_two_arms_robot_cpin.solve_DOC(w = np.concatenate((w_est_current, np.array([0.0, 0.0]))))

    # Calcul du vecteur \Phi
    cost_features_new_traj = steps_two_arms_robot_cpin.cost_function_value(results_angles_casadi_current.T, results_vitesses_angulaires_casadi_current.T) #, dPx_nopts.T)

    # Ajout du nouveau vecteur de coûts
    J_nopt_augmented = np.concatenate((J_nopt_augmented, cost_features_new_traj.reshape(-1, 1)),axis=1)
    print(J_nopt_augmented.shape)

    results_angles_list[:,:,i] = results_angles_casadi_current.T
    results_angles_velocity_list[:,:,i] = results_vitesses_angulaires_casadi_current.T

    print(f"shape results_angles_casadi_current {results_angles_casadi_current.shape}")
    rmse_q1 = (q_op[:,0] - results_angles_casadi_current[:,0])**2
    rmse_q1_tot = np.sqrt(rmse_q1.sum())
    rmse_q2 = (q_op[:,1] - results_angles_casadi_current[:,1])**2
    rmse_q2_tot = np.sqrt(rmse_q2.sum())
    # rmse_current_q1 = np.sqrt(results_angles_casadi_current
    # rmse_current = np.sqrt(results_angles_casadi_current - q_op)
    rmse_tot = rmse_q1_tot + rmse_q2_tot
    if rmse_tot <= rmse_candidate:
        rmse_candidate = rmse_tot
        ind_cand = i
    
    w_est_list.append(w_est_current)


# plt.plot(q_op[:,0], label="q1 from the initial DOC solution", linestyle='--')
# for i in range(nb_iter_IRL):
#     if (((q_op[:,0] - results_angles_list[0,:,i])**2).sum() < 5):
#         if (i == (nb_iter_IRL-1)):
#             plt.plot(results_angles_list[0,:,i], label="q1 for each iteration of MO-IRL", color=(1, 0, 0, 0.5))
#         else:
#             plt.plot(results_angles_list[0,:,i], color=(1, 0, 0, 0.2))
# plt.legend()
# plt.savefig("plots/MO_IRL_each_iteration_q1.png")
# plt.show()

# plt.plot(q_op[:,1], label="q2 from the initial DOC solution", linestyle='--')
# for i in range(nb_iter_IRL):
#     if (((q_op[:,1] - results_angles_list[1,:,i])**2).sum() < 5):
#         if (i == (nb_iter_IRL-1)):
#             plt.plot(results_angles_list[1,:,i], label="q2 for each iteration of MO-IRL", color=(1, 0, 0, 0.5))
#         else:
#             plt.plot(results_angles_list[1,:,i], alpha=0.2)
# plt.legend()
# plt.savefig("plots/MO_IRL_each_iteration_q2.png")
# plt.show()
import matplotlib.pyplot as plt

fig, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# ===== q1 =====
ax_q1.plot(
    q_op[:, 0],
    label="q1 from the initial DOC solution",
    linestyle='--'
)

for i in range(nb_iter_IRL):
    if (((q_op[:, 0] - results_angles_list[0, :, i])**2).sum() < 5):
        if i == nb_iter_IRL - 1:
            ax_q1.plot(
                results_angles_list[0, :, i],
                label="q1 for each iteration of MO-IRL",
                color=(1, 0, 0, 0.5)
            )
        else:
            ax_q1.plot(
                results_angles_list[0, :, i],
                color=(1, 0, 0, 0.2)
            )

ax_q1.legend()
ax_q1.set_ylabel("q1 (rad)")
ax_q1.set_title("MO-IRL iterations - q1")

# ===== q2 =====
ax_q2.plot(
    q_op[:, 1],
    label="q2 from the initial DOC solution",
    linestyle='--'
)

for i in range(nb_iter_IRL):
    if (((q_op[:, 1] - results_angles_list[1, :, i])**2).sum() < 5):
        if i == nb_iter_IRL - 1:
            ax_q2.plot(
                results_angles_list[1, :, i],
                label="q2 for each iteration of MO-IRL",
                color=(1, 0, 0, 0.5)
            )
        else:
            ax_q2.plot(
                results_angles_list[1, :, i],
                alpha=0.2
            )

ax_q2.legend()
ax_q2.set_ylabel("q2 (rad)")
ax_q2.set_xlabel("Time step")
ax_q2.set_title("MO-IRL iterations - q2")

plt.tight_layout()
plt.savefig("plots/MO_IRL_each_iteration_q1_q2.png")
plt.show()



# print(f"shape res IRL : {results_angles_list[:,:,:].shape}\n")
# print(f"shape opt : {q_op[:,:].shape}\n")
# print(f"q1{results_angles_list[0,-1,-1]} ")
# rmse = (results_angles_list[:,:,:].reshape((100,2,20)) - q_op[:,:].reshape((100,2,1)))**2
# # print(rmse.shape)
# rmse_tot = rmse.sum(axis=(0,1))
# print(f"Taille rmse : {rmse_tot.shape} \n")

# best_index = np.argmin(rmse_tot)

rmse_q1 = np.sqrt(np.mean((q_op[:,0] - results_angles_list[0,:,-1])**2))
rmse_q2 = np.sqrt(np.mean((q_op[:,1] - results_angles_list[1,:,-1])**2))

fig, (ax_q1, ax_q2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# ===== q1 =====
ax_q1.plot(q_op[:, 0], label="q1 from the initial DOC solution", linestyle='--')
ax_q1.plot(results_angles_list[0, :, -1], label="q1 last iteration IRL")

ax_q1.text(
    0.98, 0.05,
    f'RMSE: {rmse_q1:.2f} rad',
    transform=ax_q1.transAxes,
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
)

ax_q1.legend()
ax_q1.set_ylabel("q1 (rad)")
ax_q1.set_title("MO-IRL last iteration vs ground truth - q1")

# ===== q2 =====
ax_q2.plot(q_op[:, 1], label="q2 from the initial DOC solution", linestyle='--')
ax_q2.plot(results_angles_list[1, :, -1], label="q2 last iteration IRL")

ax_q2.text(
    0.98, 0.05,
    f'RMSE: {rmse_q2:.2f} rad',
    transform=ax_q2.transAxes,
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
)

ax_q2.legend()
ax_q2.set_ylabel("q2 (rad)")
ax_q2.set_xlabel("Time step")
ax_q2.set_title("MO-IRL last iteration vs ground truth - q2")

plt.tight_layout()
plt.savefig("plots/MO_IRL_last_iteration_q1_q2.png")
plt.show()


Px_opt = steps_two_arms_robot_cpin.L_1*ca.cos(q_op[-1,0]) + steps_two_arms_robot_cpin.L_2*ca.cos(q_op[-1,0] + q_op[-1,1])
Px_last_IRL = steps_two_arms_robot_cpin.L_1*ca.cos(results_angles_list[0,-1,-1]) + steps_two_arms_robot_cpin.L_2*ca.cos(results_angles_list[0,-1,-1] + results_angles_list[1,-1,-1])
print(f"Px_opt : {Px_opt} \n")
print(f"Px_last_IRL : {Px_last_IRL} \n")

# print(f"Taille w_est_list : {np.array(w_est_list).shape}")
# print(f"Taille J_nopt_augmented : {J_nopt_augmented[:,nb_nopt:].shape}")

print(f"mean dq1 for opt: {(dq_op[0,:]**2).sum()}")
print(f"mean dq2 for opt: {(dq_op[1,:]**2).sum()}")

# print(results_angles_velocity_list[0,:,1].shape)

print(results_angles_velocity_list.shape)
print(f"mean dq1 for one step of IRL: {(results_angles_velocity_list[0,:,1]**2).sum()}")
print(f"mean dq2 for one step of IRL: {(results_angles_velocity_list[1,:,1]**2).sum()}")

print(f"True value of w: {steps_two_arms_robot_cpin.w_true}")
print(f"Recovered value of w: {w_est_list[-1]}")

"""
w_est_array = np.array(w_est_list)
for i in range(nb_iter_IRL):
    J_curr = J_nopt_augmented[:,i]

    indices = np.arange(5)  # positions sur l'axe x
    largeur = 0.35  # largeur des barres

    plt.bar(indices - largeur/2, J_opt, largeur, label='Initial cost features')
    plt.bar(indices + largeur/2, J_curr, largeur, label=f'Cost after IRL, step {i}')

    plt.xlabel('Catégorie')
    plt.ylabel('Coût')
    plt.title('Comparaison des coûts initiaux et finaux')
    plt.xticks(indices, [f'Catégorie {i+1}' for i in indices])
    plt.legend()
    plt.show()


for i in range(nb_iter_IRL):
    J_curr = J_nopt_augmented[:,i]
    w_est_curr = w_est_array[i,:]
    # print(f"w_est_curr shape : {w_est_curr.shape}")
    weighted_cost = [w_est_curr[k]*J_curr[k] for k in range(3)] + [0.0, 0.0]
    # print(f"weighted_cost shape : {len(weighted_cost)}")

    indices = np.arange(5)  # positions sur l'axe x
    largeur = 0.35  # largeur des barres

    weighted_cost_opt = [steps_two_arms_robot_casadi.w_true[k]*J_opt[k] for k in range(5)]

    plt.bar(indices - largeur/2, weighted_cost_opt, largeur, label='Initial weighted costs')
    plt.bar(indices + largeur/2, weighted_cost, largeur, label=f'Weighted cost after IRL, step {i}')

    plt.xlabel('Catégorie')
    plt.ylabel('Coût')
    plt.title('Comparaison des coûts initiaux et finaux')
    plt.xticks(indices, [f'Catégorie {i+1}' for i in indices])
    plt.legend()
    plt.show()
"""