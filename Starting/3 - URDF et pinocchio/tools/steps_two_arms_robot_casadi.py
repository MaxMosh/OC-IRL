# IMPORT DES PACKAGES
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



# CONSTANTS
# time constants
t_0 = 0
t_f = 1
N = 100  # number of discretization points
dt = (t_f - t_0) / N

# lenghts
L_1 = 1
L_2 = 1
N_angles = 2

w_true = np.array([0.45, 0.05, 0.5, 0.0, 0.0])

# MO-IRL constants
MO_IRL_N_subsampling = 4



# COST FEATURES
# symbolic variables
q1_sym = ca.SX.sym("q1")
q2_sym = ca.SX.sym("q2")
dq1_sym = ca.SX.sym("dq1")
dq2_sym = ca.SX.sym("dq2")

# cost features functions
Px = L_1*ca.cos(q1_sym) + L_2*ca.cos(q1_sym + q2_sym)
Px_fun = ca.Function("Px", [q1_sym, q2_sym], [Px])
grad_Px = ca.gradient(Px, ca.vertcat(q1_sym, q2_sym))
dPx_fun = ca.Function("dPx", [ca.vertcat(q1_sym, q2_sym), ca.vertcat(dq1_sym, dq2_sym)] , [grad_Px.T @ ca.vertcat(dq1_sym, dq2_sym)])

Py = L_1*ca.sin(q1_sym) + L_2*ca.sin(q1_sym + q2_sym)
Py_fun = ca.Function("Py", [q1_sym, q2_sym], [Py])
grad_Py = ca.gradient(Py, ca.vertcat(q1_sym, q2_sym))
dPy_fun = ca.Function("dPy", [ca.vertcat(q1_sym, q2_sym), ca.vertcat(dq1_sym, dq2_sym)] , [grad_Py.T @ ca.vertcat(dq1_sym, dq2_sym)])

# TODO: I think I should delete Px and Py in the future, to keep only f, and compute dPx et dPy using f[0] and f[1]
f = ca.vertcat(L_1*ca.cos(q1_sym) + L_2*ca.cos(q1_sym + q2_sym),
            L_1*ca.sin(q1_sym) + L_2*ca.sin(q1_sym + q2_sym),
            0)
J = ca.jacobian(f, ca.vertcat(q1_sym, q2_sym))
J_func = ca.Function("J_func", [q1_sym, q2_sym], [J])
JTJ = J.T @ J
log_det_JJt_fun = ca.Function("log_det_JJt", [ca.vertcat(q1_sym, q2_sym)], [-ca.log(ca.det(JTJ) + 1e-9)])
JJt_fun = ca.Function("JJt", [ca.vertcat(q1_sym, q2_sym)], [JTJ])



# COST VALUES COMPUTING
def cost_function_value(q_var, dq_var):     #, dPx_var):    # NOTE : vient du test avec bruitage sur dPx
    f_10 = 0
    f_11 = 0
    f_2 = 0
    f_3 = 0
    f_4 = 0
    N_time = q_var.shape[1]
    # if N_time != N:         # display of a message in case the number of time steps differs from the initial problem
        # print(f"INFORMATION: the number of the times steps of the trajectories ({N_time}) differs from the initial problem ({N}).")
    for t in range(N_time):
        # print(q_var.shape)
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        f_10 += ca.DM(dq_var[0, t]**2)
        f_11 += ca.DM(dq_var[1, t]**2)
        # print(f_11)
        f_2 += dPx_fun(q_t, dq_t)**2
        # f_2 += ca.DM(dPx_var[0, t]**2)
        # print(f_2.shape)
        f_3 += dPy_fun(q_t, dq_t)**2
        f_4 += log_det_JJt_fun(q_t)
    return np.array([f_10, f_11, f_2, f_3, f_4])[:,0,0]



# DOC SOLVING
def solve_DOC(w):
    opti = ca.Opti()

    # variables à optimiser
    q = opti.variable(N_angles, N)
    dq = opti.variable(N_angles, N)

    # J_1 = 0
    J_10 = 0
    J_11 = 0
    J_2 = 0
    J_3 = 0
    J_4 = 0
    for t in range(N):
        q_t = ca.vertcat(q[0, t], q[1, t])
        dq_t = ca.vertcat(dq[0, t], dq[1, t])
        # J_1 += w_1[0,t] * dq[0, t]**2 + w_1[1,t] * dq[1, t]**2
        # J_1 += w[0] * dq[0, t]**2 + w[1] * dq[1, t]**2
        # J_10 += w[0] * dq[0, t]**2
        J_10 += dq[0, t]**2
        # J_11 += w[1] * dq[1, t]**2
        J_11 += dq[1, t]**2
        # J_2 += w_2[t] * dPx_fun(q_t, dq_t)**2
        # J_2 += w[2] * dPx_fun(q_t, dq_t)**2
        J_2 += dPx_fun(q_t, dq_t)**2
        # J_3 += w_3[t] * dPy_fun(q_t, dq_t)**2
        # J_3 += w[3] * dPy_fun(q_t, dq_t)**2
        J_3 += dPy_fun(q_t, dq_t)**2
        # J_4 += w_4[t] * log_det_JJt_fun(q_t)
        # J_4 += w[4] * log_det_JJt_fun(q_t)
        J_4 += log_det_JJt_fun(q_t)

    # fonction que l'on cherche à minimiser
    # opti.minimize(J_10 + J_11 + J_2 + J_3 + J_4)
    # opti.minimize(w[0]*J_10 + w[1]*J_11 + w[2]*J_2 + w[3]*J_3 + w[4]*J_4)
    opti.minimize(w[0]*J_10 + w[1]*J_11 + w[2]*J_2)

    # contraintes d’intégration
    for t in range(N-1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
    
    # conditions initiales sur q
    opti.subject_to(q[:, 0] == [0, ca.pi/4])

    # condition finale sur Px
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == -0.75)

    # conditions supplémentaires sur les angles, non demandé dans l'énoncé
    opti.subject_to(opti.bounded(-ca.pi/2, q[0, :], ca.pi/2))
    opti.subject_to(opti.bounded(-ca.pi/2, q[1, :], ca.pi/2))


    opti.solver("ipopt")
    sol = opti.solve()

    q1_casadi = sol.value(q[0,:])
    q2_casadi = sol.value(q[1,:])

    dq1_casadi = sol.value(dq[0,:])
    dq2_casadi = sol.value(dq[1,:])

    results_angles_casadi = np.array([q1_casadi, q2_casadi]).T
    results_vitesses_angulaires_casadi = np.array([dq1_casadi, dq2_casadi]).T
    
    return results_angles_casadi, results_vitesses_angulaires_casadi



# IRL (1) : réimplémentation de l'IRL de Kalakrishnan
# NOTE : cette approche semble ne pas fonctionner pour certaines trajectoires "complexes"
# PISTES : 
# - orthonogonaliser l'espace des contraintes (par exemple, l'influence du produit \dot{q_1} \times \dot{q_2} est peut-être mal gérée)
# - ne pas faire plusieurs itérations (cela perturbe la ditribution selon moi),
# - utiliser un solveur non-gradient based (pas dans casadi)
def IRL_step(expert_trajectories_real_cost_features, noisy_trajectories_real_cost_features):
    nb_noisy = noisy_trajectories_real_cost_features.shape[1]
    print(nb_noisy)
    
    opti_IRL = ca.Opti()

    w1_inv = opti_IRL.variable()
    w2_inv = opti_IRL.variable()
    w3_inv = opti_IRL.variable()
    # w4_inv = opti_IRL.variable()
    # w5_inv = opti_IRL.variable()

    # opti_IRL.subject_to(w1_inv + w2_inv + w3_inv + w4_inv + w5_inv == 1)
    # opti_IRL.subject_to(w1_inv + w2_inv + w3_inv + w4_inv == 1)
    # opti_IRL.subject_to(w1_inv + w2_inv + w3_inv == 1)
    opti_IRL.subject_to(w1_inv + w2_inv + w3_inv == 1)
    opti_IRL.subject_to(w1_inv >= 0.0)
    opti_IRL.subject_to(w2_inv >= 0.0)
    opti_IRL.subject_to(w3_inv >= 0.0)
    # opti_IRL.subject_to(w4_inv >= 0.02)
    # opti_IRL.subject_to(w5_inv >= 0.02)

    cost_opt = w1_inv * expert_trajectories_real_cost_features[0] + w2_inv * expert_trajectories_real_cost_features[1] + w3_inv * expert_trajectories_real_cost_features[2] #+ w4_inv * expert_trajectories_real_cost_features[3] + w5_inv * expert_trajectories_real_cost_features[4]

    cost_all = w1_inv * noisy_trajectories_real_cost_features[0] + w2_inv * noisy_trajectories_real_cost_features[1] + w3_inv * noisy_trajectories_real_cost_features[2] #+ w4_inv * noisy_trajectories_real_cost_features[3] + w5_inv * noisy_trajectories_real_cost_features[4]

    num = ca.exp(-cost_opt)
    denom = 0
    for k in range(nb_noisy):
        denom += ca.exp(-cost_all[k])
    log_vrais = - ca.log(num/denom)

    opti_IRL.minimize(log_vrais)

    # initialisation pour ipopt
    opti_IRL.set_initial(w1_inv, 0.5)
    opti_IRL.set_initial(w2_inv, 0.25)
    opti_IRL.set_initial(w3_inv, 0.25)
    # opti_IRL.set_initial(w4_inv, 0.05)
    # opti_IRL.set_initial(w5_inv, 0.3)

    # résolution
    opti_IRL.solver('ipopt')
    sol_inv = opti_IRL.solve()

    w1_inv_opt = sol_inv.value(w1_inv)
    w2_inv_opt = sol_inv.value(w2_inv)
    w3_inv_opt = sol_inv.value(w3_inv)
    # w4_inv_opt = sol_inv.value(w4_inv)
    # w5_inv_opt = sol_inv.value(w5_inv)

    w_est = np.array([float(w1_inv_opt), float(w2_inv_opt), float(w3_inv_opt)])
    # w_est = np.array([float(w1_inv_opt), float(w2_inv_opt)])


    print("w_true =", w_true)
    print("w_est  =", w_est)

    return w_est



# MO-IRL (2)
def compute_D(N_subsampling):
    N_time = N      # number of discretization time-steps
    D = [int(np.floor(k*N_time/N_subsampling)) for k in range(N_subsampling)]
    return D



def generate_subsampled_trajectories_costs(q, dq, N_subsampling):
    D = compute_D(N_subsampling)
    # print(f"Liste of subsample steps D: {D}")
    Phi = np.zeros((3, N_subsampling))                                                  # NOTE: we only take the first three costs in a first time
    for ind_new_cost, ind_subsampling in enumerate(D):
        q_subsampled = q[ind_subsampling:,:]
        dq_subsampled = dq[ind_subsampling:,:]
        # print(q_subsampled.shape)
        # print(dq_subsampled.shape)
        Phi[:,ind_new_cost] = cost_function_value(q_subsampled.T, dq_subsampled.T)[:3]      # NOTE: we only take the first three costs in a first time
    return Phi



def MO_IRL_solve(q_op, dq_op):
    t = 0
    
    w_0 = np.array([0.1, 0.1, 0.1])                                                            # NOTE: we only take the three first costs in a first time
    w = w_0

    # tau_set = []
    # tau_t = solve_DOC(w)
    # tau_set.append(tau_t)
    # nb_nopt_noisy = 200

    # param = {"nb_samples": 100,
    #      "nb_joints": 2,
    #      "nb_nopt": nb_nopt_noisy,
    #      "noise_std": 1e-2,

    #      "nb_cost": 5,
    #      "nb_traj" : 1,

    #      "nb_w" : 0,
    #      "variables_w": 0
    #      }

    # ddq_fic = np.zeros(q_op.shape)
    # q_nopts, dq_nopts = irl_utils.generate_non_optimal_traj(q_op, dq_op, ddq_fic, param)
    # print(f"shape q_nopts : {q_nopts.shape}")
    # print(f"shape dq_nopts : {dq_nopts.shape}")
    q_set = q_op.reshape((100,2,1))
    dq_set = dq_op.reshape((100,2,1))

    # M_1t = np.inf
    # M_2t = np.inf

    # TODO: add Wolfe conditions instead of big for loop (I didn't get for now if these conditions are only for m_1, or also for \Delta_w)
    for ind_MO_IRL in range(50):
        print(ind_MO_IRL)
        q_nopts_new, dq_nopts_new, w_new = MO_IRL_step(q_op, dq_op, q_set, dq_set, w)
        q_set = q_nopts_new
        dq_set = dq_nopts_new
        w = w_new
        t += 1
    
    return w



def MO_IRL_step(q_op, dq_op, q_set, dq_set, w_current):
    
    # Improvement direction
    opti_MO_IRL = ca.Opti()
    Delta_w = opti_MO_IRL.variable(3)           # number of cost features
    set_size = q_set.shape[2]
    print(f"Set size in IRL step: {set_size}")

    # Subsampled costs generation
    print(q_op.shape)
    Phi_opt_subsampled_cost = generate_subsampled_trajectories_costs(q_op, dq_op, MO_IRL_N_subsampling)
    Phi_opt = Phi_opt_subsampled_cost[:,0]
    # Phi_subsampled_costs = np.zeros((3, MO_IRL_N_subsampling*set_size))
    D = compute_D(MO_IRL_N_subsampling)
    Phi_subsampled_costs = np.zeros((3, set_size, MO_IRL_N_subsampling))
    for ind_set in range(set_size):
        q_current = q_set[:,:,ind_set]
        dq_current = dq_set[:,:,ind_set]

        # index_first_cost = MO_IRL_N_subsampling*ind_set
        # index_last_cost = MO_IRL_N_subsampling*(ind_set + 1)
        # Phi_subsampled_costs[:,index_first_cost:index_last_cost] = generate_subsampled_trajectories_costs(q_current, dq_current)
        Phi_subsampled_costs[:, ind_set, :] = generate_subsampled_trajectories_costs(q_current, dq_current, MO_IRL_N_subsampling)



    
    Phi_op = cost_function_value(q_op, dq_op)[:3]

    term = 0
    for ind_d, d in enumerate(D):
        theta_d = (t_f - d + 1) / (t_f + 1)



        denom_part = 0
        for ind_set in range(set_size):
            Phi_i = Phi_subsampled_costs[:,ind_set,0]
            gamma_i = ca.exp(-w_current.T @ (Phi_i - Phi_op))

            Phi_id = Phi_subsampled_costs[:, ind_set, ind_d]
            bar_Phi_id = Phi_id - Phi_opt_subsampled_cost[:,ind_d]

            # print(-Delta_w.T @ bar_Phi_id)
            # denom_part += gamma_i * ca.exp(-Delta_w.T @ bar_Phi_id)
            exp_term = -(Delta_w[0] * bar_Phi_id[0] + Delta_w[1] * bar_Phi_id[1] + Delta_w[2] * bar_Phi_id[2])
            denom_part += gamma_i * ca.exp(exp_term)

        term += -theta_d * ca.log(1/(1 + denom_part))

    # Objective function with regularization
    lam = 1e-4
    beta = 1e-4
    obj = term + (beta/2) * ca.sumsqr(Delta_w) #+ lam * ca.sum1(ca.fabs(Delta_w))

    # opti_MO_IRL.subject_to(w1_inv + w2_inv + w3_inv == 1)

    # opti_MO_IRL.subject_to(Delta_w[0] >= -0.1)
    opti_MO_IRL.subject_to(Delta_w[0] <= 0.1)
    opti_MO_IRL.subject_to(Delta_w[0] >= -w_current[0])

    # opti_MO_IRL.subject_to(Delta_w[1] >= -0.1)
    opti_MO_IRL.subject_to(Delta_w[1] <= 0.1)
    opti_MO_IRL.subject_to(Delta_w[1] >= -w_current[1])

    # opti_MO_IRL.subject_to(Delta_w[2] >= -0.1)
    opti_MO_IRL.subject_to(Delta_w[2] <= 0.1)
    opti_MO_IRL.subject_to(Delta_w[2] >= -w_current[2])

    opti_MO_IRL.minimize(obj)
    
    # initialisation pour ipopt
    opti_MO_IRL.set_initial(Delta_w, [0.001, 0.001, 0.001])

    # résolution
    opti_MO_IRL.solver('ipopt')
    sol_inv = opti_MO_IRL.solve()

    Delta_w_MO_IRL_opt = sol_inv.value(Delta_w)
    print(Delta_w_MO_IRL_opt)

    # Step acceptance loop
    alpha = 1
    max_iter_step_acceptance = 10
    ind_iter_step_acceptance = 0
    step_found = False
    m_3_eps = 1e-2

    first_joints_trajectory, first_angular_speeds = solve_DOC(w_current)
    m_3_first = np.sum(first_joints_trajectory[0,:] - q_op[0,:])**2 + np.sum(first_angular_speeds[1,:] - q_op[1,:])**2
    while (ind_iter_step_acceptance < max_iter_step_acceptance) and not(step_found):

        new_w = w_current + alpha*Delta_w_MO_IRL_opt
        new_candidate_joints_trajectory, new_candidate_angular_speeds = solve_DOC(new_w)

        # m_1 criterion (Mehrdad)
        # TODO

        # m_2 criterion (Mehrdad)
        # TODO

        # Criterion in Maxime Sabbah and Vincent Bonnet research paper
        m_3_current = np.sum(new_candidate_joints_trajectory[0,:] - q_op[0,:])**2 + np.sum(new_candidate_joints_trajectory[1,:] - q_op[1,:])**2
        if m_3_current < m_3_first:
            step_found = True
            print(q_set.shape)
            print(new_candidate_joints_trajectory.shape)
            new_q_set = np.concatenate((q_set, new_candidate_joints_trajectory.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)
            new_dq_set = np.concatenate((dq_set, new_candidate_angular_speeds.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)
        
        alpha /= 4
    
    if not(step_found):
        new_q_set = np.concatenate((q_set, first_joints_trajectory.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)
        new_dq_set = np.concatenate((dq_set, first_angular_speeds.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)

    return new_q_set, new_dq_set, new_w