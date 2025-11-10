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
NB_ITER_MOIRL = 10



# ROBOT LOADING
exampleRobotDataPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/3 - URDF et pinocchio/assets/'
urdf = exampleRobotDataPath + 'mon_robot.urdf'
robot = RobotWrapper.BuildFromURDF( urdf, [ exampleRobotDataPath, ] )
# robot.setVisualizer(GepettoVisualizer()) # AJOUT
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv



#
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

q = ca.SX.sym("q",NQ,1)
dq = ca.SX.sym("dq",NQ,1)
# ddq = ca.SX.sym("ddq",NQ,1)     # NORMALEMENT PAS BESOIN



# COSTS FEATURES FUNCTIONS
frame_id = cmodel.getFrameId("ee_link")

Jv = ca.Function('Jv', [q], [cpin.computeFrameJacobian(cmodel, cdata, q, frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2,:]])

Jv_val = Jv(q)
dP = Jv_val @ ca.vertcat(dq[0],dq[1])

vx_fun = ca.Function("vx_fun", [q, dq], [dP[0]])
vy_fun = ca.Function("vy_fun", [q, dq], [dP[1]])

# oMtool = robot.data.oMf[IDX_EE]
# oRtool = oMtool.rotation
# Jee = oRtool[:2,:2] @ (pin.computeFrameJacobian(robot.model,robot.data,np.array([q1_test[i],q2_test[i]]),IDX_EE)[:2,:2])

jac_fun = ca.Function("jac_fun", [q], [cpin.computeFrameJacobian(cmodel, cdata, q, frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)])

manipulability_fun = ca.Function("manipulability_fun", [q], [jac_fun(q).T @ jac_fun(q)])

cost_manipulability_fun = ca.Function("cost_manipulability_fun", [q], [-ca.log(ca.det(manipulability_fun(q)) + 1e-9)])



# COST VALUES COMPUTING
def cost_function_value(q_var, dq_var):
    f_10 = 0
    f_11 = 0
    f_2 = 0
    f_3 = 0
    f_4 = 0
    N_time = q_var.shape[1]
    # if N_time != N:         # display of a message in case the number of time steps differs from the initial problem
        # print(f"INFORMATION: the number of the times steps of the trajectories ({N_time}) differs from the initial problem ({N}).")
    for t in range(N_time):
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        f_10 += ca.DM(dq_var[0, t]**2)
        f_11 += ca.DM(dq_var[1, t]**2)
        f_2 += vx_fun(q_t, dq_t)**2
        f_3 += vy_fun(q_t, dq_t)**2
        f_4 += cost_manipulability_fun(q_t)
    return np.array([f_10, f_11, f_2, f_3, f_4])[:,0,0]



# DOC SOLVING
def solve_DOC(w):
    opti = ca.Opti()

    # cadre du problème
    # temporel
    t_0 = 0
    t_f = 1
    N = 100  # nombre de points de discrétisation
    dt = (t_f - t_0) / N

    # variables à optimiser
    q = opti.variable(N_angles, N)
    dq = opti.variable(N_angles, N)


    J_10 = 0
    J_11 = 0
    J_2 = 0
    J_3 = 0
    J_4 = 0
    # NOTE : dans un premier temps, on met des poids "en dur"
    for t in range(N):
        q_t = ca.vertcat(q[0, t], q[1, t])
        dq_t = ca.vertcat(dq[0, t], dq[1, t])
        # q_t = ca.vertcat(q[0, t], q[1, t], q_fic[0, t])
        # cpin.framesForwardKinematics(cmodel, cdata, q_t)
        # J_1 += w_1[0,t] * dq[0, t]**2 + w_1[1,t] * dq[1, t]**2
        J_10 += dq[0, t]**2
        J_11 += dq[1, t]**2
        # J_2 += w_2[t] * vx_fun(q_t, dq_t)**2
        J_2 += vx_fun(q_t, dq_t)**2
        # J_3 += w_3[t] * vy_fun(q_t, dq_t)**2
        J_3 += vy_fun(q_t, dq_t)**2
        # J_4 += w_4[t] * cost_manipulability_fun(q_t)
        J_4 += cost_manipulability_fun(q_t)

    # fonction que l'on cherche à minimiser
    # opti.minimize(w[0] * J_10 + w[1] * J_11 + w[2] * J_2 + w[3] * J_3 + w[4] * J_4)
    opti.minimize(w[0] * J_10 + w[1] * J_11 + w[2] * J_2)

    # contraintes d’intégration
    # opti.subject_to(q_fic[0, N-1] == 0)
    for t in range(N-1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
        opti.subject_to(opti.bounded(-ca.pi/2, q[0, t+1], ca.pi/2))
        opti.subject_to(opti.bounded(-ca.pi/2, q[1, t+1], ca.pi/2))
    opti.subject_to(opti.bounded(-ca.pi/2, q[0, 0], ca.pi/2))
    opti.subject_to(opti.bounded(-ca.pi/2, q[0, 0], ca.pi/2))
    opti.subject_to(q[:, 0] == [0, ca.pi/4])
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == -0.75)

    # conditions supplémentaires sur les angles, non demandé dans l'énoncé
    # opti.subject_to(opti.bounded(-ca.pi/2, q, ca.pi/2))

    opti.solver("ipopt")
    sol = opti.solve()

    q1_cpin = sol.value(q[0,:])
    q2_cpin = sol.value(q[1,:])

    dq1_cpin = sol.value(dq[0,:])
    dq2_cpin = sol.value(dq[1,:])

    results_angles_cpin = np.array([q1_cpin, q2_cpin]).T
    results_vitesses_angulaires_cpin = np.array([dq1_cpin, dq2_cpin]).T

    return results_angles_cpin, results_vitesses_angulaires_cpin



# IRL step
def IRL_step(expert_trajectories_real_cost_features, noisy_trajectories_real_cost_features):
    nb_noisy = noisy_trajectories_real_cost_features.shape[1]
    
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
    print("\n\n\n\n----BEGINNING OF IRL OUTPUTS----\n\n\n\n")
    t = 0
    
    w_0 = np.array([0.01, 0.01, 0.01])                                                            # NOTE: we only take the three first costs in a first time
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
    for ind_MO_IRL in range(NB_ITER_MOIRL):
        print(f"MO-IRL iteration: {ind_MO_IRL}")
        q_nopts_new, dq_nopts_new, w_new = MO_IRL_step(q_op, dq_op, q_set, dq_set, w)
        q_set = q_nopts_new
        dq_set = dq_nopts_new
        w = w_new
        t+=1

    print("\n\n\n\n----END OF IRL OUTPUTS----\n\n\n\n")
    return w



def MO_IRL_step(q_op, dq_op, q_set, dq_set, w_current):
    
    print("\n----IRL STEP PROBLEM SOLVING (OPTIMAL \Delta w)----\n")

    # Improvement direction
    opti_MO_IRL = ca.Opti()
    Delta_w = opti_MO_IRL.variable(3)           # number of cost features
    set_size = q_set.shape[2]
    print(f"Trajectory set size in IRL for current step: {set_size}\n")

    # Subsampled costs generation
    # print(q_op.shape)
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

        # term += -theta_d * ca.log(1/(1 + denom_part))
        term += -theta_d * ca.log(1/(1 + denom_part))

    # Objective function with regularization
    lam = 1e-4
    beta = 1e-4
    obj = term #+ (beta/2) * ca.sumsqr(Delta_w) #+ lam * ca.sum1(ca.fabs(Delta_w))
    # obj = ca.sumsqr(Delta_w)
    # obj = Delta_w[0]**2

    # opti_MO_IRL.subject_to(w1_inv + w2_inv + w3_inv == 1)

    # opti_MO_IRL.subject_to(Delta_w[0] >= -0.1)
    # opti_MO_IRL.subject_to(Delta_w[0] <= 0.1)

    opti_MO_IRL.subject_to(Delta_w[0] >= -w_current[0])

    # opti_MO_IRL.subject_to(Delta_w[1] >= -0.1)
    # opti_MO_IRL.subject_to(Delta_w[1] <= 0.1)

    opti_MO_IRL.subject_to(Delta_w[1] >= -w_current[1])

    # opti_MO_IRL.subject_to(Delta_w[2] >= -0.1)
    # opti_MO_IRL.subject_to(Delta_w[2] <= 0.1)

    opti_MO_IRL.subject_to(Delta_w[2] >= -w_current[2])

    opti_MO_IRL.minimize(obj)
    
    # initialisation pour ipopt
    opti_MO_IRL.set_initial(Delta_w, [0.001, 0.001, 0.001])
    # opti_MO_IRL.set_initial(Delta_w, [10, 10, 10])
    # opti_MO_IRL.set_initial(Delta_w, [0.18, 0.23, -0.187])

    # résolution
    opti_MO_IRL.solver('ipopt')
    sol_inv = opti_MO_IRL.solve()

    Delta_w_MO_IRL_opt = sol_inv.value(Delta_w)
    print(f"\nValue found for $\Delta w$ {Delta_w_MO_IRL_opt}\n")

    print("\n----END OF IRL STEP PROBLEM SOLVING (OPTIMAL \Delta w)----\n")

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

        # q tilde
        new_candidate_joints_trajectory, new_candidate_angular_speeds = solve_DOC(new_w)

        # m_1 criterion (S. Mehrdad)
        # TODO

        # m_2 criterion (S. Mehrdad)
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
        ind_iter_step_acceptance += 1
    
    # if not(step_found):
    #     new_q_set = np.concatenate((q_set, first_joints_trajectory.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)
    #     new_dq_set = np.concatenate((dq_set, first_angular_speeds.reshape((q_set.shape[0],q_set.shape[1],1))), axis=2)

    print(f"alpha value: {alpha}")

    return new_q_set, new_dq_set, new_w