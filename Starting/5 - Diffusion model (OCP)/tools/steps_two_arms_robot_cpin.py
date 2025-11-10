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


# ROBOT LOADING
exampleRobotDataPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/5 - Diffusion model/assets/'
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
def solve_DOC(w, x_fin = -0.75, q_init = [0, ca.pi/4]):
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
    opti.subject_to(q[:, 0] == q_init)
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == x_fin)

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



def generate_DOC_solutions(w, list_xfin, list_q_init):
    # nb_xfin = len(liste_xfin)
    number_of_not_added = 0
    results_angles_list = []
    results_vitesses_angulaires_list = []
    for q_init in list_q_init:
        for xfin in list_xfin:
            # print(q_init, xfin)
            try:
                results_angles_cpin, results_vitesses_angulaires_cpin = solve_DOC(w, x_fin = xfin, q_init=q_init)
                results_angles_list.append(results_angles_cpin)
                results_vitesses_angulaires_list.append(results_vitesses_angulaires_cpin)
            except:
                number_of_not_added+=1
    print(f"Number of not added: {number_of_not_added}")
    return results_angles_list, results_vitesses_angulaires_list



def plot_trajectory_q1(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_angles_cpin)
    for i in range(number_of_trajectories):
        # plt.plot(results_angles_cpin[:,0])
        plt.plot(results_angles_cpin[i][:,0], linestyle = linestyle)
        # print(results_angles_cpin[:,0].shape)
        # plt.show()
        # plt.plot(results_angles_cpin[:,1])
        # plt.plot(results_angles_cpin[i][:,1], linestyle = linestyle)
        # plt.show()



def plot_trajectory_q2(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_angles_cpin)
    for i in range(number_of_trajectories):
        # plt.plot(results_angles_cpin[:,0])
        # plt.plot(results_angles_cpin[i][:,0], linestyle = linestyle)
        # print(results_angles_cpin[:,0].shape)
        # plt.show()
        # plt.plot(results_angles_cpin[:,1])
        plt.plot(results_angles_cpin[i][:,1], linestyle = linestyle)
        # plt.show()