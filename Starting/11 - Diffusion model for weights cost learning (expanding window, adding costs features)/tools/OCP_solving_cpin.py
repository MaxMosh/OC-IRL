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
N = 50                  # number of discretization points
dt = (t_f - t_0) / N

# lenghts
L_1 = 1
L_2 = 1
N_angles = 2


# ROBOT LOADING
import os
print(os.getcwd())
# assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/9 - Diffusion model for weights cost learning/assets/'
assetsPath = '/mnt/e/MaxMosh/Repositories/OC-IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
urdf = assetsPath + 'mon_robot.urdf'
robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
# robot.setVisualizer(GepettoVisualizer()) # AJOUT
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv

q0 = pin.neutral(robot.model)
print(f"q0: {q0}")
viz = robot.viewer
robot.display(q0)



# MODEL FOR CPIN
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

q = ca.SX.sym("q",NQ,1)
dq = ca.SX.sym("dq",NQ,1)
# ddq = ca.SX.sym("ddq",NQ,1)                       # UNUSED FOR NOW
acc = ca.SX.sym("acc", NV, 1)



# COSTS FEATURES FUNCTIONS
ee_frame_id = cmodel.getFrameId("ee_link")         # END EFFECTOR ID

Jv = ca.Function('Jv', [q], [cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2,:]])

Jv_val = Jv(q)
dP = Jv_val @ ca.vertcat(dq[0],dq[1])

vx_fun = ca.Function("vx_fun", [q, dq], [dP[0]])
vy_fun = ca.Function("vy_fun", [q, dq], [dP[1]])

# oMtool = robot.data.oMf[IDX_EE]
# oRtool = oMtool.rotation
# Jee = oRtool[:2,:2] @ (pin.computeFrameJacobian(robot.model,robot.data,np.array([q1_test[i],q2_test[i]]),IDX_EE)[:2,:2])

jac_fun = ca.Function("jac_fun", [q], [cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)])
manipulability_fun = ca.Function("manipulability_fun", [q], [jac_fun(q).T @ jac_fun(q)])
cost_manipulability_fun = ca.Function("cost_manipulability_fun", [q], [-ca.log(ca.det(manipulability_fun(q)) + 1e-9)])

tau_fun = ca.Function('tau', [q, dq, acc], [cpin.rnea(cmodel, cdata, q, dq, acc)])

energy_expr = 0
for j in range(NV):
    energy_expr += ca.sumsqr(dq[j] * tau_fun(q, dq, acc)[j])
energy_fun = ca.Function('energy', [q, dq, acc], [energy_expr])



# COST VALUES COMPUTING
def cost_function_value(q_var, dq_var):
    f_10 = []
    f_11 = []
    f_2 = []                                                  # MODIF: uncomment
    f_tau = []
    f_energy = []
    # f_3 = []
    # f_4 = []
    N_time = q_var.shape[1]
    if N_time != N:         # display of a message in case the number of time steps differs from the initial problem
        print(f"INFORMATION: the number of the times steps of the trajectories ({N_time}) differs from the initial problem ({N}).")
    for t in range(N_time):
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        # f_10 += dq_var[0, t]**2
        f_10.append(dq_var[0, t]**2)
        # f_11 += dq_var[1, t]**2
        f_11.append(dq_var[1, t]**2)
        # f_2 += vx_fun(q_t, dq_t)**2
        f_2.append(vx_fun(q_t, dq_t)**2)                      # MODIF: uncomment
        
        acc_t = ca.DM.zeros(NV)
        tau_t = tau_fun(q_t, dq_t, acc_t)
        f_tau.append(ca.sumsqr(tau_t))
        f_energy.append(energy_fun(q_t, dq_t, acc_t))
        # f_3 += vy_fun(q_t, dq_t)**2
        # f_4 += cost_manipulability_fun(q_t)
    # return np.array([f_10, f_11, f_2, f_3, f_4])[:,0,0]
    return f_10, f_11, f_2, f_tau, f_energy                                    # MODIF: uncomment and comment next
    # return f_10, f_11



# DOC SOLVING
def solve_DOC(w, x_fin = -0.75, q_init = [0, ca.pi/4]):
    opti = ca.Opti()

    # time constants
    # t_0 = 0
    # t_f = 1
    # N = 50              # number of discretization points
    # dt = (t_f - t_0) / N

    # variable to optimize
    q = opti.variable(N_angles, N)
    dq = opti.variable(N_angles, N)


    # J_10 = 0
    # J_11 = 0
    # J_2 = 0
    # J_3 = 0
    # J_4 = 0
    # for t in range(N):
    #     q_t = ca.vertcat(q[0, t], q[1, t])
    #     dq_t = ca.vertcat(dq[0, t], dq[1, t])
    #     J_10 += dq[0, t]**2
    #     J_11 += dq[1, t]**2
    #     J_2 += vx_fun(q_t, dq_t)**2
    #     J_3 += vy_fun(q_t, dq_t)**2
    #     J_4 += cost_manipulability_fun(q_t)
    J_10, J_11, J_2, J_tau, J_energy = cost_function_value(q, dq)              # MODIF: uncomment and comment next
    # J_10, J_11 = cost_function_value(q, dq)


    # cost function to minimize
    # opti.minimize(w[0] * J_10 + w[1] * J_11 + w[2] * J_2 + w[3] * J_3 + w[4] * J_4)
    J_tot = 0
    for t in range(N):
        J_tot += w[0] * J_10[t] + w[1] * J_11[t] + w[2] * J_2[t]      # MODIF: uncomment and comment next
        if len(w) >= 5:
            J_tot += w[3] * J_tau[t] + w[4] * J_energy[t]
        # J_tot += w[0] * J_10[t] + w[1] * J_11[t]
    opti.minimize(J_tot)

    # Integration constraints
    # opti.subject_to(q_fic[0, N-1] == 0)
    for t in range(N-1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
        opti.subject_to(opti.bounded(-1e-2, q[0, t+1], ca.pi))
        opti.subject_to(opti.bounded(-1e-2, q[1, t+1], 3*ca.pi/4))
    opti.subject_to(opti.bounded(-1e-2, q[0, 0], ca.pi))
    opti.subject_to(opti.bounded(-1e-2, q[1, 0], 3*ca.pi/4))


    # Initial and final contraints
    opti.subject_to(q[:, 0] == q_init)
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == x_fin)

    # additionnal conditions for getting bounded values of q1 and q2
    # opti.subject_to(opti.bounded(-ca.pi/2, q, ca.pi/2))

    # parameter for removing textual output of the solver
    opts = {
        "print_time": False,
        "ipopt.print_level": False,
        # "ipopt.print_level": 5,
        "ipopt.sb": "yes",
    }

    # opti.solver("ipopt")
    opti.solver("ipopt", opts)
    sol = opti.solve()

    q1_cpin = sol.value(q[0,:])
    q2_cpin = sol.value(q[1,:])

    dq1_cpin = sol.value(dq[0,:])
    dq2_cpin = sol.value(dq[1,:])

    results_angles_cpin = np.array([q1_cpin, q2_cpin]).T
    results_vitesses_angulaires_cpin = np.array([dq1_cpin, dq2_cpin]).T

    return results_angles_cpin, results_vitesses_angulaires_cpin



def generate_DOC_solutions_list_xfin(w, list_xfin, list_q_init):
    # nb_xfin = len(liste_xfin)
    number_of_added = 0
    number_of_not_added = 0
    results_angles_list = []
    results_vitesses_angulaires_list = []
    for qinit in list_q_init:
        for xfin in list_xfin:
            # print(q_init, xfin)
            try:
                results_angles_cpin, results_vitesses_angulaires_cpin = solve_DOC(w, x_fin = xfin, q_init=qinit)
                results_angles_list.append(results_angles_cpin)
                results_vitesses_angulaires_list.append(results_vitesses_angulaires_cpin)
                number_of_added+=1
            except:
                number_of_not_added+=1
    print(f"Number of not added: {number_of_not_added}")
    return results_angles_list, results_vitesses_angulaires_list



def generate_DOC_solutions_list_w(array_w, x_fin, q_init, remove_bad_w = True):
    nb_w = array_w.shape[0]
    number_of_added = 0
    number_of_not_added = 0
    results_angles_list = []
    results_vitesses_angulaires_list = []
    good_w = np.empty((0, array_w.shape[1]))
    for ind_w, w in enumerate(array_w):
        # print(q_init, xfin)
        print(f"\rGenerating trajectory {ind_w}/{nb_w}...", end="", flush=True)
        # try:
        print(w)
        results_angles_cpin, results_vitesses_angulaires_cpin = solve_DOC(w = w, x_fin = x_fin, q_init=q_init)
        results_angles_list.append(results_angles_cpin)
        results_vitesses_angulaires_list.append(results_vitesses_angulaires_cpin)
        number_of_added+=1
        print(good_w.shape)
        print(w.T.shape)
        # good_w = np.concatenate((good_w, np.array([w])), axis=0)
        good_w = np.concatenate((good_w, w.T), axis=0)
        # except:
        #     number_of_not_added+=1
    if remove_bad_w:
        array_w = good_w
    print(f"Number of solved cases: {number_of_added}")
    print(f"Number of unsolved cases: {number_of_not_added}")
    return results_angles_list, results_vitesses_angulaires_list, array_w



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
    plt.title(f"q1 values for {number_of_trajectories} DOC solving")
    plt.savefig(f"plots/q1_values_{number_of_trajectories}_trajectories.png")
    plt.show()



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
    plt.title(f"q2 values for {number_of_trajectories} DOC solving")
    plt.savefig(f"plots/q2_values_{number_of_trajectories}_trajectories.png")
    plt.show()
