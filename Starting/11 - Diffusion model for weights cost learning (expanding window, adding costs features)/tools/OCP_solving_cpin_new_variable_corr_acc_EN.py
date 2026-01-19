# PACKAGES IMPORT
import pinocchio as pin
import math
import time
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
import eigenpy
import numpy as np

import casadi as ca
from pinocchio import casadi as cpin

import matplotlib
import matplotlib.pyplot as plt
import os

# --- CONSTANTS & CONFIGURATION ---
FREQ = 100.0  # 100 Hz requested
DT_REF = 1.0 / FREQ

# Lengths
L_1 = 1
L_2 = 1
N_angles = 2

# Limits
q1_min_lim_deg, q1_max_lim_deg = -135, 90
q2_min_lim_deg, q2_max_lim_deg = 0, 160
q1_min_lim_rad, q1_max_lim_rad = np.deg2rad(q1_min_lim_deg), np.deg2rad(q1_max_lim_deg)
q2_min_lim_rad, q2_max_lim_rad = np.deg2rad(q2_min_lim_deg), np.deg2rad(q2_max_lim_deg)

dq_min_lim_deg_par_s, dq_max_lim_deg_par_s = -800, 800
dq_min_lim_rad_par_s, dq_max_lim_rad_par_s = np.deg2rad(dq_min_lim_deg_par_s), np.deg2rad(dq_max_lim_deg_par_s)

# Acceleration limits
ddq_min_lim_rad_par_s2, ddq_max_lim_rad_par_s2 = -100.0, 100.0

# --- ROBOT LOADING ---
assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
urdf = assetsPath + 'mon_robot.urdf'

try:
    if not os.path.exists(urdf):
        raise FileNotFoundError("URDF not found, using dummy dimensions")
    robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
    NQ = robot.model.nq
    NV = robot.model.nv
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
except Exception as e:
    NQ = 2
    NV = 2
    model = pin.Model()
    model.addJoint(0, pin.JointModelRX(), pin.SE3.Identity(), "j1")
    model.addJoint(1, pin.JointModelRX(), pin.SE3.Identity(), "j2")
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

# --- CASADI FUNCTIONS ---
q_sym = ca.SX.sym("q", NQ, 1)
dq_sym = ca.SX.sym("dq", NQ, 1)
acc_sym = ca.SX.sym("acc", NV, 1)

ee_frame_id = cmodel.getFrameId("ee_link") if hasattr(cmodel, "getFrameId") else 2

Jv = ca.Function('Jv', [q_sym], [cpin.computeFrameJacobian(cmodel, cdata, q_sym, ee_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2,:]])
Jv_val = Jv(q_sym)
dP = Jv_val @ ca.vertcat(dq_sym[0], dq_sym[1])

vx_fun = ca.Function("vx_fun", [q_sym, dq_sym], [dP[0]])
vy_fun = ca.Function("vy_fun", [q_sym, dq_sym], [dP[1]])

# RNEA uses the real acceleration 'acc_sym'
tau_fun = ca.Function('tau', [q_sym, dq_sym, acc_sym], [cpin.rnea(cmodel, cdata, q_sym, dq_sym, acc_sym)])

energy_expr = 0
for j in range(NV):
    # Energy/Power term: dq * tau
    energy_expr += ca.sumsqr(dq_sym[j] * tau_fun(q_sym, dq_sym, acc_sym)[j])
energy_fun = ca.Function('energy', [q_sym, dq_sym, acc_sym], [energy_expr])


def cost_function_value(q_var, dq_var, ddq_var, N_steps):
    """
    Computes raw cost values for each time step.
    Uses ddq_var to compute dynamic torque.
    """
    f_10, f_11, f_2, f_tau, f_energy = [], [], [], [], []

    for t in range(N_steps):
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        acc_t = ca.vertcat(ddq_var[0, t], ddq_var[1, t])
        
        f_10.append(dq_var[0, t]**2)
        f_11.append(dq_var[1, t]**2)
        f_2.append(vx_fun(q_t, dq_t)**2)
        
        tau_t = tau_fun(q_t, dq_t, acc_t)
        f_tau.append(ca.sumsqr(tau_t))
        f_energy.append(energy_fun(q_t, dq_t, acc_t))
        
    return f_10, f_11, f_2, f_tau, f_energy


def solve_DOC(w_matrix, N_steps, x_fin=1.9, q_init=[-np.pi/2, np.pi/2], verbose=False):
    """
    Solves the Optimal Control Problem with Acceleration as a decision variable.
    """
    opti = ca.Opti()
    dt = DT_REF 

    # Variables
    q = opti.variable(N_angles, N_steps)
    dq = opti.variable(N_angles, N_steps)
    ddq = opti.variable(N_angles, N_steps)

    J_10, J_11, J_2, J_tau, J_energy = cost_function_value(q, dq, ddq, N_steps)

    # Multi-phase cost handling
    idx_tier_1 = int(N_steps / 3)
    idx_tier_2 = int(2 * N_steps / 3)
    J_tot = 0
    
    if w_matrix.ndim == 1:
        w_matrix = np.column_stack([w_matrix, w_matrix, w_matrix])

    for t in range(N_steps):
        if t < idx_tier_1:
            w = w_matrix[:, 0]
        elif t < idx_tier_2:
            w = w_matrix[:, 1]
        else:
            w = w_matrix[:, 2]

        J_tot += (w[0] * J_10[t] + w[1] * J_11[t] + w[2] * J_2[t] + 
                  w[3] * J_tau[t] + w[4] * J_energy[t])

    opti.minimize(J_tot)

    # Dynamics & Constraints
    for t in range(N_steps - 1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
        opti.subject_to(dq[:, t+1] == dq[:, t] + dt * ddq[:, t])
        
        opti.subject_to(opti.bounded(q1_min_lim_rad, q[0, t+1], q1_max_lim_rad))
        opti.subject_to(opti.bounded(q2_min_lim_rad, q[1, t+1], q2_max_lim_rad))
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[0, t+1], dq_max_lim_rad_par_s))
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[1, t+1], dq_max_lim_rad_par_s))

    # Boundary Conditions
    opti.subject_to(q[:, 0] == q_init)
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == x_fin)

    opts = {
        "print_time": False,
        "ipopt.print_level": 0 if not verbose else 5,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        return (np.array([sol.value(q[0,:]), sol.value(q[1,:])]).T, 
                np.array([sol.value(dq[0,:]), sol.value(dq[1,:])]).T,
                np.array([sol.value(ddq[0,:]), sol.value(ddq[1,:])]).T)
    except Exception as e:
        if verbose: print(f"Optimization failed: {e}")
        return None, None, None


# --- PLOTTING FUNCTIONS ---

def plot_joint_dynamics(joint_id, list_q, list_dq, list_ddq=None, linestyle='-', save_path=None):
    """
    Plots q, dq, and ddq for a given joint.
    """
    rows = 3 if list_ddq is not None else 2
    fig, axs = plt.subplots(rows, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Joint {joint_id + 1} Dynamics")

    # 1. Position
    for q in list_q:
        if q is not None: axs[0].plot(np.rad2deg(q[:, joint_id]), linestyle=linestyle, alpha=0.7)
    axs[0].set_ylabel("Position [deg]")
    axs[0].grid(True)

    # 2. Velocity
    for dq in list_dq:
        if dq is not None: axs[1].plot(np.rad2deg(dq[:, joint_id]), linestyle=linestyle, alpha=0.7)
    axs[1].set_ylabel("Velocity [deg/s]")
    axs[1].grid(True)

    # 3. Acceleration
    if list_ddq is not None:
        for ddq in list_ddq:
            if ddq is not None: axs[2].plot(np.rad2deg(ddq[:, joint_id]), linestyle=linestyle, alpha=0.7)
        axs[2].set_ylabel("Acceleration [deg/s²]")
        axs[2].set_xlabel("Time steps")
        axs[2].grid(True)
    else:
        axs[1].set_xlabel("Time steps")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    if matplotlib.get_backend() != 'Agg':
        plt.show()
    plt.close(fig)


def plot_trajectory_q1(list_q, list_dq, list_ddq=None, linestyle='-', save_path='data/trajectory_q1.png'):
    plot_joint_dynamics(0, list_q, list_dq, list_ddq, linestyle, save_path)


def plot_trajectory_q2(list_q, list_dq, list_ddq=None, linestyle='-', save_path='data/trajectory_q2.png'):
    plot_joint_dynamics(1, list_q, list_dq, list_ddq, linestyle, save_path)


def plot_trajectory_ee(list_q, x_fin_target=1.9, linestyle='-', save_path='data/trajectory_ee.png'):
    fig = plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), L_1 + L_2, color='k', fill=False, linestyle=':', alpha=0.3)
    plt.gca().add_patch(circle)
    
    count = 0
    for q in list_q:
        if q is not None:
            q1, q2 = q[:, 0], q[:, 1]
            x = L_1 * np.cos(q1) + L_2 * np.cos(q1 + q2)
            y = L_1 * np.sin(q1) + L_2 * np.sin(q1 + q2)
            plt.plot(x, y, linestyle=linestyle, alpha=0.5)
            plt.plot(x[0], y[0], 'g.', markersize=5) 
            plt.plot(x[-1], y[-1], 'r.', markersize=5)
            count += 1
            
    plt.axvline(x=x_fin_target, color='purple', linestyle='--', linewidth=2, label='Target X-axis')
    plt.plot(0, 0, 'ko', markersize=10, label='Base')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"End-Effector Trajectories - {count} samples")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")

    if matplotlib.get_backend() != 'Agg':
        plt.show()
    plt.close(fig)