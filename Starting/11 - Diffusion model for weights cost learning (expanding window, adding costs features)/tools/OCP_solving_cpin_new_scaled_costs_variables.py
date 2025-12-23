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
matplotlib.use("Agg") # Backend for server/headless
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


# --- ROBOT LOADING ---
# (Assumed to be in the same relative path context)
assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
# assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
urdf = assetsPath + 'mon_robot.urdf'

try:
    # Try loading robot
    if not os.path.exists(urdf):
        # Fallback for code execution without actual URDF file
        raise FileNotFoundError("URDF not found, using dummy dimensions")
    robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
    NQ = robot.model.nq
    NV = robot.model.nv
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
except Exception as e:
    # print(f"Warning: Robot loading failed ({e}). Using generic 2DOF model parameters.")
    NQ = 2
    NV = 2
    # Create a dummy model if pinocchio load fails to allow python compilation
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

# Jacobian and Velocity Functions
Jv = ca.Function('Jv', [q_sym], [cpin.computeFrameJacobian(cmodel, cdata, q_sym, ee_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2,:]])
Jv_val = Jv(q_sym)
dP = Jv_val @ ca.vertcat(dq_sym[0], dq_sym[1])

vx_fun = ca.Function("vx_fun", [q_sym, dq_sym], [dP[0]])
vy_fun = ca.Function("vy_fun", [q_sym, dq_sym], [dP[1]])

# Dynamics (RNEA)
tau_fun = ca.Function('tau', [q_sym, dq_sym, acc_sym], [cpin.rnea(cmodel, cdata, q_sym, dq_sym, acc_sym)])

# Energy Function
energy_expr = 0
for j in range(NV):
    energy_expr += ca.sumsqr(dq_sym[j] * tau_fun(q_sym, dq_sym, acc_sym)[j])
energy_fun = ca.Function('energy', [q_sym, dq_sym, acc_sym], [energy_expr])


def cost_function_value(q_var, dq_var, N_steps):
    """
    Computes raw cost values for each time step.
    """
    f_10 = []
    f_11 = []
    f_2 = []
    f_tau = []
    f_energy = []

    # Check shape consistency
    if q_var.shape[1] != N_steps:
        print(f"Warning: q_var shape {q_var.shape} mismatch with N={N_steps}")

    for t in range(N_steps):
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        
        f_10.append(dq_var[0, t]**2)
        f_11.append(dq_var[1, t]**2)
        f_2.append(vx_fun(q_t, dq_t)**2)
        
        acc_t = ca.DM.zeros(NV) # Zero acceleration approximation for cost calculation
        tau_t = tau_fun(q_t, dq_t, acc_t)
        f_tau.append(ca.sumsqr(tau_t))
        f_energy.append(energy_fun(q_t, dq_t, acc_t))
        
    return f_10, f_11, f_2, f_tau, f_energy


def solve_DOC(w_matrix, N_steps, x_fin=1.9, q_init=[-np.pi/2, np.pi/2], scale_factors=None, verbose=False):
    """
    Solves the OCP.
    w_matrix: shape (5, 3) where columns are weights for 1st, 2nd, 3rd tier of trajectory.
    N_steps: int, number of grid points (length of trajectory).
    """
    opti = ca.Opti()

    if scale_factors is None:
        scale_factors = {"J_10": 1.0, "J_11": 1.0, "J_2": 1.0, "J_tau": 1.0, "J_energy": 1.0}

    # Time step (fixed frequency 100Hz)
    dt = DT_REF 

    # Variables
    q = opti.variable(N_angles, N_steps)
    dq = opti.variable(N_angles, N_steps)

    J_10, J_11, J_2, J_tau, J_energy = cost_function_value(q, dq, N_steps)

    # Indices for the 3 tiers
    idx_tier_1 = int(N_steps / 3)
    idx_tier_2 = int(2 * N_steps / 3)
    
    # Total Cost
    J_tot = 0
    
    # Handle w format: if 1D array provided, broadcast to all tiers (backward compatibility)
    if w_matrix.ndim == 1:
        w_matrix = np.column_stack([w_matrix, w_matrix, w_matrix])

    for t in range(N_steps):
        # Determine which weight vector to use based on time t
        if t < idx_tier_1:
            w = w_matrix[:, 0]
        elif t < idx_tier_2:
            w = w_matrix[:, 1]
        else:
            w = w_matrix[:, 2]

        term_10 = w[0] * (J_10[t] * scale_factors["J_10"])
        term_11 = w[1] * (J_11[t] * scale_factors["J_11"])
        term_2  = w[2] * (J_2[t]  * scale_factors["J_2"])
        term_tau    = w[3] * (J_tau[t]    * scale_factors["J_tau"])
        term_energy = w[4] * (J_energy[t] * scale_factors["J_energy"])
        
        J_tot += term_10 + term_11 + term_2 + term_tau + term_energy

    opti.minimize(J_tot)

    # Integration (Euler)
    for t in range(N_steps - 1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
        
        # Limits
        opti.subject_to(opti.bounded(q1_min_lim_rad, q[0, t+1], q1_max_lim_rad))
        opti.subject_to(opti.bounded(q2_min_lim_rad, q[1, t+1], q2_max_lim_rad))
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[0, t+1], dq_max_lim_rad_par_s))
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[1, t+1], dq_max_lim_rad_par_s))

    # Initial Conditions
    opti.subject_to(q[:, 0] == q_init)
    
    # Final Condition (Forward Kinematics)
    # Using simple trigonometry for 2-link planar
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == x_fin)

    # Solver Setup
    opts = {
        "print_time": False,
        "ipopt.print_level": 0 if not verbose else 5,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        q1_cpin = sol.value(q[0,:])
        q2_cpin = sol.value(q[1,:])
        dq1_cpin = sol.value(dq[0,:])
        dq2_cpin = sol.value(dq[1,:])
        return np.array([q1_cpin, q2_cpin]).T, np.array([dq1_cpin, dq2_cpin]).T
    except Exception as e:
        if verbose: print(f"Optimization failed: {e}")
        return None, None


def compute_scaling_factors(num_samples=20, x_fin=1.9, q_init=[-np.pi/2, np.pi/2]):
    """
    Computes scaling factors using a fixed N=100 (1 second) as reference.
    """
    print(f"--- Computing Scaling Factors ({num_samples} samples) ---")
    raw_costs_sums = {"J_10": [], "J_11": [], "J_2": [], "J_tau": [], "J_energy": []}
    
    # Reference settings for scaling computation
    N_ref = 100 
    
    for i in range(num_samples):
        # Use constant random weights for scaling calibration
        w_random = np.random.rand(5)
        w_random = w_random / np.sum(w_random)
        w_mat = np.column_stack([w_random, w_random, w_random]) # Constant over time

        try:
            res_q, res_dq = solve_DOC(w_mat, N_ref, x_fin=x_fin, q_init=q_init, scale_factors=None)
            
            if res_q is None: continue

            # Re-evaluate raw costs
            for t in range(res_q.shape[0]):
                q_val = ca.DM(res_q[t, :])
                dq_val = ca.DM(res_dq[t, :])
                acc_val = ca.DM.zeros(NV)
                
                raw_costs_sums["J_10"].append(float(dq_val[0]**2))
                raw_costs_sums["J_11"].append(float(dq_val[1]**2))
                raw_costs_sums["J_2"].append(float(vx_fun(q_val, dq_val).full())**2)
                raw_costs_sums["J_tau"].append(float(ca.sumsqr(tau_fun(q_val, dq_val, acc_val)).full()))
                raw_costs_sums["J_energy"].append(float(energy_fun(q_val, dq_val, acc_val).full()))

        except Exception:
            continue

    scale_factors = {}
    print("Computing means...")
    for key, values in raw_costs_sums.items():
        if len(values) == 0:
            scale_factors[key] = 1.0
        else:
            avg_val = np.mean(values)
            # Factor is 1/mean so that weighted term becomes ~1
            scale_factors[key] = 1.0 / avg_val if avg_val > 1e-9 else 1.0
            
    return scale_factors

# (Plotting functions kept as is, omit for brevity unless requested, 
# but you should keep them in the file if you overwrite it)
def plot_trajectory_q1(list_q, list_dq, linestyle='-'):
    plt.figure()
    for q in list_q:
        if q is not None: plt.plot(np.rad2deg(q[:,0]), linestyle=linestyle)
    plt.title("q1 trajectories")
    plt.show()

def plot_trajectory_q2(list_q, list_dq, linestyle='-'):
    plt.figure()
    for q in list_q:
        if q is not None: plt.plot(np.rad2deg(q[:,1]), linestyle=linestyle)
    plt.title("q2 trajectories")
    plt.show()

def plot_trajectory_dq1(list_q, list_dq, linestyle='-'):
    plt.figure()
    for dq in list_dq:
        if dq is not None: plt.plot(np.rad2deg(dq[:,0]), linestyle=linestyle)
    plt.title("dq1 trajectories")
    plt.show()

def plot_trajectory_dq2(list_q, list_dq, linestyle='-'):
    plt.figure()
    for dq in list_dq:
        if dq is not None: plt.plot(np.rad2deg(dq[:,1]), linestyle=linestyle)
    plt.title("dq2 trajectories")
    plt.show()

def plot_trajectory_ee(list_q, x_fin_target=1.9, linestyle='-'):
    """
    Trace la trajectoire de l'effecteur final dans le plan (X, Y).
    """
    plt.figure(figsize=(8, 8))
    
    # Zone de travail (cercle de rayon L1+L2)
    circle = plt.Circle((0, 0), L_1 + L_2, color='k', fill=False, linestyle=':', alpha=0.3)
    plt.gca().add_patch(circle)

    count = 0
    for q in list_q:
        if q is not None:
            # Cinématique Directe
            # q[:, 0] -> q1
            # q[:, 1] -> q2
            q1 = q[:, 0]
            q2 = q[:, 1]
            
            # Formules géométriques simples pour 2-DOF plan
            x = L_1 * np.cos(q1) + L_2 * np.cos(q1 + q2)
            y = L_1 * np.sin(q1) + L_2 * np.sin(q1 + q2)
            
            plt.plot(x, y, linestyle=linestyle, alpha=0.5)
            
            # Marquer le début (vert) et la fin (rouge)
            plt.plot(x[0], y[0], 'g.', markersize=5) 
            plt.plot(x[-1], y[-1], 'r.', markersize=5)
            count += 1

    # Ligne cible théorique (x_fin moyen)
    plt.axvline(x=x_fin_target, color='purple', linestyle='--', linewidth=2, label='Objectif X moyen')
    
    # Base du robot
    plt.plot(0, 0, 'ko', markersize=10, label='Base')
    
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"Trajectoire de l'End-Effector (X, Y) - {count} samples")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()