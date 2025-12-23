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

import matplotlib.pyplot as plt
import os

# CONSTANTS
# time constants
t_0 = 0
t_f = 1
N = 50                  # number of discretization points
dt = (t_f - t_0) / N

# lengths
L_1 = 1
L_2 = 1
N_angles = 2


# ROBOT LOADING
print(os.getcwd())
# Update the path to your assets folder if necessary
assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
urdf = assetsPath + 'mon_robot.urdf'

try:
    robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
    robot.initViewer(loadModel=True)
    NQ = robot.model.nq
    NV = robot.model.nv
    q0 = pin.neutral(robot.model)
    # viz = robot.viewer
    # robot.display(q0)
except Exception as e:
    print(f"Warning: Robot loading failed or viewer issue. Ensure paths are correct. {e}")
    # Fallback for dimensions if robot fails to load (assuming 2DOF)
    NQ = 2
    NV = 2

# MODEL FOR CPIN
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

q = ca.SX.sym("q",NQ,1)
dq = ca.SX.sym("dq",NQ,1)
acc = ca.SX.sym("acc", NV, 1)

# COSTS FEATURES FUNCTIONS
ee_frame_id = cmodel.getFrameId("ee_link")         # END EFFECTOR ID

Jv = ca.Function('Jv', [q], [cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2,:]])
Jv_val = Jv(q)
dP = Jv_val @ ca.vertcat(dq[0],dq[1])

vx_fun = ca.Function("vx_fun", [q, dq], [dP[0]])
vy_fun = ca.Function("vy_fun", [q, dq], [dP[1]])

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
    f_2 = []
    f_tau = []
    f_energy = []

    N_time = q_var.shape[1]
    if N_time != N:
        print(f"INFORMATION: the number of the times steps of the trajectories ({N_time}) differs from the initial problem ({N}).")
    
    for t in range(N_time):
        q_t = ca.vertcat(q_var[0, t], q_var[1, t])
        dq_t = ca.vertcat(dq_var[0, t], dq_var[1, t])
        
        f_10.append(dq_var[0, t]**2)
        f_11.append(dq_var[1, t]**2)
        f_2.append(vx_fun(q_t, dq_t)**2)
        
        acc_t = ca.DM.zeros(NV)
        tau_t = tau_fun(q_t, dq_t, acc_t)
        f_tau.append(ca.sumsqr(tau_t))
        f_energy.append(energy_fun(q_t, dq_t, acc_t))
        
    return f_10, f_11, f_2, f_tau, f_energy


# DOC SOLVING CONSTANTS
q1_min_lim_deg, q1_max_lim_deg = -135, 90
q2_min_lim_deg, q2_max_lim_deg = 0, 160
q1_min_lim_rad, q1_max_lim_rad = (2*np.pi/360)*q1_min_lim_deg, (2*np.pi/360)*q1_max_lim_deg
q2_min_lim_rad, q2_max_lim_rad = (2*np.pi/360)*q2_min_lim_deg, (2*np.pi/360)*q2_max_lim_deg

dq_min_lim_deg_par_s, dq_max_lim_deg_par_s = -800, 800
dq_min_lim_rad_par_s, dq_max_lim_rad_par_s = (2*np.pi/360)*dq_min_lim_deg_par_s ,(2*np.pi/360)*dq_max_lim_deg_par_s


def compute_scaling_factors(num_samples=10, x_fin=1.9, q_init=[-np.pi/2, np.pi/2]):
    """
    Computes normalization factors by averaging raw costs over several random trajectories.
    This ensures that J_tau (approx 1000) doesn't dominate J_velocity (approx 10).
    """
    print(f"--- Computing Scaling Factors (using {num_samples} samples) ---")
    
    raw_costs_sums = {"J_10": [], "J_11": [], "J_2": [], "J_tau": [], "J_energy": []}

    for i in range(num_samples):
        # Random weights on simplex
        w_random = np.random.rand(5)
        w_random = w_random / np.sum(w_random)
        
        try:
            # Solve with unit scales initially to get raw physical values
            res_q, res_dq = solve_DOC(w_random, x_fin=x_fin, q_init=q_init, scale_factors=None, verbose=False)
            
            # Recalculate raw costs (summed over time)
            # We use CasADi numeric types (DM) for evaluation
            total_costs = {"J_10": 0, "J_11": 0, "J_2": 0, "J_tau": 0, "J_energy": 0}
            
            for t in range(res_q.shape[0]):
                q_val = ca.DM(res_q[t, :])
                dq_val = ca.DM(res_dq[t, :])
                acc_val = ca.DM.zeros(NV)
                
                total_costs["J_10"] += float(dq_val[0]**2)
                total_costs["J_11"] += float(dq_val[1]**2)
                total_costs["J_2"]  += float(vx_fun(q_val, dq_val).full())**2
                total_costs["J_tau"] += float(ca.sumsqr(tau_fun(q_val, dq_val, acc_val)).full())
                total_costs["J_energy"] += float(energy_fun(q_val, dq_val, acc_val).full())

            for key in raw_costs_sums:
                raw_costs_sums[key].append(abs(total_costs[key]))

        except Exception as e:
            # print(f"Sample {i} failed: {e}")
            continue

    scale_factors = {}
    print("\n--- Scaling Factors Results ---")
    for key, values in raw_costs_sums.items():
        if len(values) == 0:
            scale_factors[key] = 1.0
        else:
            avg_val = np.mean(values)
            scale_factors[key] = 1.0 / avg_val if avg_val > 1e-6 else 1.0
        print(f"Avg {key}: {1.0/scale_factors[key]:.4f} \t-> Scale: {scale_factors[key]:.6f}")

    return scale_factors


def solve_DOC(w, x_fin=1.9, q_init=[-np.pi/2, np.pi/2], scale_factors=None, verbose=False):
    opti = ca.Opti()

    # Default scaling if None provided (weights apply to raw values)
    if scale_factors is None:
        scale_factors = {"J_10": 1.0, "J_11": 1.0, "J_2": 1.0, "J_tau": 1.0, "J_energy": 1.0}

    # Variables to optimize
    q = opti.variable(N_angles, N)
    dq = opti.variable(N_angles, N)

    J_10, J_11, J_2, J_tau, J_energy = cost_function_value(q, dq)

    # Total Cost Calculation with Scaling
    J_tot = 0
    for t in range(N):
        # Apply scaling factor to normalize orders of magnitude, then apply weight w
        term_10 = w[0] * (J_10[t] * scale_factors["J_10"])
        term_11 = w[1] * (J_11[t] * scale_factors["J_11"])
        term_2  = w[2] * (J_2[t]  * scale_factors["J_2"])
        
        J_tot += term_10 + term_11 + term_2

        if len(w) >= 5:
            term_tau    = w[3] * (J_tau[t]    * scale_factors["J_tau"])
            term_energy = w[4] * (J_energy[t] * scale_factors["J_energy"])
            J_tot += term_tau + term_energy

    opti.minimize(J_tot)

    # Integration constraints (Euler)
    for t in range(N-1):
        opti.subject_to(q[:, t+1] == q[:, t] + dt * dq[:, t])
        
        # Position limits
        opti.subject_to(opti.bounded(q1_min_lim_rad, q[0, t+1], q1_max_lim_rad))
        opti.subject_to(opti.bounded(q2_min_lim_rad, q[1, t+1], q2_max_lim_rad))

        # Velocity limits
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[0, t+1], dq_max_lim_rad_par_s))
        opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[1, t+1], dq_max_lim_rad_par_s))
    
    # Start limits
    opti.subject_to(opti.bounded(q1_min_lim_rad, q[0, 0], q1_max_lim_rad))
    opti.subject_to(opti.bounded(q2_min_lim_rad, q[1, 0], q2_max_lim_rad))
    opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[0, 0], dq_max_lim_rad_par_s))
    opti.subject_to(opti.bounded(dq_min_lim_rad_par_s, dq[1, 0], dq_max_lim_rad_par_s))

    # Boundary conditions
    opti.subject_to(q[:, 0] == q_init)
    opti.subject_to(L_1*ca.cos(q[0, -1]) + L_2*ca.cos(q[0, -1] + q[1, -1]) == x_fin)

    # Solver options
    opts = {
        "print_time": False,
        "ipopt.print_level": 0 if not verbose else 5,
        "ipopt.sb": "yes",
    }

    opti.solver("ipopt", opts)
    sol = opti.solve()

    q1_cpin = sol.value(q[0,:])
    q2_cpin = sol.value(q[1,:])
    dq1_cpin = sol.value(dq[0,:])
    dq2_cpin = sol.value(dq[1,:])

    results_angles_cpin = np.array([q1_cpin, q2_cpin]).T
    results_vitesses_angulaires_cpin = np.array([dq1_cpin, dq2_cpin]).T

    return results_angles_cpin, results_vitesses_angulaires_cpin


def generate_DOC_solutions_list_w(array_w, x_fin, q_init, scale_factors=None, remove_bad_w=True):
    nb_w = array_w.shape[0]
    number_of_added = 0
    number_of_not_added = 0
    results_angles_list = []
    results_vitesses_angulaires_list = []
    good_w = np.empty((0, array_w.shape[1]))
    
    for ind_w, w in enumerate(array_w):
        print(f"\rGenerating trajectory {ind_w+1}/{nb_w}...", end="", flush=True)
        try:
            results_angles_cpin, results_vitesses_angulaires_cpin = solve_DOC(
                w=w, x_fin=x_fin, q_init=q_init, scale_factors=scale_factors
            )
            results_angles_list.append(results_angles_cpin)
            results_vitesses_angulaires_list.append(results_vitesses_angulaires_cpin)
            number_of_added += 1
            # print(good_w.shape)
            # print(w.T.shape)
            good_w = np.concatenate((good_w, w.T), axis=0) # Fixed concatenation
        except Exception:
            number_of_not_added += 1

    if remove_bad_w:
        array_w = good_w
        
    print(f"\nNumber of solved cases: {number_of_added}")
    print(f"Number of unsolved cases: {number_of_not_added}")
    
    return results_angles_list, results_vitesses_angulaires_list, array_w


# Plotting functions remain unchanged but included for completeness
def plot_trajectory_q1(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_angles_cpin)
    plt.figure()
    for i in range(number_of_trajectories):
        plt.plot((360/(2*np.pi))*results_angles_cpin[i][:,0], linestyle = linestyle)
    plt.axhline(y=q1_min_lim_deg, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=q1_max_lim_deg, color='red', linestyle='--', linewidth=2)
    plt.title(f"q1 values for {number_of_trajectories} trajectories")
    plt.savefig(f"plots/q1_values_{number_of_trajectories}_trajectories_lim_joint_velocities_{dq_max_lim_deg_par_s}_scaled_costs.png")
    plt.show()

def plot_trajectory_q2(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_angles_cpin)
    plt.figure()
    for i in range(number_of_trajectories):
        plt.plot((360/(2*np.pi))*results_angles_cpin[i][:,1], linestyle = linestyle)
    plt.axhline(y=q2_min_lim_deg, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=q2_max_lim_deg, color='red', linestyle='--', linewidth=2)
    plt.title(f"q2 values for {number_of_trajectories} trajectories")
    plt.savefig(f"plots/q2_values_{number_of_trajectories}_trajectories_lim_joint_velocities_{dq_max_lim_deg_par_s}_scaled_costs.png")
    plt.show()

def plot_trajectory_dq1(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_vitesses_angulaires_cpin)
    plt.figure()
    for i in range(number_of_trajectories):
        plt.plot((360/(2*np.pi))*results_vitesses_angulaires_cpin[i][:,0], linestyle = linestyle)
    plt.axhline(y=dq_min_lim_deg_par_s, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=dq_max_lim_deg_par_s, color='red', linestyle='--', linewidth=2)
    plt.title(f"dq1 values for {number_of_trajectories} trajectories")
    plt.savefig(f"plots/dq1_values_{number_of_trajectories}_trajectories_lim_joint_velocities_{dq_max_lim_deg_par_s}_scaled_costs.png")
    plt.show()

def plot_trajectory_dq2(results_angles_cpin, results_vitesses_angulaires_cpin, linestyle = '-'):
    number_of_trajectories = len(results_vitesses_angulaires_cpin)
    plt.figure()
    for i in range(number_of_trajectories):
        plt.plot((360/(2*np.pi))*results_vitesses_angulaires_cpin[i][:,1], linestyle = linestyle)
    plt.axhline(y=dq_min_lim_deg_par_s, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=dq_max_lim_deg_par_s, color='red', linestyle='--', linewidth=2)
    plt.title(f"dq2 values for {number_of_trajectories} trajectories")
    plt.savefig(f"plots/dq2_values_{number_of_trajectories}_trajectories_lim_joint_velocities_{dq_max_lim_deg_par_s}_scaled_costs.png")
    plt.show()