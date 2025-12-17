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
import time



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

# parameters of the loaded .npy files
dq_max_lim_deg_par_s = 800
r = 11


# ROBOT LOADING
assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/11 - Diffusion model for weights cost learning (expanding window, adding costs features)/assets/'
urdf = assetsPath + 'mon_robot.urdf'
robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
# robot.setVisualizer(GepettoVisualizer()) # AJOUT
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv

q0 = pin.neutral(robot.model)
# print(type(q0))
print(f"q0: {q0}")
viz = robot.viewer
robot.display(q0)
input()


result_angles = np.load(f"data/array_results_angles_simplex_21_lim_joint_velocities_{dq_max_lim_deg_par_s}.npy")
array_w = np.load(f"data/array_w_simplex_21_lim_joint_velocities_{dq_max_lim_deg_par_s}.npy")
for ind_traj, traj in enumerate(result_angles):
    print(f"current w: {array_w[ind_traj]}")
    for current_angles in traj:
        robot.display(current_angles)
        time.sleep(0.01)
    input()