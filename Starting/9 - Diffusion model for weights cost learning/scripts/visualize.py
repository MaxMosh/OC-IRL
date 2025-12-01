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


# ROBOT LOADING
assetsPath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/9 - Diffusion model for weights cost learning/assets/'
urdf = assetsPath + 'mon_robot.urdf'
robot = RobotWrapper.BuildFromURDF(urdf, [assetsPath,])
# robot.setVisualizer(GepettoVisualizer()) # AJOUT
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv

q0 = pin.neutral(robot.model)
print(type(q0))
print(f"q0: {q0}")
viz = robot.viewer
robot.display(q0)
input()


result_angles = np.load("data/array_results_angles_11.npy")
for traj in result_angles:
    for current_angles in traj:
        robot.display(current_angles)
        time.sleep(0.01)
    input()