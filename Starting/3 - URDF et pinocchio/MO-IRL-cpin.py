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


results_angle_cpin, results_vitesses_angulaires_cpin = steps_two_arms_robot_cpin.solve_DOC(w = steps_two_arms_robot_cpin.w_true)


w_MO_IRL = steps_two_arms_robot_cpin.MO_IRL_solve(results_angle_cpin, results_vitesses_angulaires_cpin)


results_angle_cpin_IRL, results_vitesses_angulaires_cpin_IRL = steps_two_arms_robot_cpin.solve_DOC(w = w_MO_IRL)


plt.plot(results_angle_cpin[:,0])
plt.plot(results_angle_cpin_IRL[:,0])
plt.title("$q_1$ from the solved DOC and $q_1$ with IRL weights")
plt.show()


plt.plot(results_angle_cpin[:,1])
plt.plot(results_angle_cpin_IRL[:,1])
plt.title("$q_2$ from the solved DOC and $q_2$ with IRL weights")
plt.show()