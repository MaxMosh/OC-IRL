from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import example_robot_data as erd
from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
import casadi
import matplotlib.pyplot as plt
import time 

from pinocchio.visualize import MeshcatVisualizer
from example_robot_data import load

import meshcat

from pinocchio.robot_wrapper import RobotWrapper


def convert_my_double_joints_urdf():
    myURDFpath = '/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/3 - URDF et pinocchio/assets/'
    urdf = myURDFpath + 'mon_robot.urdf'
    robot = RobotWrapper.BuildFromURDF( urdf, [ myURDFpath, ] )
    model = robot.model

    cmodel = cpin.Model(model)
    cdata = cdata = cmodel.createData()

    nq = model.nq
    nv = model.nv

    model_name = 'double_joints'

    # set up states & controls
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    dq1 = SX.sym('dq1')
    dq2 = SX.sym('dq2')

    cx = vertcat(q1, q2, dq1, dq2)

    tau = SX.sym('tau',2)
    cu = vertcat(tau)

    # # xdot
    q1_dot = SX.sym('q1_dot')
    q2_dot = SX.sym('q2_dot')
    dq1_dot = SX.sym('dq1_dot')
    dq2_dot = SX.sym('dq2_dot')

    xdot = vertcat(q1_dot, q2_dot, dq1_dot, dq2_dot)

    acc = casadi.Function(
    "acc",
    [cx, cu],
    [cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], cu)],
    )

    f_expl=casadi.vertcat(cx[model.nq:],acc(cx,cu)) # composed of dq and ddq(from aba)

    f_impl = xdot - f_expl

    aca_model = AcadosModel()

    aca_model.f_impl_expr = f_impl
    aca_model.f_expl_expr = f_expl
    aca_model.x = cx
    aca_model.xdot = xdot
    aca_model.u = cu
    aca_model.name = model_name

    # store meta information
    # NOTE : je pense qu'il y a une coquille sur le label \dot{q2}
    aca_model.x_labels = ['$q1$ [rad]', r'q2 [rad]', r'$\dot{q2}$ [rad/s]', r'$\dot{q2}$ [rad/s]']
    aca_model.u_labels = ['$tau1$','$tau2$']
    aca_model.t_label = '$t$ [s]'

    return robot, aca_model