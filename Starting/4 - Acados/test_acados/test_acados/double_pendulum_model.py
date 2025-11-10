#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

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
    
def export_double_pendulum_ode_model() -> AcadosModel:

    
    # Load model
    robot = load("double_pendulum_simple")
    model = robot.model

    

    cmodel = cpin.Model(model)
    cdata = cdata = cmodel.createData()


    nq = model.nq
    nv = model.nv
        
    
    
    model_name = 'double_pendulum_simple'

    # set up states & controls
    q1      = SX.sym('q1')
    q2   = SX.sym('q2')
    dq1      = SX.sym('dq1')
    dq2  = SX.sym('dq2')

    cx = vertcat(q1, q2, dq1, dq2)

    tau = SX.sym('tau',2)
    cu = vertcat(tau)

    # # xdot
    q1_dot      = SX.sym('q1_dot')
    q2_dot   = SX.sym('q2_dot')
    dq1_dot      = SX.sym('dq1_dot')
    dq2_dot  = SX.sym('dq2_dot')

    xdot = vertcat(q1_dot, q2_dot, dq1_dot, dq2_dot)

    # # dynamics
    
    # * Define the robot dynamics, this creates a relation between the symbolic variables defined above
    acc = casadi.Function(
    "acc",
    [cx, cu],
    [cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], cu)],
    )
    
    f_expl=casadi.vertcat(cx[model.nq:],acc(cx,cu)) # composed of dq and ddq(from aba)
    
    # cos_theta = cos(theta)
    # sin_theta = sin(theta)
    # denominator = M + m - m*cos_theta*cos_theta
    # f_expl = vertcat(v1,
    #                  dtheta,
    #                  (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
    #                  (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
    #                  )

    f_impl = xdot - f_expl

    aca_model = AcadosModel()

    aca_model.f_impl_expr = f_impl
    aca_model.f_expl_expr = f_expl
    aca_model.x = cx
    aca_model.xdot = xdot
    aca_model.u = cu
    aca_model.name = model_name

    # store meta information
    aca_model.x_labels = ['$q1$ [rad]', r'q2 [rad]', r'$\dot{q2}$ [rad/s]', r'$\dot{q2}$ [rad/s]']
    aca_model.u_labels = ['$tau1$','$tau2$']
    aca_model.t_label = '$t$ [s]'

    return robot, aca_model

