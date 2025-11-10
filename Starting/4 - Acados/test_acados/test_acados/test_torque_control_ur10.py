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

# Load model
robot = load("ur10")
model = robot.model

visual_model = robot.visual_model
collision_model = robot.collision_model
data = robot.data

cmodel = cpin.Model(model)
cdata = cdata = cmodel.createData()

ee_frame_name = "tool0"
ee_frame_id = model.getFrameId(ee_frame_name)
nq = model.nq
nv = model.nv
q0=pin.neutral(model)

 

# -----------------------
# PIN+CASADI model definition
# -----------------------
# We will define a state x = (q, v)^T to describe the robot dynamics
nx = nq +  nv     # state dimension: positions and velocities
ndx = 2 * nv  # state derivative 
nu =  nv#        # control dimension: the accelerations


qd=q0.copy()
qd=qd+0.001
qd[0]=np.pi/2
qd[1]=np.pi/2
qd[2]=-np.pi/2
# Forward kinematics
pin.forwardKinematics(robot.model, data, qd)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
pin.updateFramePlacements(robot.model, data)                   # Update the frames placement to symbolic expressions in data
        
desired_pose = data.oMf[ee_frame_id].copy()  # Desired EE pose

    



# * Create casadi symbolic variables
# These variables are used to define symbolic expression and are replaced in the solver by some values according to the decision variables
cx = casadi.SX.sym("x", nx, 1) # state: the positions and velocities
cdx = casadi.SX.sym("dx", ndx, 1)
cu = casadi.SX.sym("u", nu, 1) # control: the torques


 # because acados needs different name for state derivative
q_dot = casadi.SX.sym("q_dot", nv, 1)
dq_dot = casadi.SX.sym("dq_dot", nv, 1)


# Forward kinematics
cpin.forwardKinematics(cmodel, cdata, cx[:nq], cx[nq:])  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data
        
def symbolic_log3(R):
    """Symbolic log3 for a CasADi rotation matrix."""
    cos_theta = (casadi.trace(R) - 1) / 2
    cos_theta = casadi.fmin(casadi.fmax(cos_theta, -1), 1)  # clamp to [-1, 1]
    theta = casadi.acos(cos_theta)

    # To avoid division by zero:
    if isinstance(theta, casadi.SX) or isinstance(theta, casadi.MX):
        near_zero = casadi.logic_and(theta < 1e-6, theta > -1e-6)
    else:
        near_zero = abs(theta) < 1e-6

    # Skew-symmetric part
    omega_hat = (R - R.T) / (2 * casadi.sin(theta))

    # Extract vector from skew-symmetric matrix
    omega = casadi.vertcat(omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0])

    # Handle small angle limit
    omega = casadi.if_else(near_zero, casadi.SX.zeros(3), theta * omega)

    return omega
    
def approx_log6(id, M,cx,cu):
    # Update data symbolically inside the function
    cpin.forwardKinematics(cmodel, cdata, cx[:nq], cx[nq:],cu)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
    cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data
    
    tran_error = cdata.oMf[id].translation - M.translation
    #rot_error = casadi.diag(self.cdata.oMf[id].rotation.T @ M.rotation)-casadi.SX.ones(3)
    
    R_err = cdata.oMf[id].rotation.T @ M.rotation
    rot_error = symbolic_log3(R_err)  # returns a 3D vector in so(3)
    
    return(casadi.vertcat(tran_error, rot_error)) 


cost_to_target=[]
cost_to_target.append(approx_log6(ee_frame_id, desired_pose,cx,cu))
# cost_to_target.append(self.approx_log6(param.FOI_to_set_Id[1], Mtarget,cx,cu))

cost_to_target = casadi.Function('cost_to_target', [cx, cu], [casadi.vertcat( *cost_to_target) ])

# Desired end-effector position function 
def tran_error(id, M,cx):
    # Update data symbolically inside the function
    cpin.forwardKinematics(cmodel, cdata, cx[:robot.model.nq])#, cx[model.nq:])#,cu)   
    cpin.updateFramePlacements(cmodel, cdata)                  


    tran_error =  cdata.oMf[id].translation - M.translation
    
    return tran_error 

#cost_to_target=tran_error(ee_frame_id, desired_pose,cx)#.T@tran_error(ee_frame_id, desired_pose,model.x)
#cost_to_target = casadi.Function('cost_to_target', [cx], [cost_to_target])




 # * Define a function to get the next state from the robot dynamics 
 
# * Define the robot dynamics, this creates a relation between the symbolic variables defined above
acc = casadi.Function(
    "acc",
    [cx, cu],
    [cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], cu)],
)
 

# -------------------------------------
# ACADOS Optimal Control Problem (OCP)
# -------------------------------------
ocp = AcadosOcp()



f_expl=casadi.vertcat(cx[model.nq:],acc(cx,cu)) # composed of dq and ddq(from aba)
xdot=casadi.vertcat(q_dot,dq_dot)

f_impl= xdot - f_expl
ocp.model.f_impl_expr = f_impl

ocp.model.f_expl_expr = f_expl 
 
ocp.model.p = []


ocp.model.name = "ur10_min_torque"
ocp.model.x = cx
ocp.model.u = cu
ocp.model.xdot = xdot

 

# # Time horizon
N=200
Tf=2
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf


# cost matrices
gain_q= 1e-1
gain_dq= 1e-2#1e-1
q_mat = gain_q*np.diag(np.ones(6)) 
dq_mat = gain_dq*np.diag(np.ones(6)) 

    
tau_mat = 1e-3*np.eye(6) 

p_mat= 1e3*np.eye(3) #1e3*np.eye(3) 

# # path cost
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = casadi.vertcat( cx[:model.nq], cx[model.nq:], cu, cost_to_target(cx,cu)[0:3])
ocp.cost.yref = np.concatenate( [np.zeros(nv), np.zeros(nv), np.zeros(nv), np.zeros(3)] )#0.707*np.ones((model.nq))
ocp.cost.W = casadi.diagcat(q_mat, dq_mat, tau_mat, p_mat).full()#casadi.diagcat(Q_mat, R_mat).full()

# # terminal cost
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.model.cost_y_expr_e = casadi.vertcat(cost_to_target(cx,cu)[0:3])#cx[:model.nq]) 
ocp.cost.yref_e =  np.zeros(3)#qd# 
ocp.cost.W_e = 100*p_mat

 

# Initial guess      
ocp.constraints.x0 = np.concatenate( [q0,np.zeros(nv)] )

epsilon = 0.005  # 0.5cm tolerance
ocp.model.nh_e = 3
ocp.model.con_h_expr_e = casadi.vertcat(cost_to_target(cx,cu)[0:3])
ocp.constraints.lh_e = -epsilon * np.ones(3)
ocp.constraints.uh_e = +epsilon * np.ones(3)


# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
#ocp.solver_options.qp_solver_cond_N = 5  # horizon is long and DOF is modest, partial condensing makes the QP smaller

ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
#ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
#ocp.solver_options.ext_cost_num_hess = 1  # 1 = exact Hessian from CasADi
  # Relax line search to allow more aggressive steps
#ocp.solver_options.line_search_use_sufficient_descent = 1

# NLP options  
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
ocp.solver_options.regularize_method = 'MIRROR' # if SQP tehn regularize the hessian vaialble are NO_REGULARIZE, MIRROR, PROJECT, PROJECT_REDUC_HESS, CONVEXIFY, GERSHGORIN_LEVENBERG_MARQUARDT.
ocp.solver_options.reg_epsilon = 1e-6#1e-6
ocp.solver_options.nlp_solver_tol_stat = 5e-3
ocp.solver_options.nlp_solver_tol_eq   = 5e-3
ocp.solver_options.nlp_solver_tol_ineq = 5e-3  

ocp.solver_options.nlp_solver_max_iter = 2000

  
ocp.solver_options.print_level = 1

# Create solver
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")#, build=False, generate=False)

 

# Solve
t0 = time.time()
status = ocp_solver.solve()
elapsed = time.time() - t0
print(f"Solve time: {elapsed*1e3:.3f} ms")

if status != 0:
    print("❌ Solver failed with status:", status)
else:
    print("✅ Solver succeeded")


    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))
    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")


    # # Plot each state over time
    # plt.figure(figsize=(10, 6))
    # for i in range(nq):
    #     plt.plot(simX[:, i], label=f'q[{i}]')

    # plt.xlabel('Time step')
    # plt.ylabel('State value')
    # plt.title('State trajectory from acados simulation')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # Forward kinematics
    pin.forwardKinematics(robot.model, data, simX[-1,:robot.model.nq])  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
    pin.updateFramePlacements(robot.model, data)                   # Update the frames placement to symbolic expressions in data
         
    actual_pose = data.oMf[ee_frame_id].copy()  # Desired EE pose

    print("desured ee pose")
    print(desired_pose.translation)
    
    
    print("actual ee pose")
    print(actual_pose.translation)
    
    print("position error")
    print(desired_pose.translation-actual_pose.translation)
    
        
    print("meshcat animation...")
    
    
    # Print all joints with their IDs
    # print("\nJoint list:")
    # for joint_id in range(model.njoints):
    #     joint_name = model.names[joint_id]
    #     joint_type = type(model.joints[joint_id]).__name__
    #     print(f"ID: {joint_id:2d}, Name: {joint_name:15s}, Type: {joint_type}")     
    

    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    #viz.initViewer(open=True)  # open=True opens browser automatically

    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.loadViewerModel()
    #

    # native_viz = viz.viewer
    # native_viz["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
    # native_viz["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])  # Same color → flat background
    # grid_height = -1.0  # Negative z = lower the grid
    # native_viz["/Grid"].set_transform(
    # np.array([
    #     [1, 0, 0, 0],  # Rotation (identity)
    #     [0, 1, 0, 0],
    #     [0, 0, 1, grid_height],
    #     [0, 0, 0, 1]
    # ]))
        
    #driver = webdriver.Safari()
    #driver.get(viz.viewer.url())  # MeshCat URL

    for i in range(N):
    
        #print("samples "+str(i))
        q0=pin.neutral(model) 
        pin.forwardKinematics(model, data,  simX[-1, :model.nq])
        pin.updateFramePlacements(model, data)
        
        
        
        # # display vertical desired position of the FOI
        # addViewerSphere( viz,'R0',0.035 , [0, 0, 1, 1])
        # applyViewerConfiguration(viz, 'R0',  np.hstack( (0*param.FOI_position[0],np.array([0,0,0,1])) ) ) 
        
        
        
        viz.display( simX[i, :model.nq])
        
        # Save screenshot
        #save_img=driver.save_screenshot(f"example/squat/video/frame_{i:04d}.png")
        #print(save_img)
        
        time.sleep(0.005)
        

