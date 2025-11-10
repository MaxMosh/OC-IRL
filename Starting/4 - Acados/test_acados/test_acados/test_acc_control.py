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
print(q0)
q0[1]=-np.pi/2
q0[2]=-np.pi/2
# q0[2]=np.pi/2
# q0[3]=np.pi/4

# tau0 = pin.rnea(model, data, q0, np.zeros(model.nv), np.zeros(model.nv))

qd=q0.copy()
qd=qd+0.001
qd[1]=0
 

# Forward kinematics
pin.forwardKinematics(model, data, qd)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
pin.updateFramePlacements(model, data)                   # Update the frames placement to symbolic expressions in data
         
desired_pose = data.oMf[ee_frame_id].copy()  # Desired EE pose


# ddq0=pin.aba(model, data, q0, np.zeros(model.nv), tau0)
# print(ddq0)

# -----------------------
# PIN+CASADI model definition
# -----------------------
# We will define a state x = (q, v)^T to describe the robot dynamics
nx = nq +  nv     # state dimension: positions and velocities
ndx = 2 * nv  # state derivative 
nu =  nv#        # control dimension: the accelerations


# * Create casadi symbolic variables
# These variables are used to define symbolic expression and are replaced in the solver by some values according to the decision variables
cx = casadi.SX.sym("x", nx, 1) # state: the positions and velocities
cdx = casadi.SX.sym("dx", ndx, 1)
cu = casadi.SX.sym("u", nu, 1) # control: the acc

 

 # because acados needs different name for state derivative
dq = casadi.SX.sym("dq", nv, 1)
ddq = casadi.SX.sym("ddq", nv, 1)

# Forward kinematics
cpin.forwardKinematics(cmodel, cdata, cx[:nq], cx[nq:])  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data



def pin_tran_error(id, M,cx):
    # Update data symbolically inside the function
    pin.forwardKinematics(model, data, cx[:model.nq])#, cx[model.nq:])#,cu)   
    pin.updateFramePlacements(model, data)                  
    
    tran_error =  data.oMf[id].translation - M.translation
 
    return tran_error 

def tran_error(id, M,cx):
    # Update data symbolically inside the function
    cpin.forwardKinematics(cmodel, cdata, cx[:model.nq])#, cx[model.nq:])#,cu)   
    cpin.updateFramePlacements(cmodel, cdata)                  
 
   
    tran_error =  cdata.oMf[id].translation - M.translation
    
    return tran_error 
        


cost_to_target=tran_error(ee_frame_id, desired_pose,cx).T@tran_error(ee_frame_id, desired_pose,cx)
cost_to_target = casadi.Function('cost_to_target', [cx], [cost_to_target])

 

# * Define the robot dynamics, this creates a relation between the symbolic variables defined above

 
dt=1/100
  # * Define a function to get the next state from the robot dynamics 
cnext = casadi.Function('cnext', [cx], [cpin.integrate(cmodel, cx[:nq], cx[nq:]*dt)])
qnext=cnext(cx)
dqnext=cx[nq:]+cu*dt 
dyn_fun = casadi.Function('dyn', [cx, cu], [casadi.vertcat(qnext,dqnext)])   



# -------------------------------------
# ACADOS Optimal Control Problem (OCP)
# -------------------------------------
ocp = AcadosOcp()
 

xdot=casadi.vertcat(dq,ddq)

f_expl= dyn_fun(cx, cu) 

f_impl= xdot - f_expl
ocp.model.f_impl_expr = f_impl
ocp.model.f_expl_expr = f_expl


ocp.model.name = "ur5_min_torque"
ocp.model.x = cx
ocp.model.u = cu
ocp.model.xdot = xdot

 

# # Time horizon
N=100
Tf=1
ocp.dT = dt
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf


# path cost
W=1e-1#*np.eye(6)
W1=1e-3#*np.eye(6)
acc_mat=1e-5#*np.eye(6)

ocp.cost.cost_type = "EXTERNAL"
ocp.cost.cost_type_e = "EXTERNAL"

cost_expr_ext = W1*(qd-cx[:nq]).T@(qd-cx[:nq]) + W1*(cx[nq:]).T@(cx[nq:]) +W*cost_to_target(cx) 

ocp.model.cost_expr_ext_cost =  cost_expr_ext+ acc_mat*casadi.sumsqr(cu) 
ocp.model.cost_expr_ext_cost_e =  cost_expr_ext
ocp.model.cost_expr_ext_cost_0 = cost_expr_ext  # for initial cost 
 

# ocp.cost.cost_type = 'NONLINEAR_LS'
# ocp.model.cost_y_expr =  casadi.vertcat(ocp.model.x,ocp.model.u) 
# ocp.cost.yref = np.concatenate( [qd,np.zeros(nv), np.zeros(nv)] ) 
# ocp.cost.W = casadi.diagcat(W,W1, acc_mat).full()

# # # terminal cost


# ocp.cost.cost_type_e = 'NONLINEAR_LS'
# ocp.model.cost_y_expr_e = casadi.vertcat(ocp.model.x) 
# ocp.cost.yref_e = np.concatenate( [qd,np.zeros(nv)] ) ##np.array([0,-np.pi/2,0,0,0,0])
# ocp.cost.W_e = casadi.diagcat(W,W1).full()




# # Set bounds on this constraint
# ocp.constraints.constr_type = 'BGH'
# ocp.constraints.constr_expr = cost_to_target(cx)

# ocp.constraints.constr_lower_bound = -1e-3*np.ones(3)
# ocp.constraints.constr_upper_bound = 1e-3*np.ones(3)



# Initial guess      
ocp.constraints.x0 = np.concatenate( [qd  ,np.zeros(nv)] )



# Constraints states
# Define state bounds
 

# Indices of states to constrain (e.g., all)
# ocp.constraints.idxbx = np.arange(ocp.model.x.size()[0])

## Apply bounds at all stages
## Apply constraints on q only (first nq elements of x)
# q_min = model.lowerPositionLimit.copy()
# q_max = model.upperPositionLimit.copy()

# ocp.constraints.idxbx = np.arange(nq)  # indices of q in x

# # Lower and upper bounds
# ocp.constraints.lbx = q_min  # shape (nq,)
# ocp.constraints.ubx = q_max  # shape (nq,)

# # Constraint control
# u_min = np.array([-100.0]*ocp.model.u.size()[0])
# u_max = np.array([ 100.0]*ocp.model.u.size()[0])
# ocp.constraints.idxbu = np.arange(ocp.model.u.size()[0])
# ocp.constraints.lbu = u_min
# ocp.constraints.ubu = u_max

ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
#ocp.cost.ext_cost_num_hess = True
#ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
#ocp.solver_options.integrator_type = "DISCRETE"
#ocp.solver_options.integrator_type = 'ERK'
# robust solver settings:
#ocp.solver_options.nlp_solver_tol_stat = 1e-6
#ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
#ocp.solver_options.levenberg_marquardt = 1e-2  # Helps with numerical stability
#ocp.solver_options.nlp_solver_type = 'SQP'  # try 'SQP_RTI' only after stable
#ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

ocp.solver_options.nlp_solver_max_iter = 100  # or any integer you prefer


ocp.solver_options.print_level = 2
ocp.solver_options.compile_interface = True

# Create solver

ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# for i in range(N):
#     ocp_solver.set(i, "u", tau0) 

# Solve
status = ocp_solver.solve()

total_cost = ocp_solver.get_cost()   # Get the total cost at current solution
print(f" total cost: {total_cost}")

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


#test_fun = casadi.Function("test_cost", [cx], [cost_expr_ext])
#print("cost_expr_ext")
#print( test_fun( np.concatenate([simX[-2,:]]) ) )


# total_cost=0
# for i in range(N):
    
#     total_cost+=test_fun( np.concatenate([simX[i,:]]) )

# print("total_cost")
# print(total_cost)

 
# print("q0:")
# print(q0)

# print("q_opt(0):")
# print(simX[0,:])

# print("q_opt(tf):")
# print(simX[-1,:])


# print("q0:")
# print(q0)

# print("q_opt(0):")
# print(simX[0,:])

print("q0:")
print(q0)
print("qd:")
print(qd)
print("q_opt(tf):")
print(simX[-1,:nq])
# print("qmin:")
# print(q_min)
# print("qmax:")
# print(q_max)

cost=cost_to_target(np.array(simX[1,:]))

print("trans error")
print(cost)

pin.forwardKinematics(model, data, simX[-1, :nq])  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
pin.updateFramePlacements(model, data)                   # Update the frames placement to symbolic expressions in data
        
 



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



    
print("meshcat animation...")

 

viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
#viz.initViewer(open=True)  # open=True opens browser automatically

viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.loadViewerModel()
#

native_viz = viz.viewer
native_viz["/Background"].set_property("top_color", [1, 1, 1])  # Dark gray (RGB values in [0, 1])
native_viz["/Background"].set_property("bottom_color", [0.65, 0.65, 0.65])  # Same color → flat background
grid_height = -1.0  # Negative z = lower the grid
native_viz["/Grid"].set_transform(
np.array([
    [1, 0, 0, 0],  # Rotation (identity)
    [0, 1, 0, 0],
    [0, 0, 1, grid_height],
    [0, 0, 0, 1]
]))
    
#driver = webdriver.Safari()
#driver.get(viz.viewer.url())  # MeshCat URL
 
for i in range(N):

   # print("samples "+str(i))
     
    pin.forwardKinematics(model, data, simX[i, :model.nq])
    pin.updateFramePlacements(model, data)
    
    
    
    # # display vertical desired position of the FOI
    # addViewerSphere( viz,'R0',0.035 , [0, 0, 1, 1])
    # applyViewerConfiguration(viz, 'R0',  np.hstack( (0*param.FOI_position[0],np.array([0,0,0,1])) ) ) 
    
    
    
    viz.display(simX[i, :model.nq])
    
    # Save screenshot
    #save_img=driver.save_screenshot(f"example/squat/video/frame_{i:04d}.png")
    #print(save_img)
    
    time.sleep(0.005)
    

