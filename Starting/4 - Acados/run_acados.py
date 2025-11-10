from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from my_tools_acados.my_acados_model import convert_my_double_joints_urdf
import numpy as np
import casadi as ca
from pinocchio import casadi as cpin
import pinocchio as pin


def main():
    
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    # robot, aca_model = export_double_pendulum_ode_model()
    robot, aca_model = convert_my_double_joints_urdf()

    
    
    visual_model = robot.visual_model
    collision_model = robot.collision_model
    data = robot.data
    q0=pin.neutral(robot.model)
    
    cmodel = cpin.Model(robot.model)
    cdata = cdata = cmodel.createData()
    
    ee_frame_name = "ee_link"
    ee_frame_id = robot.model.getFrameId(ee_frame_name)

     
 
    nq = robot.model.nq
    nv = robot.model.nv
    
    
    qd=q0.copy()
    qd=qd+0.001
    qd[0]=np.pi/4
    qd[1]=-np.pi/4
    # Forward kinematics
    pin.forwardKinematics(robot.model, data, qd)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
    pin.updateFramePlacements(robot.model, data)                   # Update the frames placement to symbolic expressions in data
         
    desired_pose = data.oMf[ee_frame_id].copy()  # Desired EE pose

    
    
    def tran_error(id, M,cx):
        # Update data symbolically inside the function
        cpin.forwardKinematics(cmodel, cdata, cx[:robot.model.nq])#, cx[model.nq:])#,cu)   
        cpin.updateFramePlacements(cmodel, cdata)                  
    
    
        tran_error =  cdata.oMf[id].translation - M.translation
        
        return tran_error 

    cost_to_target=tran_error(ee_frame_id, desired_pose,aca_model.x)#.T@tran_error(ee_frame_id, desired_pose,model.x)
    cost_to_target = ca.Function('cost_to_target', [aca_model.x], [cost_to_target])


    ocp.model = aca_model

    Tf = 1
    nx = aca_model.x.rows()
    nu = aca_model.u.rows()
    N = 200

    # set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # cost matrices
 

    # path cost
    # cost matrices
    # Q_mat = 2*np.diag([1e2, 1e2, 1e-2, 1e-2])
    # R_mat = 2*np.diag([1e-2,1e-2])
    # ocp.cost.cost_type = 'NONLINEAR_LS'
    # ocp.model.cost_y_expr = ca.vertcat(model.x, model.u,cost_to_target(model.x)[2])
    # ocp.cost.yref = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0,desired_pose.translation[2]])#np.zeros((nx+nu,))
    # ocp.cost.W = ca.diagcat(Q_mat, R_mat, 1e-3).full()

    #  # terminal cost
    # ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # ocp.model.cost_y_expr_e = model.x
    # ocp.cost.yref_e = np.zeros((nx,))
    # ocp.cost.W_e = Q_mat

    ################# @ludo uncomment pour voir l'erreur
    #cost matrices
    x_mat = 2*np.diag([1e-4, 1e-4, 1e-3, 1e-3])
    tau_mat = 2*np.diag([1e-3,1e-3])
    P_mat = 2*np.diag([1e1,1e1]) 
    
    
     # running cost 
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(aca_model.x, aca_model.u, cost_to_target(aca_model.x)[1:])
    ocp.cost.yref = np.concatenate([ np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), desired_pose.translation[1:] ])
    #np.zeros((nx+nu,))
    ocp.cost.W = ca.diagcat(x_mat,tau_mat, P_mat).full()


    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ca.vertcat(cost_to_target(aca_model.x)[1:])
    ocp.cost.yref_e = np.array([desired_pose.translation[1:]])#np.zeros((nx+nu,))
    ocp.cost.W_e = ca.diagcat(P_mat).full()
    ################# end  @ludo uncomment pour voir l'erreur

    # # set constraints
    tau_max = 1
    ocp.constraints.lbu = np.array([-tau_max, -tau_max])
    ocp.constraints.ubu = np.array([+tau_max, +tau_max])
    ocp.constraints.idxbu = np.array([0,1])


    #ocp.model.con_h_expr =  ca.vertcat(cost_to_target(model.x)[1:])

    epsilon = 0.001  # 1 cm tolerance
    # ocp.constraints.lh = -epsilon * np.ones(2)
    # ocp.constraints.uh = +epsilon * np.ones(2)

    ocp.model.nh_e = 2
    ocp.model.con_h_expr_e = ca.vertcat(cost_to_target(aca_model.x)[1:])
    ocp.constraints.lh_e = -epsilon * np.ones(2)
    ocp.constraints.uh_e = +epsilon * np.ones(2)

    #Initial condition
    ocp.constraints.x0 = np.array([0.0, 0.1, 0.0, 0.0])#np.concatenate([ qd,np.array([0.0, 0.0] )] )#np.array([0.0, 0.1, 0.0, 0.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    #ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    #ocp_solver = AcadosOcpSolver(ocp)

    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json", build=True, generate=True)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")



    plot_trajectories(
        x_traj_list=[simX],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, Tf, N+1)],
        time_label=aca_model.t_label,
        labels_list=['OCP result'],
        x_labels=aca_model.x_labels,
        u_labels=aca_model.u_labels,
        idxbu=ocp.constraints.idxbu,
        lbu=ocp.constraints.lbu,
        ubu=ocp.constraints.ubu,
        X_ref=None,
        U_ref=None,
        fig_filename='double_pendulum_ocp.png',
        x_min=None,
        x_max=None,
    )
    
    
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
        # # # Print joint positions in world frame
    # # print("Joint placements in world frame (R0):")
    # # for i in range(1, aca_model.njoints):  # skip joint 0 (universe)
    # #     placement = data.oMi[i]
    # #     joint_name = aca_model.names[i]
    # #     print(f"Joint ID: {i}, Name: {joint_name}, Translation: {placement.translation}")

    # # # Print frame positions in world frame
    # # print("\nFrame placements in world frame (R0):")
    # # for i, frame in enumerate(aca_model.frames):
    # #     placement = data.oMf[i]
    # #     print(f"Frame ID: {i}, Name: {frame.name}, Type: {frame.type}, Translation: {placement.translation}")
    
        
    
    # print("meshcat animation...")

    

    # viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    # #viz.initViewer(open=True)  # open=True opens browser automatically

    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.loadViewerModel()
    # #

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
        
    # #driver = webdriver.Safari()
    # #driver.get(viz.viewer.url())  # MeshCat URL
    # N=1
    # for i in range(N):

    # # print("samples "+str(i))
        
    #     pin.forwardKinematics(model, data, q0)
    #     pin.updateFramePlacements(model, data)
        
        
    #     viz.display(q0)
    
    #     time.sleep(0.005)
    


if __name__ == '__main__':
    main()