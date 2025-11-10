# double_pendulum_minimal_example_ocp.py -- adapted to use the URDF-based model
from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
import numpy as np
import casadi as ca
from pinocchio import casadi as cpin
import pinocchio as pin

# import our adapted model exporter
from test_cg_two_arms import export_double_pendulum_ode_model

def main():

    ocp = AcadosOcp()

    # --- get robot and acados model built from URDF
    # You can change the URDF path / viscous coefficients by passing arguments to export_double_pendulum_ode_model
    robot, model = export_double_pendulum_ode_model(
        urdf_path="/home/n7student/Documents/Boulot/CNRS@CREATE/Codes/OC & IRL/Starting/3 - URDF et pinocchio/assets/mon_robot.urdf",
        viscous_coeffs=(0.1, 0.1),   # <-- tune viscous friction here
        gravity_vec=(0.0, 0.0, -9.81) # <-- tune gravity here
    )

    # If visual/collision models are not available, set to None (placeholder)
    visual_model = getattr(robot, "visual_model", None)
    collision_model = getattr(robot, "collision_model", None)
    data = robot.data

    # get a reasonable nominal config: neutral configuration (zeros) if available
    try:
        q0 = pin.neutral(robot.model)
    except Exception:
        q0 = np.zeros(robot.model.nq)

    # Build a casadi model for frame computations (for cost)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    # pick an end-effector frame ID:
    # - try to find a named frame 'link3' (as in the original example)
    # - otherwise use last frame available
    ee_frame_name = "ee_link"
    try:
        ee_frame_id = robot.model.getFrameId(ee_frame_name)
    except Exception:
        # fallback: last defined frame
        if len(robot.model.frames) > 0:
            ee_frame_id = robot.model.frames[-1].id
            ee_frame_name = robot.model.frames[-1].name
        else:
            raise RuntimeError("Aucun frame trouvé dans le modèle Pinocchio. Vérifie l'URDF.")

    # build a desired pose by slightly moving q0 (just like original script)
    qd = q0.copy()
    if robot.model.nq >= 2:
        qd = qd + 0.001
        qd[0] = np.pi / 4
        qd[1] = -np.pi / 4

    # compute desired frame pose
    pin.forwardKinematics(robot.model, data, qd)
    pin.updateFramePlacements(robot.model, data)
    desired_pose = data.oMf[ee_frame_id].copy()

    # helper: translation error CASADI function (uses the casadi pinocchio wrapper)
    def tran_error(id, M, cx):
        cpin.forwardKinematics(cmodel, cdata, cx[:robot.model.nq])
        cpin.updateFramePlacements(cmodel, cdata)
        return cdata.oMf[id].translation - M.translation

    cost_to_target = tran_error(ee_frame_id, desired_pose, model.x)
    cost_to_target = ca.Function('cost_to_target', [model.x], [cost_to_target])

    ocp.model = model

    # horizon and sizes
    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 200

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # --- cost: state, input, and end-effector position tracking (translation)
    x_mat = 2 * np.diag([1e-4]*nx)
    if nx >= 4:
        # small tweak to mimic original weighting split (q,v)
        x_mat = 2 * np.diag([1e-4, 1e-4, 1e-3, 1e-3] + [1e-4]*(nx-4))
    tau_mat = 2 * np.diag([1e-3] * nu)
    # weight on end-effector y/z (2 components)
    P_mat = 2 * np.diag([1e1, 1e1])

    ocp.cost.cost_type = 'NONLINEAR_LS'
    # model.x, model.u, and the two last components of translation error
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u, cost_to_target(model.x)[1:])
    ocp.cost.yref = np.concatenate([np.zeros(nx + nu), desired_pose.translation[1:]])
    ocp.cost.W = ca.diagcat(x_mat, tau_mat, P_mat).full()

    # terminal cost on translation error (y,z)
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ca.vertcat(cost_to_target(model.x)[1:])
    ocp.cost.yref_e = np.array(desired_pose.translation[1:])
    ocp.cost.W_e = ca.diagcat(P_mat).full()

    # input bounds (tau)
    tau_max = 1.0
    ocp.constraints.lbu = -tau_max * np.ones(nu)
    ocp.constraints.ubu = +tau_max * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    # equality constraint on final EE position (small epsilon window)
    epsilon = 0.001
    ocp.model.nh_e = 2
    ocp.model.con_h_expr_e = ca.vertcat(cost_to_target(model.x)[1:])
    ocp.constraints.lh_e = -epsilon * np.ones(2)
    ocp.constraints.uh_e = +epsilon * np.ones(2)

    # initial condition: q0 + zeros velocities
    x0 = np.zeros(nx)
    x0[:robot.model.nq] = q0
    ocp.constraints.x0 = x0

    # solver options (same as original example)
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    # create solver (do not rebuild here; follow original pattern)
    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json", build=False, generate=False)

    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics()

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    for i in range(N):
        simX[i, :] = ocp_solver.get(i, "x")
        simU[i, :] = ocp_solver.get(i, "u")
    simX[N, :] = ocp_solver.get(N, "x")

    plot_trajectories(
        x_traj_list=[simX],
        u_traj_list=[simU],
        time_traj_list=[np.linspace(0, Tf, N + 1)],
        time_label=model.t_label,
        labels_list=['OCP result'],
        x_labels=model.x_labels,
        u_labels=model.u_labels,
        idxbu=ocp.constraints.idxbu,
        lbu=ocp.constraints.lbu,
        ubu=ocp.constraints.ubu,
        X_ref=None,
        U_ref=None,
        fig_filename='double_pendulum_ocp.png',
    )

    # forward kinematics on final q
    final_q = simX[-1, :robot.model.nq]
    pin.forwardKinematics(robot.model, data, final_q)
    pin.updateFramePlacements(robot.model, data)
    actual_pose = data.oMf[ee_frame_id].copy()

    print("desired ee pose (translation):", desired_pose.translation)
    print("actual ee pose  (translation):", actual_pose.translation)
    print("position error:", desired_pose.translation - actual_pose.translation)


if __name__ == '__main__':
    main()
