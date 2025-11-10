# double_pendulum_model.py -- adapted to load a URDF and include viscous friction
from acados_template import AcadosModel
from casadi import SX, vertcat
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import types

def export_double_pendulum_ode_model(urdf_path="/mnt/data/mon_robot.urdf",
                                     viscous_coeffs=(0.1, 0.1),
                                     gravity_vec=(0.0, 0.0, -9.81),
                                     model_name='double_pendulum_from_urdf'):
    """
    Build an AcadosModel from a URDF using Pinocchio + CasADi ABA.
    - urdf_path: path to the user's URDF (default: /mnt/data/mon_robot.urdf)
    - viscous_coeffs: tuple/list with one viscous damping coefficient per DOF (nv)
    - gravity_vec: gravity vector to set in the Pinocchio model
    Returns: (robot_namespace, aca_model)
    """

    # -----------------------
    # 1) Load Pinocchio model from URDF
    # -----------------------
    try:
        model = pin.buildModelFromUrdf(urdf_path)
    except Exception as e:
        # Some pinocchio installs require a different helper; surface a helpful error
        raise RuntimeError(f"Impossible de charger le URDF '{urdf_path}': {e}")

    # set gravity explicitly (Pinocchio uses model.gravity)
    try:
        model.gravity = np.array(gravity_vec)
    except Exception:
        # If attribute not present, try pin.setGravity (older/newer APIs may differ)
        try:
            pin.setGravity(model, np.array(gravity_vec))
        except Exception:
            # fallback: ignore — Pinocchio usually has -9.81 by default
            pass

    # create data
    data = pin.Data(model)

    # Build a small "robot" namespace similar to example_robot_data output that the example script expects
    robot = types.SimpleNamespace()
    robot.model = model
    robot.data = data
    # keep placeholders for compatibility with existing code that may access visual/collision models
    robot.visual_model = None
    robot.collision_model = None

    # -----------------------
    # 2) CASADI wrapper for ABA (including viscous friction)
    # -----------------------
    nq = model.nq
    nv = model.nv

    # state symbols: q (nq) and v (nv)
    q_syms = [casadi.SX.sym(f"q{i}") for i in range(nq)]
    v_syms = [casadi.SX.sym(f"v{i}") for i in range(nv)]
    q = vertcat(*q_syms) if nq > 0 else casadi.SX()
    v = vertcat(*v_syms) if nv > 0 else casadi.SX()
    cx = vertcat(q, v)

    # controls (joint torques)
    tau = casadi.SX.sym('tau', nv)
    cu = vertcat(tau)

    # viscous damping vector (B * v)
    # make sure viscous_coeffs has length nv; if not, broadcast or truncate
    if len(viscous_coeffs) < nv:
        viscous_coeffs = list(viscous_coeffs) + [viscous_coeffs[-1]] * (nv - len(viscous_coeffs))
    B_vec = casadi.SX.zeros(nv)
    for i in range(nv):
        B_vec[i] = viscous_coeffs[i]

    # acceleration function: ABA expects (model, data, q, v, tau)
    # we provide tau_eff = tau - B*v
    # create a casadi Function "acc" that returns ddq
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # Note: cpin.aba accepts CasADi SX arguments and returns CasADi SX vector
    acc = casadi.Function("acc", [cx, cu],
                          [cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], cu - casadi.diag(B_vec) @ cx[nq:])])

    # explicit dynamics f_expl = [v; ddq]
    f_expl = casadi.vertcat(cx[nq:], acc(cx, cu))

    # formal xdot symbols for implicit form
    qdot_syms = [casadi.SX.sym(f"qdot{i}") for i in range(nq)]
    vdot_syms = [casadi.SX.sym(f"vdot{i}") for i in range(nv)]
    xdot = vertcat(*(qdot_syms + vdot_syms))

    f_impl = xdot - f_expl

    # -----------------------
    # 3) Fill AcadosModel
    # -----------------------
    aca_model = AcadosModel()
    aca_model.f_impl_expr = f_impl
    aca_model.f_expl_expr = f_expl
    aca_model.x = cx
    aca_model.xdot = xdot
    aca_model.u = cu
    aca_model.name = model_name

    # labels (generic)
    x_labels = []
    for i in range(nq):
        x_labels.append(f"q{i+1} [rad]")
    for i in range(nv):
        x_labels.append(f"v{i+1} [rad/s]")
    aca_model.x_labels = x_labels
    aca_model.u_labels = [f"tau{i+1}" for i in range(nv)]
    aca_model.t_label = '$t$ [s]'

    # Return a robot-like object and AcadosModel
    return robot, aca_model
