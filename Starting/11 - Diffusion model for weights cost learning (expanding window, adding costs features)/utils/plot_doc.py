# utils/plot_doc.py
import numpy as np
import casadi
import matplotlib.pyplot as plt
import pinocchio as pin


__all__ = [
    "com_xy_traj",
    "bos_geometry_from_state",
    "bos_violations",
    "plot_com_vs_bos",
]


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_joint_trajectories(doc, xs, param, q_hum=None, show_limits=True, cols=3, figsize=None):
    """
    Plot joint position trajectories (in degrees).
    - DOC estimate: blue line
    - Human IK (optional): orange line
    - Joint limits (optional): black dashed lines

    Also computes RMSE (deg) between DOC and human trajectories per joint
    and displays the average RMSE in the figure title.
    Returns a list of RMSEs where rmse_list[0] is the average RMSE, and the
    remaining entries are per-joint RMSEs (in subplot order).

    Parameters
    ----------
    doc : object
        Your DocHumanMotionGeneration_InvDyn instance (must have `model`).
    xs : np.ndarray, shape (T, nx)
        State trajectory; first `model.nq` columns are joint positions [rad].
    param : object
        Must provide:
          - param.active_joints : list[str] of joint names to display (order for labels)
          - param.free_flyer    : bool
    q_hum : np.ndarray or None, shape (T_h, model.nq), optional
        Human joint positions [rad] to overlay; will be clipped to min(T, T_h).
    show_limits : bool
    cols : int
    figsize : tuple or None

    Returns
    -------
    fig, axes, rmse_list : (Figure, Axes array, list[float])
        rmse_list[0] = average RMSE across plotted joints (deg)
        rmse_list[1:] = per-joint RMSEs in subplot order (deg)
    """
    model = doc.model
    T = xs.shape[0]
    nq = model.nq
    q_est_rad = xs[:, :nq]  # (T, nq)

    # radians -> degrees
    def rad2deg(arr):
        return np.rad2deg(np.asarray(arr, dtype=float))

    # which part of q is actuated (exclude free-flyer if any)
    ff_q = 7 if getattr(param, "free_flyer", False) else 0

    # joint limits (rad)
    q_lower = np.asarray(model.lowerPositionLimit)
    q_upper = np.asarray(model.upperPositionLimit)

    # Will build sequences in plotting order
    series = []  # tuples of (label, est_deg, hum_deg_or_None, (lo_deg, hi_deg))

    # Iterate according to param.active_joints
    for jname in getattr(param, "active_joints", []):
        if jname in ("universe", "root_joint"):
            continue

        jid = model.getJointId(jname)
        if jid <= 0 or jid >= model.njoints:
            continue

        j = model.joints[jid]
        i0, nj = j.idx_q, j.nq

        # skip anything fully before actuated block
        if i0 + nj <= ff_q:
            continue

        start = max(i0, ff_q)
        end = i0 + nj

        for k in range(start, end):
            label = jname if (end - start == 1) else f"{jname}[{k - i0}]"
            est_deg = rad2deg(q_est_rad[:, k])

            hum_deg = None
            if q_hum is not None and q_hum.shape[1] >= nq:
                # Use common time horizon
                Teff = min(T, q_hum.shape[0])
                hum_deg = rad2deg(q_hum[:Teff, k])
                # also clip est to Teff for plotting/RMSE fairness
                est_deg = est_deg[:Teff]

            lo_deg = rad2deg(q_lower[k])
            hi_deg = rad2deg(q_upper[k])

            series.append((label, est_deg, hum_deg, (lo_deg, hi_deg)))

    if not series:
        raise ValueError("No actuated joints found to plot. Check param.active_joints and free_flyer setting.")

    n = len(series)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (cols * 4.0, rows * 2.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # Compute per-joint RMSEs (deg) when human is provided
    per_joint_rmse = []
    t0 = 0  # time starts at 0
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        lbl, est_deg, hum_deg, (lo_deg, hi_deg) = series[i]
        t = np.arange(t0, t0 + len(est_deg))

        # DOC estimate (blue)
        ax.plot(t, est_deg, '-', label="DOC", linewidth=1.8)

        # Human (orange) + RMSE
        if hum_deg is not None:
            # Clip to common length (already done above)
            m = min(len(est_deg), len(hum_deg))
            rmse_i = float(np.sqrt(np.mean((est_deg[:m] - hum_deg[:m])**2)))
            per_joint_rmse.append(rmse_i)

            ax.plot(t[:m], hum_deg[:m], '-', label="Human", linewidth=1.4)
        else:
            per_joint_rmse.append(np.nan)

        # Limits (black dashed)
        if show_limits:
            ax.axhline(lo_deg, color="k", linestyle="--", linewidth=0.9, label="limit" if i == 0 else None)
            ax.axhline(hi_deg, color="k", linestyle="--", linewidth=0.9)

        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    # Axis labels & legend
    axes[min(n - 1, len(axes) - 1)].set_xlabel("time step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")

    # Average RMSE across joints (ignore NaNs if no human provided for some)
    if np.all(np.isnan(per_joint_rmse)):
        avg_rmse = np.nan
    else:
        avg_rmse = float(np.nanmean(per_joint_rmse))

    # Title w/ average RMSE
    title = "Joint positions (deg)"
    if not np.isnan(avg_rmse):
        title += f" — avg RMSE: {avg_rmse:.2f}°"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Build RMSE list with avg first
    rmse_list = [avg_rmse] + per_joint_rmse
    return fig, axes, rmse_list


import numpy as np
import matplotlib.pyplot as plt

def plot_joint_velocities(doc, xs, param, dq_hum=None, show_limits=False, cols=3, figsize=None):
    """
    Plot joint velocity trajectories (deg/s).
    - DOC estimate: blue
    - Human (optional): orange
    - Limits (optional): black dashed, if param.dq_min/dq_max exist

    Parameters
    ----------
    doc : object
        Instance with `model` (Pinocchio).
    xs : np.ndarray, shape (T, nx)
        State trajectory; xs[:, :nq] are q, xs[:, nq:] are dq (rad or rad/s).
    param : object
        Needs:
          - active_joints : list[str]
          - free_flyer    : bool
        Optional (for limits):
          - dq_min, dq_max : arrays length nv (rad/s)
    dq_hum : np.ndarray or None, shape (T, nv), optional
        Human joint velocities (rad/s) to overlay.
    show_limits : bool
        Draw dq_min/dq_max if present in `param`.
    cols : int
        Subplot columns.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    fig, axes
    """
    model = doc.model
    T = xs.shape[0]
    nq, nv = model.nq, model.nv

    q_rad  = xs[:, :nq]
    dq_rad = xs[:, nq:nq+nv]  # (T, nv)

    def rad2deg(arr): return np.rad2deg(np.asarray(arr, dtype=float))

    # Free-flyer offsets
    ff_q = 7 if getattr(param, "free_flyer", False) else 0
    ff_v = 6 if getattr(param, "free_flyer", False) else 0

    # Optional limits (rad/s) → deg/s
    have_v_limits = hasattr(param, "dq_min") and hasattr(param, "dq_max")
    if show_limits and have_v_limits:
        dq_min_deg = rad2deg(param.dq_min)
        dq_max_deg = rad2deg(param.dq_max)

    series = []  # (label, est_deg, hum_deg, (lo_deg, hi_deg) or None)

    for jname in getattr(param, "active_joints", []):
        if jname in ("universe", "root_joint"):
            continue

        jid = model.getJointId(jname)
        if jid <= 0 or jid >= model.njoints:
            continue

        j = model.joints[jid]
        i0_v, nj = j.idx_v, j.nv

        # Skip free-flyer region
        if i0_v + nj <= ff_v:
            continue

        start = max(i0_v, ff_v)
        end   = i0_v + nj

        for k in range(start, end):
            lbl = jname if (end - start == 1) else f"{jname}[{k - i0_v}]"

            est_deg = rad2deg(dq_rad[:, k])

            hum_deg = None
            if dq_hum is not None and dq_hum.shape[1] >= nv:
                hum_deg = rad2deg(dq_hum[:T, k])

            lims = None
            if show_limits and have_v_limits:
                lims = (dq_min_deg[k], dq_max_deg[k])

            series.append((lbl, est_deg, hum_deg, lims))

    if not series:
        raise ValueError("No actuated velocity DoFs to plot. Check active_joints/free_flyer.")

    n = len(series)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (cols * 4.0, rows * 2.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    t = np.arange(T)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        lbl, est_deg, hum_deg, lims = series[i]

        ax.plot(t, est_deg, '-', linewidth=1.8, label="DOC")
        if hum_deg is not None:
            ax.plot(t, hum_deg, '-', linewidth=1.4, label="Human")

        if show_limits and lims is not None:
            lo, hi = lims
            ax.axhline(lo, color="k", linestyle="--", linewidth=0.9, label="limit" if i == 0 else None)
            ax.axhline(hi, color="k", linestyle="--", linewidth=0.9)

        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    axes[min(n - 1, len(axes) - 1)].set_xlabel("time step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
    #fig.tight_layout()
    fig.suptitle("Joint velocities (deg/s)", y=0.995)             # <— add this
    fig.tight_layout(rect=[0, 0, 1, 0.97])   # <— leave space for title
    return fig, axes
     


def plot_joint_accelerations(doc, us, param, ddq_hum=None, show_limits=False, cols=3, figsize=None):
    """
    Plot joint accelerations (controls) in deg/s^2.
    - DOC estimate: blue
    - Human (optional): orange
    - Limits (optional): black dashed, if param.ddq_min/ddq_max exist

    Parameters
    ----------
    doc : object
        Instance with `model` (Pinocchio).
    us : np.ndarray, shape (T, m)
        Control trajectory; first nv entries are joint accelerations (rad/s^2).
        (If you append external wrenches after nv, they are ignored here.)
    param : object
        Needs:
          - active_joints : list[str]
          - free_flyer    : bool
        Optional (for limits):
          - ddq_min, ddq_max : arrays length nv (rad/s^2)
    ddq_hum : np.ndarray or None, shape (T, nv), optional
        Human accelerations (rad/s^2) to overlay.
    show_limits : bool
        Draw ddq_min/ddq_max if present in `param`.
    cols : int
        Subplot columns.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    fig, axes
    """
    model = doc.model
    T = us.shape[0]
    nv = model.nv

    ddq_rad = us[:, :nv]  # (T, nv) — assumes your controls start with ddq

    def rad2deg(arr): return np.rad2deg(np.asarray(arr, dtype=float))

    ff_v = 6 if getattr(param, "free_flyer", False) else 0

    # Optional limits (rad/s^2) → deg/s^2
    have_a_limits = hasattr(param, "ddq_min") and hasattr(param, "ddq_max")
    if show_limits and have_a_limits:
        ddq_min_deg = rad2deg(param.ddq_min)
        ddq_max_deg = rad2deg(param.ddq_max)

    series = []  # (label, est_deg, hum_deg, (lo_deg, hi_deg) or None)

    for jname in getattr(param, "active_joints", []):
        if jname in ("universe", "root_joint"):
            continue

        jid = model.getJointId(jname)
        if jid <= 0 or jid >= model.njoints:
            continue

        j = model.joints[jid]
        i0_v, nj = j.idx_v, j.nv

        if i0_v + nj <= ff_v:
            continue

        start = max(i0_v, ff_v)
        end   = i0_v + nj

        for k in range(start, end):
            lbl = jname if (end - start == 1) else f"{jname}[{k - i0_v}]"

            est_deg = rad2deg(ddq_rad[:, k])

            hum_deg = None
            if ddq_hum is not None and ddq_hum.shape[1] >= nv:
                hum_deg = rad2deg(ddq_hum[:T, k])

            lims = None
            if show_limits and have_a_limits:
                lims = (ddq_min_deg[k], ddq_max_deg[k])

            series.append((lbl, est_deg, hum_deg, lims))

    if not series:
        raise ValueError("No actuated acceleration DoFs to plot. Check active_joints/free_flyer.")

    n = len(series)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (cols * 4.0, rows * 2.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    t = np.arange(T)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        lbl, est_deg, hum_deg, lims = series[i]

        ax.plot(t, est_deg, '-', linewidth=1.8, label="DOC")
        if hum_deg is not None:
            ax.plot(t, hum_deg, '-', linewidth=1.4, label="Human")

        if show_limits and lims is not None:
            lo, hi = lims
            ax.axhline(lo, color="k", linestyle="--", linewidth=0.9, label="limit" if i == 0 else None)
            ax.axhline(hi, color="k", linestyle="--", linewidth=0.9)

        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    axes[min(n - 1, len(axes) - 1)].set_xlabel("time step")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
     #fig.tight_layout()
    fig.suptitle("Joint accelerations (deg/s²)", y=0.995)             # <— add this
    fig.tight_layout(rect=[0, 0, 1, 0.97])   # <— leave space for title
    return fig, axes
     

import numpy as np
import matplotlib.pyplot as plt
import casadi

def plot_joint_torques(doc, xs, us, param, tau_hum=None, show_limits=False, cols=3, figsize=None,
                       title="Joint torques (Nm)"):
    """
    Plot joint torques computed via doc.tau.
    - Handles both cases:
        * free_flyer=False  -> tau(x, ddq)
        * free_flyer=True   -> tau(x, ddq, foot_wrenches)
    - DOC estimate: blue
    - Human (optional): orange
    - Limits (optional): black dashed, if param.tau_min/tau_max exist

    Parameters
    ----------
    doc : object
        Your DocHumanMotionGeneration_InvDyn instance (must expose `model` and `tau`).
    xs : np.ndarray, shape (T, nx)
        State trajectory; first nq columns are positions, next nv columns are velocities.
    us : np.ndarray, shape (T, m)
        Control trajectory; first nv columns are joint accelerations; if free_flyer, next 12 are wrenches.
    param : object
        Needs:
          - active_joints : list[str]
          - free_flyer    : bool
        Optional (for limits):
          - tau_min, tau_max : arrays length nv (Nm)
    tau_hum : np.ndarray or None, shape (T, nv), optional
        External torques to overlay (e.g., from an inverse dynamics pass).
    show_limits : bool
        Draw tau_min/tau_max if present in `param`.
    cols : int
        Subplot columns.
    figsize : tuple or None
        Figure size.
    title : str
        Figure title.

    Returns
    -------
    fig, axes
    """
    model = doc.model
    T = xs.shape[0]
    nq, nv = model.nq, model.nv

    # Split inputs
    X = casadi.DM(np.asarray(xs[:, :nq+nv]).T)   # (nx x T)
    Uddq = casadi.DM(np.asarray(us[:, :nv]).T)  # (nv x T)

    # Vectorized torque evaluation using CasADi .map for speed
    if getattr(param, "free_flyer", False):
        # Expect 12 extra controls for two 6D foot wrenches after nv
        F = casadi.DM(np.asarray(us[:, nv:nv+12]).T)  # (12 x T)
        tau_map = doc.tau.map(T)(X, Uddq, F)          # (nv x T) DM
    else:
        tau_map = doc.tau.map(T)(X, Uddq)             # (nv x T) DM

    tau_est = np.array(tau_map.full()).T              # (T x nv) numpy

    # Free-flyer torque indices to exclude from plotting
    ff_v = 6 if getattr(param, "free_flyer", False) else 0

    # Optional limits
    have_limits = hasattr(param, "tau_min") and hasattr(param, "tau_max")
    if show_limits and have_limits:
        tau_min = np.asarray(param.tau_min, dtype=float)
        tau_max = np.asarray(param.tau_max, dtype=float)

    # Build per-DoF series in the order of param.active_joints
    series = []   # entries: (label, est, hum, (lo, hi) or None)
    for jname in getattr(param, "active_joints", []):
        if jname in ("universe", "root_joint"):
            continue
        jid = model.getJointId(jname)
        if jid <= 0 or jid >= model.njoints:
            continue
        j = model.joints[jid]
        i0_v, nj = j.idx_v, j.nv  # torque/velocity space indexing

        # Skip fully in free-flyer region
        if i0_v + nj <= ff_v:
            continue

        start = max(i0_v, ff_v)
        end   = i0_v + nj

        for k in range(start, end):
            lbl = jname if (end - start == 1) else f"{jname}[{k - i0_v}]"
            est = tau_est[:, k]
            hum = None
            if tau_hum is not None and tau_hum.shape[1] >= nv:
                hum = np.asarray(tau_hum[:T, k], dtype=float)

            lims = None
            if show_limits and have_limits:
                lims = (float(tau_min[k]), float(tau_max[k]))

            series.append((lbl, est, hum, lims))

    if not series:
        raise ValueError("No actuated torque DoFs to plot. Check active_joints/free_flyer and inputs.")

    n = len(series)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (cols * 4.0, rows * 2.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(T)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        lbl, est, hum, lims = series[i]
        ax.plot(t, est, '-', linewidth=1.8, label="DOC")
        if hum is not None:
            ax.plot(t, hum, '-', linewidth=1.4, label="Human")

        if show_limits and lims is not None:
            lo, hi = lims
            ax.axhline(lo, color="k", linestyle="--", linewidth=0.9, label="limit" if i == 0 else None)
            ax.axhline(hi, color="k", linestyle="--", linewidth=0.9)

        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    axes[min(n - 1, len(axes) - 1)].set_xlabel("time step")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


 

def plot_external_wrenches(doc,
                           xs,
                           fs_sol,
                           frame: str = "foot",          # "foot" or "world"
                           human_fs_world: np.ndarray | None = None,  # optional (T x 12) in WORLD
                           left_foot_name: str = "left_foot",
                           right_foot_name: str = "right_foot",
                           title: str = "External Wrenches"):
    """
    Plot external wrenches (forces & moments) under each foot using fs_sol.

    Args
    ----
    doc : your DocHumanMotionGeneration_InvDyn instance (needs .model and .data)
    xs  : (T or T+1, nq+nv) state trajectory; only q is used
    fs_sol : (T, 12) -> [LF Fx,Fy,Fz,Mx,My,Mz,  RF Fx,Fy,Fz,Mx,My,Mz] in *foot* frames
    frame : "foot" (plot as-is) or "world" (transform foot->world per-sample)
    human_fs_world : optional (T, 12) wrenches in WORLD frame to overlay (orange)
    left_foot_name, right_foot_name : frame names
    title : figure title

    Returns
    -------
    fig, axes, data : (matplotlib objects, dict with arrays used for plotting)
    """
    model, data = doc.model, doc.data
    nq, nv = model.nq, model.nv

    if fs_sol is None or fs_sol.ndim != 2 or fs_sol.shape[1] < 12:
        raise ValueError(f"fs_sol must be (T, 12+). Got {None if fs_sol is None else fs_sol.shape}.")

    T = fs_sol.shape[0]
    q_traj = np.asarray(xs[:T, :nq])  # align with T

    # Split left/right foot local wrenches from fs_sol (assumed in FOOT frames)
    LF_loc = np.asarray(fs_sol[:, 0:6])
    RF_loc = np.asarray(fs_sol[:, 6:12])

    # Optional human overlay, provided in WORLD; will be mapped if needed
    if human_fs_world is not None:
        human_fs_world = np.asarray(human_fs_world)
        if human_fs_world.shape[0] != T or human_fs_world.shape[1] < 12:
            raise ValueError("human_fs_world must be (T,12+) in WORLD frame.")
        HLF_world = human_fs_world[:, 0:6]
        HRF_world = human_fs_world[:, 6:12]
    else:
        HLF_world = HRF_world = None

    # Foot frame IDs
    id_lf = model.getFrameId(left_foot_name)
    id_rf = model.getFrameId(right_foot_name)

    # Transform if required
    if frame.lower() == "world":
        LF = np.zeros_like(LF_loc)
        RF = np.zeros_like(RF_loc)
        HLF = np.zeros_like(LF_loc) if HLF_world is not None else None
        HRF = np.zeros_like(RF_loc) if HRF_world is not None else None

        for k in range(T):
            qk = q_traj[k]
            pin.forwardKinematics(model, data, qk)
            pin.updateFramePlacements(model, data)

            oMf_l = data.oMf[id_lf]
            oMf_r = data.oMf[id_rf]

            # fs_sol is local-to-foot -> map to WORLD
            F_l = pin.Force(LF_loc[k, 0:3], LF_loc[k, 3:6])
            F_r = pin.Force(RF_loc[k, 0:3], RF_loc[k, 3:6])
            F_l_w = oMf_l.act(F_l)
            F_r_w = oMf_r.act(F_r)

            LF[k, 0:3] = F_l_w.linear
            LF[k, 3:6] = F_l_w.angular
            RF[k, 0:3] = F_r_w.linear
            RF[k, 3:6] = F_r_w.angular

            # Human overlay is world; if we’re plotting in world, take as-is
            if HLF is not None:
                HLF[k, :] = HLF_world[k, :6]
                HRF[k, :] = HRF_world[k, :6]
    elif frame.lower() == "foot":
        # Plot as-is; map human WORLD -> FOOT if overlay requested
        LF, RF = LF_loc, RF_loc
        if human_fs_world is not None:
            HLF = np.zeros_like(LF_loc)
            HRF = np.zeros_like(RF_loc)
            for k in range(T):
                qk = q_traj[k]
                pin.forwardKinematics(model, data, qk)
                pin.updateFramePlacements(model, data)

                oMf_l = data.oMf[id_lf]
                oMf_r = data.oMf[id_rf]

                F_lw = pin.Force(HLF_world[k, 0:3], HLF_world[k, 3:6])
                F_rw = pin.Force(HRF_world[k, 0:3], HRF_world[k, 3:6])
                F_l_local = oMf_l.actInv(F_lw)
                F_r_local = oMf_r.actInv(F_rw)

                HLF[k, 0:3] = F_l_local.linear
                HLF[k, 3:6] = F_l_local.angular
                HRF[k, 0:3] = F_r_local.linear
                HRF[k, 3:6] = F_r_local.angular
        else:
            HLF = HRF = None
    else:
        raise ValueError("frame must be 'foot' or 'world'.")

    # ---- Plot: 2 columns (Left/Right), 2 rows (Forces/Moments) ----
    t = np.arange(T)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{title} ({frame.upper()} frame)")

    # Left foot
    axF_L, axM_L = axes[0, 0], axes[1, 0]
    axF_L.plot(t, LF[:, 0], label="Fx")
    axF_L.plot(t, LF[:, 1], label="Fy")
    axF_L.plot(t, LF[:, 2], label="Fz")
    if HLF is not None:
        axF_L.plot(t, HLF[:, 0], "--", alpha=0.7, label="Fx (human)")
        axF_L.plot(t, HLF[:, 1], "--", alpha=0.7, label="Fy (human)")
        axF_L.plot(t, HLF[:, 2], "--", alpha=0.7, label="Fz (human)")
    axF_L.set_ylabel("Force [N]")
    axF_L.set_title(f"{left_foot_name}")
    axF_L.grid(True); axF_L.legend(loc="best", ncol=3, fontsize=8)

    axM_L.plot(t, LF[:, 3], label="Mx")
    axM_L.plot(t, LF[:, 4], label="My")
    axM_L.plot(t, LF[:, 5], label="Mz")
    if HLF is not None:
        axM_L.plot(t, HLF[:, 3], "--", alpha=0.7, label="Mx (human)")
        axM_L.plot(t, HLF[:, 4], "--", alpha=0.7, label="My (human)")
        axM_L.plot(t, HLF[:, 5], "--", alpha=0.7, label="Mz (human)")
    axM_L.set_ylabel("Moment [N·m]")
    axM_L.set_xlabel("Time step")
    axM_L.grid(True); axM_L.legend(loc="best", ncol=3, fontsize=8)

    # Right foot
    axF_R, axM_R = axes[0, 1], axes[1, 1]
    axF_R.plot(t, RF[:, 0], label="Fx")
    axF_R.plot(t, RF[:, 1], label="Fy")
    axF_R.plot(t, RF[:, 2], label="Fz")
    if HRF is not None:
        axF_R.plot(t, HRF[:, 0], "--", alpha=0.7, label="Fx (human)")
        axF_R.plot(t, HRF[:, 1], "--", alpha=0.7, label="Fy (human)")
        axF_R.plot(t, HRF[:, 2], "--", alpha=0.7, label="Fz (human)")
    axF_R.set_title(f"{right_foot_name}")
    axF_R.grid(True); axF_R.legend(loc="best", ncol=3, fontsize=8)

    axM_R.plot(t, RF[:, 3], label="Mx")
    axM_R.plot(t, RF[:, 4], label="My")
    axM_R.plot(t, RF[:, 5], label="Mz")
    if HRF is not None:
        axM_R.plot(t, HRF[:, 3], "--", alpha=0.7, label="Mx (human)")
        axM_R.plot(t, HRF[:, 4], "--", alpha=0.7, label="My (human)")
        axM_R.plot(t, HRF[:, 5], "--", alpha=0.7, label="Mz (human)")
    axM_R.set_xlabel("Time step")
    axM_R.grid(True); axM_R.legend(loc="best", ncol=3, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    data = {
        "LF": LF, "RF": RF,
        "LF_local": LF_loc, "RF_local": RF_loc,
        "HLF": HLF, "HRF": HRF,
        "frame": frame
    }
    return fig, axes, data




def com_xy_traj(doc, xs_sol: np.ndarray, us_sol: np.ndarray | None):
    """
    Compute CoM XY trajectory using the CasADi function already built in `doc`.

    Parameters
    ----------
    doc : DocHumanMotionGeneration_InvDyn
        Instance with .com CasADi Function (inputs: [cx, cu] -> 3x1).
    xs_sol : (N, nx) array
        State trajectory.
    us_sol : (N, nu or nu+nf) array or None
        Control trajectory. If None, zeros are used.

    Returns
    -------
    com_xy : (N, 2) ndarray
        Center-of-mass XY over the trajectory.
    """
    N = xs_sol.shape[0]
    X = casadi.DM(np.asarray(xs_sol).T)
    if us_sol is None:
        # fall back to zeros with correct rows = doc.nv
        U = casadi.DM.zeros(doc.nv, N)
    else:
        U = casadi.DM(np.asarray(us_sol).T)

    com_map = doc.com.map(N)
    COM = com_map(X, U)             # 3 x N
    com_xy = np.asarray(COM.full()).T[:, :2]
    return com_xy


def bos_geometry_from_state(doc, xk: np.ndarray):
    """
    Build a constant BoS rectangle and heel/toe half-spaces from a single state xk.

    Returns
    -------
    bos_rect : (5,2) ndarray
        Closed rectangle polyline (min->max->min).
    (n_h, c_h) : (ndarray(2,), float)
        Heel-line half-space: n_h^T p - c_h >= 0 means inside.
    (n_t, c_t) : (ndarray(2,), float)
        Toe-line half-space:  n_t^T p - c_t >= 0 means inside.
    feet_pts : dict
        Keys: tiL, toL, heL, tiR, toR, heR, mid_toes, mid_heels (all np.ndarray(2,))
    """
    if not getattr(doc.param, "free_flyer", True):
        raise RuntimeError("BoS plotting requires free_flyer=True.")

    # BoS min/max (CasADi functions are already in `doc`)
    bos_min_dm, bos_max_dm = doc.bos_limits(xk)   # two outputs
    bos_min = np.array(bos_min_dm).ravel()
    bos_max = np.array(bos_max_dm).ravel()
    
    lower = np.asarray(bos_min).ravel()
    upper = np.asarray(bos_max).ravel()
    xmin, ymin = float(lower[0]), float(lower[1])
    xmax, ymax = float(upper[0]), float(upper[1])
    bos_rect = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

    # Foot points (world XY) from current state
    toe_in_lf, toe_out_lf, heel_lf, toe_in_rf, toe_out_rf, heel_rf = doc.foot_points(xk)
    tiL = np.array(toe_in_lf.full()).ravel()
    toL = np.array(toe_out_lf.full()).ravel()
    heL = np.array(heel_lf.full()).ravel()
    tiR = np.array(toe_in_rf.full()).ravel()
    toR = np.array(toe_out_rf.full()).ravel()
    heR = np.array(heel_rf.full()).ravel()

    # Midpoints define direction for inside half-space
    mid_toes = 0.5 * (tiL + tiR)
    mid_heels = 0.5 * (heL + heR)

    # Half-spaces (n^T p - c >= 0 inside)
    n_h, c_h = doc.halfspace_from_segment(heL, heR, toward_point=mid_toes)
    n_t, c_t = doc.halfspace_from_segment(toL, toR, toward_point=mid_heels)

    feet_pts = dict(
        tiL=tiL, toL=toL, heL=heL,
        tiR=tiR, toR=toR, heR=heR,
        mid_toes=mid_toes, mid_heels=mid_heels
    )
    return bos_rect, (n_h, c_h), (n_t, c_t), feet_pts


def bos_violations(com_xy: np.ndarray, n_h, c_h, n_t, c_t):
    """
    Check BoS heel/toe half-space satisfaction for CoM XY.

    Returns
    -------
    g_h, g_t : (N,) arrays of half-space values
    viol_h, viol_t : ints, number of violations
    """
    g_h = com_xy @ n_h - c_h
    g_t = com_xy @ n_t - c_t
    viol_h = int(np.sum(g_h < 0))
    viol_t = int(np.sum(g_t < 0))
    return g_h, g_t, viol_h, viol_t


def plot_com_vs_bos(
    doc,
    xs_sol: np.ndarray,
    us_sol: np.ndarray,
    q_hum: np.ndarray | None = None,
    k_bos: int = 0,
    ax: plt.Axes | None = None,
    title: str = "CoM vs BoS",
):
    """
    One-shot plot: CoM trajectory against a BoS rectangle + heel/toe lines.

    Parameters
    ----------
    doc : DocHumanMotionGeneration_InvDyn
        Instance used to solve & holding CasADi functions (com, bos_limits, foot_points).
    xs_sol : (N, nx)
        State trajectory.
    us_sol : (N, nu or nu+nf)
        Control trajectory.
    q_hum : (N, nq), optional
        Human reference joint trajectory for CoM overlay (velocities assumed zero).
    k_bos : int
        Index of the state used to derive BoS & half-spaces.
    ax : matplotlib Axes or None
        If None, create a figure.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib Axes
    (g_h, g_t, viol_h, viol_t) : tuple
        Half-space values and violation counts for diagnostics/logging.
    """
    assert 0 <= k_bos < xs_sol.shape[0], "k_bos out of range"

    # CoM estimate
    com_xy = com_xy_traj(doc, xs_sol, us_sol)

    # Optional human overlay using the same com() function (v=0)
    com_hum_xy = None
    if q_hum is not None:
        N = q_hum.shape[0]
        X_hum = casadi.DM(np.vstack([q_hum.T, np.zeros((doc.nv, N))]))
        U_dummy = casadi.DM.zeros(doc.nv, N)
        COM_hum = doc.com.map(N)(X_hum, U_dummy)
        com_hum_xy = np.asarray(COM_hum.full()).T[:, :2]

    # BoS & halfspaces from a representative state
    bos_rect, (n_h, c_h), (n_t, c_t), feet = bos_geometry_from_state(doc, xs_sol[k_bos])

    # Violations
    g_h, g_t, viol_h, viol_t = bos_violations(com_xy, n_h, c_h, n_t, c_t)
    print(f"Heel-line violations: {viol_h}, Toe-line violations: {viol_t}")

    # Plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        created_fig = True

    ax.plot(com_xy[:, 0], com_xy[:, 1], '-', label='CoM_est')
    if com_hum_xy is not None:
        ax.plot(com_hum_xy[:, 0], com_hum_xy[:, 1], '-', label='CoM_hum')

    # BoS rectangle
    ax.plot(bos_rect[:, 0], bos_rect[:, 1], '-', label='BoS')

    # Foot markers & heel/toe segments
    ax.plot([feet["tiL"][0], feet["toL"][0]], [feet["tiL"][1], feet["toL"][1]], 'o', ms=5, label='Left toes')
    ax.plot(feet["heL"][0], feet["heL"][1], 'o', ms=6, label='Left heel')
    ax.plot([feet["tiR"][0], feet["toR"][0]], [feet["tiR"][1], feet["toR"][1]], 'x', ms=6, label='Right toes')
    ax.plot(feet["heR"][0], feet["heR"][1], 'x', ms=7, label='Right heel')

    ax.plot([feet["heL"][0], feet["heR"][0]], [feet["heL"][1], feet["heR"][1]], lw=2, label='Heels line')
    ax.plot([feet["toL"][0], feet["toR"][0]], [feet["toL"][1], feet["toR"][1]], lw=2, label='Medial toes')

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_title(title)

    #if created_fig:
    #    plt.show()

    return ax, (g_h, g_t, viol_h, viol_t)
