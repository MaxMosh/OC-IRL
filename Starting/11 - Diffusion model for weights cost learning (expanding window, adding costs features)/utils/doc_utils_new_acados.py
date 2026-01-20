import casadi
from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from acados_template import AcadosOcp, AcadosOcpSolver

from pprint import pprint

 
 
class DocHumanMotionGeneration_InvDyn:
    
    def __init__(self,model,param):
         
        
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        
        self.model = model = pin.Model(model)
        self.data = data = model.createData()
         
        self.param=param
        
        self.nq = cmodel.nq 
        self.nv = cmodel.nv 

        # * Define the problem dimensions
        # We will define a state x = (q, v)^T to describe the robot dynamics
        nx = self.nq + self.nv     # state dimension: positions and velocities
        nu = self.nv               # control dimension: the accelerations including the free-flyer
        

        # * Create casadi symbolic variables
        # These variables are used to define symbolic expression and are replaced in the solver by some values according to the decision variables
        cx = casadi.SX.sym("x", nx, 1) # state: the positions and velocities
        cu = casadi.SX.sym("u", nu, 1) # control: the accelerations

        # SE3 position and rotation of the n targets to reach
        M_target = [pin.SE3(param.FOI_orientation[i], param.FOI_position[i]).copy()
        for i in range(len(param.FOI_orientation)) ]
         
        #### Calculation related to contraints
        cpin.forwardKinematics(self.cmodel, self.cdata, cx[:self.nq], cx[self.nq:],cu)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        cpin.updateFramePlacements(self.cmodel, self.cdata)                   # Update the frames placement to symbolic expressions in data
        cpin.centerOfMass(self.cmodel, self.cdata, cx[:self.nq]) 
            
        self.com = casadi.Function('com', [cx,cu], [self.cdata.com[0]]) # CoM
        self.vcom = casadi.Function('vcom', [cx,cu], [self.cdata.vcom[0]]) # CoM velocity
          
          
    
        
        if param.free_flyer:
            # * Definition of multi contact problem
            
            nf = 12               # Got two 6D forces
            cf = casadi.SX.sym("f", nf, 1) # two 6D external forces at feet level
            
            
            id_lf = model.getFrameId("left_foot")   # The left foot has to stay in contact with the floor
            id_rf = model.getFrameId("right_foot")     # The right foot has to stay in contact with the floor
            
            # get feet original pose (supposedly not moving during the entire task)
            pin.forwardKinematics(model, data, param.qdi)
            pin.updateFramePlacements(model, data)
            #self.M_lf = data.oMf[id_lf]   # placement of left foot in world frame
            #self.M_rf = data.oMf[id_rf]  # placement of right foot in world frame
            
            self.M_lf = pin.SE3(param.lfi_orientation,param.lfi_position) # SE3 position and rotation of the left foot to hold
            self.M_rf = pin.SE3(param.rfi_orientation,param.rfi_position) # SE3 position and rotation of the right foot to hold

            # get COP base of support limits
            self.BoS_min=np.zeros(2)
            for ax in range(2):# XY axes
                if self.M_lf.translation[ax]<=self.M_rf.translation[ax]:
                    self.BoS_min[ax]=self.M_lf.translation[ax]
                else:
                    self.BoS_min[ax]=self.M_rf.translation[ax]
                    
            self.BoS_max=np.zeros(2)
            for ax in range(2):# XY axes
                if self.M_lf.translation[ax]>=self.M_rf.translation[ax]:
                    self.BoS_max[ax]=self.M_lf.translation[ax]
                else:
                    self.BoS_max[ax]=self.M_rf.translation[ax]
            
            
            # --- Foot geometry (EDIT these constants to your foot last) ---
            self.toe_len  = getattr(self, "toe_len", 0.10)   # meters, ankle -> toe tip
            self.heel_len = getattr(self, "heel_len", 0.09)  # meters, ankle -> heel tip
            self.half_w_in  = getattr(self, "half_w_in",  0.15)  # medial  half-width
            self.half_w_out = getattr(self, "half_w_out", 0.045)  # lateral half-width

            # Choose which local axis is "forward" (0: x, 1: y). Keep it linear.
            fwd_axis = 0 # x is the cosmik model
            lat_axis = 1

            # Ankle placements (Pinocchio placements depending on cx)
            R_lf, t_lf = self.M_lf.rotation, self.M_lf.translation  # 3x3, 3x1 (SX/MX)
            R_rf, t_rf = self.M_rf.rotation, self.M_rf.translation

            # World XY unit axes for forward/lateral (symbolic, linear extraction)
            fwd_xy_lf = R_lf[:2, fwd_axis]  # 2x1
            lat_xy_lf = R_lf[:2, lat_axis]  # 2x1
            fwd_xy_rf = R_rf[:2, fwd_axis]
            lat_xy_rf = R_rf[:2, lat_axis]

            ankle_xy_lf = t_lf[:2]  # 2x1
            ankle_xy_rf = t_rf[:2]

            # Toe/heel points (all linear combinations of R columns and t)
            toe_in_lf  = ankle_xy_lf + self.toe_len  * fwd_xy_lf - self.half_w_in  * lat_xy_lf
            toe_out_lf = ankle_xy_lf + self.toe_len  * fwd_xy_lf + self.half_w_out * lat_xy_lf
            heel_lf    = ankle_xy_lf - self.heel_len * fwd_xy_lf

            toe_in_rf  = ankle_xy_rf + self.toe_len  * fwd_xy_rf - self.half_w_in  * lat_xy_rf
            toe_out_rf = ankle_xy_rf + self.toe_len  * fwd_xy_rf + self.half_w_out * lat_xy_rf
            heel_rf    = ankle_xy_rf - self.heel_len * fwd_xy_rf

            # Pack into a CasADi function from state -> six 2D points
            self.foot_points = casadi.Function(
                "foot_points",
                [cx],
                [toe_in_lf, toe_out_lf, heel_lf, toe_in_rf, toe_out_rf, heel_rf]
            )

            # (Optional) ankles too, if you want to plot them:
            self.ankles_xy = casadi.Function("ankles_xy", [cx], [ankle_xy_lf, ankle_xy_rf])

            
            
            # Function: from state -> (lower_bos, upper_bos)
            self.bos_limits = casadi.Function("bos_limits", [cx], [self.BoS_min,  self.BoS_max])

            # get the free flyer and feet symbolic poses
            oMf_ff = cdata.oMi[1]  # placement of root joint in world

            self.oMf_lf = cdata.oMf[id_lf]  # placement of left foot in world frame
            self.oMf_rf = cdata.oMf[id_rf ]  # placement of right foot in world frame
            
            force_lf = casadi.SX.sym("force_lf", 6)  # 6D spatial force  
            force_rf = casadi.SX.sym("force_rf", 6)  # 6D spatial force  
            
            cframe_lf = cmodel.frames[id_lf] #left foot frame
            cframe_rf = cmodel.frames[id_rf] #rigth foot  frame
            
            # * Build contact forces list
            forces = [ cpin.Force.Zero() for _ in cmodel.joints] #Initializes a list of zero forces, one per joint
            forces[cframe_lf.parentJoint] = cframe_lf.placement.act(cpin.Force(cf[0:6])) # cf[0:6] is the left foot wrench (spatial force: 3 force + 3 torque)
            forces[cframe_rf.parentJoint] = cframe_rf.placement.act(cpin.Force(cf[6:])) 

            cforces = cpin.StdVec_Force()
            for f in forces:
                cforces.append(f)

            if self.param.external_forces=="linear_zmp_distance_forces_estimation":# if the external GRFM under each foot at linearly interpolated
                
                cf_ff=casadi.SX.sym("f_ff", 6)
                
                total_force_ff = cpin.Force(cf_ff)#casadi.SX.sym("f_ff", 6))  # spatial force for free flyer 
                total_force_world=oMf_ff.act(total_force_ff)
            
            
                f = total_force_world.linear
                tau = total_force_world.angular
                
                fz = f[2] + 1e-8  # Add epsilon to avoid division by zero

                x_zmp = -tau[1] / fz  # -τ_y / f_z
                y_zmp =  tau[0] / fz  #  τ_x / f_z

                zmp = casadi.vertcat(x_zmp, y_zmp)  # ZMP in world frame (ground plane)
                self.zmp_world = casadi.Function("zmp_world", [cx, cu, cf_ff], [ zmp ])
                
                # Compute distances in world frame and weights 
                dl = casadi.norm_2( zmp - self.oMf_lf.translation[0:1])
                dr = casadi.norm_2( zmp - self.oMf_rf.translation[0:1])
                denom = dl + dr + 1e-8  # Avoid division by zero
            
                wl = dr / denom
                wr = dl / denom

                # Partition world-frame wrench 
                fl_world = wl * total_force_world #Note: the weight of the left foot depends on the distance to the right and vice versa — this ensures that if the ZMP is closer to the left foot, it bears more load.
                fr_world = wr * total_force_world

                # Express in local foot frames
                fl_local = self.oMf_lf.actInv(fl_world)
                fr_local = self.oMf_rf.actInv(fr_world)

                #  # Create CasADi function
                self.fl_local = casadi.Function("fl_local", [cx, cu, cf_ff], [ fl_local.vector ])
                self.fr_local = casadi.Function("fr_local", [cx, cu, cf_ff], [ fr_local.vector ])
        
            
            
            if self.param.external_forces=="optimal_forces_estimation":# if the external GRFM under each foot are decision variables of the ocp
                # Transform from foot to freeflyer
                lffMf = oMf_ff.inverse() * self.oMf_lf
                rffMf = oMf_ff.inverse() * self.oMf_rf
            
                # Express feet forces in the freeflyer frame
                force_lf_at_ff = lffMf.act(cpin.Force(force_lf))
                force_rf_at_ff = rffMf.act(cpin.Force(force_rf))
                
                # Create CasADi function
                self.lf_force_at_ff = casadi.Function("left_force_at_freeflyer", [cx, cu, force_lf], [force_lf_at_ff.vector])
                self.rf_force_at_ff = casadi.Function("right_force_at_freeflyer", [cx, cu, force_rf], [force_rf_at_ff.vector])

            
                # CoP a calculation
                # Extract vertical force (FY) and torques (Mx, MZ) in local frame

                Fy_l = cf[1]# Y is vertical axis in the local frame
                Mx_l = cf[3]
                Mz_l = cf[5]

                Fy_r = cf[1+6]
                Mx_r = cf[3+6]
                Mz_r = cf[5+6]

                zmp_l_local = casadi.vertcat(Mz_l / Fy_l, -Mx_l / Fy_l, 0)
                zmp_r_local = casadi.vertcat(Mz_r / Fy_r, -Mx_r / Fy_r, 0)

                zmp_l_global=casadi.mtimes(self.oMf_lf.rotation, zmp_l_local)+ self.oMf_lf.translation
                zmp_r_global=casadi.mtimes(self.oMf_rf.rotation, zmp_r_local)+ self.oMf_rf.translation

                F_lg=self.oMf_lf.rotation@cf[0:3] #left forces in the GSR
                F_rg=self.oMf_rf.rotation@cf[6:9] #right  forces in the GSR
            
                self.F_lg=casadi.Function("F_lg", [cx,cu,cf], [F_lg])
                self.F_rg=casadi.Function("F_rg", [cx,cu,cf], [F_rg])
            
                zmp_world = ( F_lg[2]* zmp_l_global +  F_rg[2] * zmp_r_global) / (F_lg[2] + F_rg[2] + 1e-6) # barycenter to get the total cop
                zmp_world[2] =0
                self.zmp_world=casadi.Function("zmp_world", [cx,cu,cf], [zmp_world])
        
        # * Contraint on the contacts
        
            # Position error
            self.pos_error_lf = casadi.Function('placement_contact_error_lf', [cx, cu], [self.approx_log6(id_lf, self.M_lf,cx,cu)])
            self.pos_error_rf = casadi.Function('placement_contact_error_rf', [cx, cu], [self.approx_log6(id_rf, self.M_rf,cx,cu)])
            
            # Velocity error
            self.vel_error_lf = casadi.Function('velocity_contact_error_lf', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_lf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
            self.vel_error_rf = casadi.Function('velocity_contact_error_rf', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_rf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
        
            # Acceleration error
            self.acc_cstr_lf = casadi.Function('acc_constraint_error_lf', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_lf, pin.LOCAL).vector])
            self.acc_cstr_rf = casadi.Function('acc_constraint_error_rf', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_rf, pin.LOCAL).vector])

        # to be used for torque control
        # a = cpin.aba(cmodel, cdata, cx[:self.nq], cx[self.nq:], cu, cforces)           # a is a symbolic expression corresponding to the joints acceleration and depending on the symbolic variables cx and cu
        # cpin.forwardKinematics(cmodel, cdata, cx[:self.nq], cx[self.nq:], a)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        # cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data
        # self.acc = casadi.Function( "xdot", [cx, cu, cf], [a])  # Casadi function: Takes values of cx and cu and returns corresponding value of the symbolic expression a



        
        
        # Calculations related to the cost functions
        
        self.tau_free=casadi.Function('tau_freeflyer',[cx, cu],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)  ])# joint torques without external wrench 
        
        if param.free_flyer:
            self.tau=casadi.Function('tau',[cx, cu, cf],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu, cforces)  ])#  joint torques with external wrench 
        else:
            self.tau=casadi.Function('tau',[cx, cu],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)  ])#  joint torques without external wrench 

        energy=[]
        for j in range(6,self.nv):
            if param.free_flyer:
                energy+=casadi.fabs(cx[self.nq+j]*self.tau(cx,cu,cf)[j])
            else:
                energy+=casadi.fabs(cx[self.nq+j]*self.tau(cx,cu)[j])
                
        if param.free_flyer:        
            self.energy=casadi.Function('energy',[cx, cu, cf],[ energy  ]) 
        else:
            self.energy=casadi.Function('energy',[cx, cu],[ energy  ]) 
            
            
        dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)
        if param.free_flyer:     
            self.dtau=casadi.Function('dtau',[cx, cu, cf],[ dtau_dq@cx[self.nq:]    ])
        else:
            self.dtau=casadi.Function('dtau',[cx, cu],[ dtau_dq@cx[self.nq:]    ])
            
        # Casadi Functions for cost function definition
        # self.geodesic=casadi.Function('geodesic',[q,dq,tau],[ dq.T@cdata.M@dq   ])
        # self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[-1].translation[[0,2]] ])
        # self.vtip =casadi.Function('vtip', [q,dq,tau], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('hand') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
    

        # * Cost on the distance to the target(s)-eventually
        # Loop over all pairs (FOI id, target)
        cost_to_target=[]
        for foi_id, target in zip(param.FOI_to_set_Id, M_target):
            cost_to_target.append(self.approx_log6(foi_id, target, cx, cu))
        
        self.cost_to_target = casadi.Function('cost_to_target', [cx, cu], [casadi.vertcat( *cost_to_target) ])

        
        def normalize_quat(q):
            quat = q[3:7]
            quat_norm = quat / casadi.norm_2(quat)
            return casadi.vertcat(q[:3], quat_norm, q[7:])
        
    
        # Normalize quaternion part of q before integration
        #q_in = normalize_quat(cx[:self.nq])
        dt=self.param.Tf/self.param.nb_samples
        
        # Integrate position (q)
        q_next = cpin.integrate(self.cmodel, cx[:self.nq], cx[self.nq:] *  dt)

        # Update velocity (dq) using Euler forward
        dq_next = cx[self.nq:] + cu * dt#self.acc(cx, cu[:self.nv], cu[self.nv:])#cu * self.param.dt
        
        # Create CasADi function for next state concatenation
        self.dyn_fun = casadi.Function('dyn_fun', [cx, cu], [casadi.vertcat(q_next, dq_next)])
        
  
    
    def solve_doc_acados(self,param):
          
        #T=int(param.nb_samples)
        #quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), np.deg2rad(0), 0)).coeffs()# set the model up rigth
        #q0=pin.neutral(self.model) 
        #q0[3:7]=quat
       
        
        q0=param.qdi#
        
        if param.free_flyer:
            # calculate static initial conditions for external forces and moments
            pin.forwardKinematics(self.model, self.data, q0)
            pin.updateFramePlacements(self.model, self.data)
            
            tau_ff0=pin.rnea(self.model,self.data,q0,np.zeros(self.nv),np.zeros(self.nv)) 
            tau_ff0 = np.asarray(tau_ff0 )#.reshape(6,)
            force_ff = pin.Force(tau_ff0[:3].reshape(3,), tau_ff0[3:6].reshape(3,))
            F_world = self.data.oMi[0].act(force_ff)
        
            F_lf = self.M_lf.inverse().actInv(F_world/2) # transforms F_world from world into the left foot frame.
            F_rf = self.M_rf.inverse().actInv(F_world/2) # transforms F_world from world into the right foot frame.
 
        
      
        # * Define the problem dimensions
        # define a state x = (q, v)^T to describe the robot dynamics
        nq=self.nq
        nv=self.nv
        nx =  nq +  nv     # state dimension
        nu = nv           # control
        nf = 12           # Got two 6D forces at the feet
        
       # -------------------------------------
        # ACADOS Optimal Control Problem (OCP)
        # -------------------------------------
        ocp = AcadosOcp()

        
        cx = casadi.SX.sym("cx", nx, 1) # state: the positions and velocities
        if param.free_flyer:
            cu = casadi.SX.sym("cu", nu + nf, 1) # control: the torques and the two external wrenches
        else:
            cu = casadi.SX.sym("cu", nu , 1)
            
        ocp.model.disc_dyn_expr = self.dyn_fun(cx, cu[:nv])
        ocp.model.dyn_type = "discrete"
        
        # Solve
        ocp.solver_options.integrator_type = "DISCRETE"
        
        ocp.model.p = []

        ocp.model.name = "human_turbo_doc"
        ocp.model.x = cx
        ocp.model.u = cu


        # # Time horizon
        ocp.solver_options.N_horizon = self.param.nb_samples
        ocp.solver_options.tf = param.Tf


        ## path cost
        ocp.cost.cost_type = 'NONLINEAR_LS'
        
        cost_terms, W_blocks, slice_map, target_slices = self.calculate_cost(cx, cu, param) # go to declaration to add more cost functions
         
        ocp.model.cost_y_expr = casadi.vertcat(*cost_terms)
        ocp.cost.yref = np.zeros(int(ocp.model.cost_y_expr.shape[0]))# yref is filled with zeros
        ocp.cost.W = casadi.diagcat(*[casadi.DM(np.atleast_2d(B)) for B in W_blocks]).full()   # build W (acados expects a numpy array)
        
       
        
   
    
   
        

    # # # terminal cost (recopy of path cost but we keep it like that for clarity)
        
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        cost_terms_e = []
        W_blocks_e   = []
        axes_map = {"x":[0], "y":[1], "z":[2], "xy":[0,1], "xz":[0,2], "yz":[1,2], "xyz":[0,1,2]}

        for i in range(len(param.FOI_to_set_Id) ):
            block_i = self.cost_to_target(cx, cu[:nv])[i*6:(i+1)*6]
       
            ax_target = axes_map[param.FOI_axes[i]]
            if param.FOI_sample[0] >=param.nb_samples-1:  
                #print("target at the end of the motion")  
                cost_terms_e.append(block_i[ax_target])
                W_target =param.weights["target"][i] * np.eye(len(ax_target))
                W_blocks_e.append(W_target)
            
        if param.free_flyer:
            W_contact_pos=1e1*np.eye(6)         # regularisation terms (those should not be modified)

            cost_terms_e.append(self.pos_error_lf(cx, np.zeros(nv)) )  # 6
            cost_terms_e.append(self.pos_error_rf(cx,np.zeros(nv))  )  # 6
            W_blocks_e += [W_contact_pos, W_contact_pos]    # 6x6 each
            
        ocp.model.cost_y_expr_e = casadi.vertcat(*cost_terms_e)    
        ocp.cost.yref_e =  np.zeros(int(ocp.model.cost_y_expr_e.shape[0]) ) 
        dm_blocks_e = [casadi.DM(np.atleast_2d(B)) for B in W_blocks_e]  # ensure 2-D CasADi DM
        ocp.cost.W_e = casadi.diagcat(*dm_blocks_e).full()

        
        ## State bounds
        if param.free_flyer:
            q_min = np.array(self.model.lowerPositionLimit[7:])
            q_max = np.array(self.model.upperPositionLimit[7:])
            ocp.constraints.idxbx = np.arange(7, nq)         # indices in x to constrain in the state-vector

        else:
            q_min = np.array(self.model.lowerPositionLimit)
            q_max = np.array(self.model.upperPositionLimit)
            ocp.constraints.idxbx = np.arange(0, nq)         # indices in x to constrain in the state-vector

        # # # joint limits
        ocp.constraints.lbx = np.array(q_min)
        ocp.constraints.ubx = np.array(q_max)

        ## control bounds
        # indices of vertical forces in u
        # idx_fy_lf = nv + 1         # [Fx,Fy,Fz,Mx,My,Mz] -> Fy is +1 and vertical in foot frame
        # idx_fy_rf = nv + 6 + 1     # second foot block

        # ocp.constraints.idxbu = np.array([idx_fy_lf, idx_fy_rf], dtype=int)
        # ocp.constraints.lbu   = np.array([0.0, 0.0])      # Fy >= 0
        # ocp.constraints.ubu   = np.array([5e3, 5e3])      # no upper boun
        
        # Add constraints to the optimization problem
        self.con_terms = []
        self.lh_blocks = []
        self.uh_blocks = []
        self.con_slice_map = {}

        idx_con = 0  # running index in stacked h(x,u,p)
        
        
        
        
        
        if param.free_flyer:
            
            tol = 5e-3
            s_tgt0 = casadi.SX.sym('s_tgt0', 1)     # selector parameter (index 0)

            # Start p with selector
            ocp.model.p = s_tgt0
            self.idx_s_tgt0 = 0
            
            # Append BOS(6) so they are the LAST 6 in p
            p_bos = casadi.SX.sym('p_bos', 6)
            ocp.model.p = casadi.vertcat(ocp.model.p, p_bos)

            # parameter_values of correct length (1 + 6)
            ocp.parameter_values = np.zeros(int(ocp.model.p.size1()))

            # ---------------------------
            # Constraint: target0 z only (scaled by tol, gated by selector)
            # ---------------------------
            z_err_t0 = self.cost_to_target(cx, cu[:self.nv])[2]    # scalar
            expr_tgt0 = casadi.vertcat(s_tgt0 * (z_err_t0 / tol))  # 1×1
            idx_con = self._add_con("target0_z", expr_tgt0, np.array([-1.0]), np.array([+1.0]), idx_con)

        
        
        
        
        # Bos parameters always at the end of parameters vector
        if param.free_flyer:
           # ocp.model.p = casadi.SX.sym('p_bos', 6) # [n_hx, n_hy, c_h, n_tx, n_ty, c_t]
           # ocp.parameter_values = np.zeros((int(ocp.model.p.size1()),))

            com_xy = self.com(cx, cu[:nv])[0:2] 
            
                # unpack parameters
            n_hx, n_hy, c_h, n_tx, n_ty, c_t = (
                ocp.model.p[-6], ocp.model.p[-5], ocp.model.p[-4],
                ocp.model.p[-3], ocp.model.p[-2], ocp.model.p[-1]
            )
            
            # foot polygon from initial joint confiuration
            toe_in_lf, toe_out_lf, heel_lf, toe_in_rf, toe_out_rf, heel_rf = self.foot_points(np.concatenate( (param.qdi,np.zeros(nv))) )
             
         # Convert to np arrays
            tiL = np.array(toe_in_lf.full()).ravel()
            toL = np.array(toe_out_lf.full()).ravel()
            heL = np.array(heel_lf.full()).ravel()
            tiR = np.array(toe_in_rf.full()).ravel()
            toR = np.array(toe_out_rf.full()).ravel()
            heR = np.array(heel_rf.full()).ravel()

            mid_toes  = 0.5*(tiL + tiR)
            mid_heels = 0.5*(heL + heR)

            # Half-space 1: heels line, inside pointing toward toes
            n_h, c_h = self.halfspace_from_segment(heL, heR, mid_toes)

            # Half-space 2: medial toes line, inside pointing toward heels
            n_t, c_t = self.halfspace_from_segment(toL, toR, mid_heels)
            p_bos_vals = np.array([n_h[0], n_h[1], c_h, n_t[0], n_t[1], c_t])

            g_h = n_hx*com_xy[0] + n_hy*com_xy[1] - c_h   # >= 0
            g_t = n_tx*com_xy[0] + n_ty*com_xy[1] - c_t   # >= 0
            
            dyn_cons = 1*(self.lf_force_at_ff(cx, cu[:nv], cu[nv:nv+6]) + self.rf_force_at_ff( cx, cu[:nv], cu[nv+6:] ) - self.tau_free(cx, cu[:nv])[:6]) # 
           
            
            #initialize base param vector once
            p_base = np.copy(ocp.parameter_values)
            p_base[0] = 0.0    # selector default 0 at index 0
        
         
            # ---- add blocks ----
            idx_con = self._add_con("dyn_cons",  dyn_cons,                         -1e-2*np.ones(6), +1e-2*np.ones(6), idx_con)
            idx_con = self._add_con("pose_lf",   self.pos_error_lf(cx, cu[:self.nv]), -1e-2*np.ones(6), +1e-2*np.ones(6), idx_con)
            idx_con = self._add_con("pose_rf",   self.pos_error_rf(cx, cu[:self.nv]), -1e-2*np.ones(6), +1e-2*np.ones(6), idx_con)
            idx_con = self._add_con("bos_heels", g_h,                              np.array([0.0]),  np.array([1e6]),     idx_con)
            idx_con = self._add_con("bos_toes",  g_t,                              np.array([0.0]),  np.array([1e6]),     idx_con)         
            
            # put BOS at the tail (last 6)
            p_base[-6:] = p_bos_vals
     
       
            ocp.parameter_values = p_base
        
        
        
        if self.con_terms:# check if there are actual constraints
          
            ocp.model.con_h_expr = casadi.vertcat(*self.con_terms)
            ocp.constraints.lh   = np.concatenate(self.lh_blocks)
            ocp.constraints.uh   = np.concatenate(self.uh_blocks)
           
 
        

        epsilon = np.deg2rad(1)  # 0.5cm tolerance
        # ocp.model.nh_e = nq+nv
        # q_final_cons=cx[:nq]-param.qdf
        # ocp.model.con_h_expr_e =  casadi.vertcat(q_final_cons, cx[nq:]) # final joint configuration  and joint velocity are zero at terminal node
        
        # ocp.constraints.lh_e = -epsilon * np.ones(nq+nv)
        # ocp.constraints.uh_e = +epsilon * np.ones(nq+nv)
        ocp.model.nh_e = nv
        q_final_cons=cx[:nq]-param.qdf
        ocp.model.con_h_expr_e =  casadi.vertcat(cx[nq:]) # final joint configuration  and joint velocity are zero at terminal node
        
        ocp.constraints.lh_e = -epsilon * np.ones( nv)
        ocp.constraints.uh_e = +epsilon * np.ones( nv)
      
     # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        #ocp.solver_options.qp_solver_cond_N = 5  # horizon is long and DOF is modest, partial condensing makes the QP smaller

        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        #ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        #ocp.solver_options.ext_cost_num_hess = 1  # 1 = exact Hessian from CasADi
        # Relax line search to allow more aggressive steps
        #ocp.solver_options.line_search_use_sufficient_descent = 1

        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.regularize_method = 'MIRROR' # if SQP tehn regularize the hessian vaialble are NO_REGULARIZE, MIRROR, PROJECT, PROJECT_REDUC_HESS, CONVEXIFY, GERSHGORIN_LEVENBERG_MARQUARDT.
        ocp.solver_options.reg_epsilon = 1e-4#1e-6 more regularization can help larger steps and fewer backtracks:
        
        # QP options
        ocp.solver_options.qp_tol_stat  = 1e-4
        ocp.solver_options.qp_tol_eq    = 1e-4
        ocp.solver_options.qp_tol_ineq  = 1e-4
        ocp.solver_options.qp_tol_comp  = 1e-6


        # NLP options  

        ocp.solver_options.nlp_solver_tol_stat = 1e-1 # KKT residual value: most important stopping criteria !
        ocp.solver_options.nlp_solver_tol_eq   = 1e-3
        ocp.solver_options.nlp_solver_tol_ineq = 1e-3  
        ocp.solver_options.qp_tol_comp  = 5e-6
        
        ocp.solver_options.nlp_solver_max_iter = 100
        #ocp.solver_options.qp_solver_iter_max = 10 
        ocp.solver_options.qp_solver_warm_start = 1
         
        
        ocp.solver_options.print_level = 1
        
        
        ## Initial guess      
        ocp.constraints.x0 = np.concatenate( [q0,np.zeros(self.nv)] )
        
        
        # Create solver
        
        if param.build_solver:
            print("building ocp now")
            ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json") 
        else:
            ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json", build=False, generate=False)
        
        ## NOTE: From here all parameters,weigths are rewriting at run time the ones set duing build
        if param.free_flyer:
            
            k_star = int(np.clip(int(param.FOI_sample[0] if isinstance(param.FOI_sample, (list, tuple, np.ndarray)) else param.FOI_sample), 0, int(ocp.solver_options.N_horizon)-1))


            ## Pass parameters for the Base of Support definition (from feet corners). The ffet are assumed fixed for all trajectory
            for k in range(param.nb_samples + 1):   # include terminal stage
                p_vec = np.copy(ocp.parameter_values)
                #p_vec[-6:] = p_bos_vals
                p_vec[0] = 1.0 if k == k_star else 0.0
                ocp_solver.set(k, "p", p_vec)
                
                
        #set the cost function and target weigths. For target default uses ramp interpolation 0-->1 at desired time index
        self.update_path_weights_absolute(ocp_solver, ocp, slice_map, param, ramp_targets=True)
        
        
        # --- update end stage weights at runtime  --- Un-used for the moment
        dm_blocks_e = [casadi.DM(np.atleast_2d(B)) for B in W_blocks_e]  # ensure 2-D CasADi DM
        W_e = casadi.diagcat(*dm_blocks_e).full()
        ocp_solver.cost_set(ocp.solver_options.N_horizon, "W", W_e)
        
        
        # Set initial
        t0 = time.time()
        for k in range(self.param.nb_samples-1):
        
            if param.free_flyer:     
                ocp_solver.set(k, "u", np.concatenate([np.zeros(self.nv), F_lf.linear,F_lf.angular,F_rf.linear,F_rf.angular]))
            else:
                ocp_solver.set(k, "u", np.concatenate([np.zeros(self.nv)]))

        
        for repeat_solve in range(1):
            print("iter:")
            print(repeat_solve)
            
            status = ocp_solver.solve()
        
        
            for k in range(self.param.nb_samples):
                ocp_solver.set(k, "x", ocp_solver.get(k+1, "x"))
                if k<self.param.nb_samples-1:
                    ocp_solver.set(k, "u", ocp_solver.get(k+1, "u"))
        
        
        elapsed = time.time() - t0
        print(f"Solve time: {elapsed*1e3:.3f} ms")     
        
        if status != 0:
            raise RuntimeError(f" ❌ acados solve failed with status={status}")
            #print("❌ Solver failed with status:", status)
        else:
            print("✅ Solver succeeded")


            simX = np.zeros((self.param.nb_samples+1, nx))
            if param.free_flyer:
                simU = np.zeros((self.param.nb_samples, nu+12))
            else:
                simU = np.zeros((self.param.nb_samples, nu))
                
            # get solution
            for i in range(self.param.nb_samples):
                simX[i,:] = ocp_solver.get(i, "x")
                simU[i,:] = ocp_solver.get(i, "u")
            simX[self.param.nb_samples,:] = ocp_solver.get(self.param.nb_samples, "x")
        
            
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, simX[ param.FOI_sample[0], :nq])  
            pin.updateFramePlacements(self.model, self.data)                   

            actual_pose = self.data.oMf[self.model.getFrameId("right_hand")].copy()  # Desired EE pose

            print("position error")
            print(self.param.FOI_position[0]-actual_pose.translation)
            
            # h_vals = np.array([h_fun(simX[k], simU[k]).full().squeeze() 
            #        for k in range(len(simU))])

            # plt.plot(h_vals)
            # plt.title("h values over trajectory")
            # plt.show()
            
            
        ########## ADD This to a plot function
            
            # joint_indices = np.arange(7,  nq)  # e.g., joints excluding free-flyer
            # joint_names=[]
            # for idx in range(2,nq-5):
               
            #     joint_names.append(self.model.names[idx] )
            
            # #joint_names=joint_names[1:]
           
            # t = np.arange(simX.shape[0])  # discrete time steps
            
            
            # fig, axes = plt.subplots(int(len(joint_names)/2), 3, figsize=(6, 0.8*(len(joint_names))), sharex=True)
            # axes = axes.flatten()
            
            # for i, ax in enumerate(axes):  # start=1 to skip universe
            #     if i < nq-7:
            #         q_i = simX[:,i+7]

            #         ax.plot(q_i, 'r-', label="trajectory")
            #         ax.axhline(q_min[i], color="k", linestyle="--", label="q_min" if i == 1 else "")
            #         ax.axhline(q_max[i], color="k", linestyle="--", label="q_max" if i == 1 else "")
            #         ax.plot(t[0], q0[i+7], "kx", markersize=10, mew=2)
            #         ax.set_ylabel(joint_names[i])
            #         ax.grid(True)

            # axes[-1].set_xlabel("time step")
            # axes[0].legend()
            # plt.tight_layout()
            # plt.show()  
            
 
      

        u_sol=simU[:,:nu]
        fs_sol=simU[:,nu:]
        return simX, u_sol, fs_sol   
        
    
    
    
    def calculate_cost(self, cx, cu, param):
        """
        Build path-cost terms and base weights, honoring `param.variables_w`.
        Returns:
        y_terms      : list[CasADi SX]
        W_blocks     : list[np.ndarray]
        slice_map    : dict[str, slice]
        target_slices: list[slice]
        """
        nq, nv = self.nq, self.nv
        y_terms, W_blocks = [], []
        slice_map, target_slices = {}, []
        idx = 0

        def _push(name, expr, W):
            nonlocal idx
            n = int(expr.shape[0])
            y_terms.append(expr)
            W_blocks.append(np.asarray(W, dtype=float).reshape(n, n))
            slice_map[name] = slice(idx, idx + n)
            idx += n

        def _as_W(w, n, variables_w):
            """
            variables_w=True  -> use first element if w is array-like (stage profile), scalar*I.
            variables_w=False -> scalar*I, len-n vector -> diag, (n,n) matrix -> pass-through.
            """
            # Fast path: scalar
            if np.isscalar(w):
                return float(w) * np.eye(n)

            w_arr = np.asarray(w)
            if variables_w:
                # Compile with first scalar; you will rescale at runtime
                return float(np.ravel(w_arr)[0]) * np.eye(n)

            # Non-variable mode: accept richer shapes
            if w_arr.shape == (n,):
                return np.diag(w_arr.astype(float))
            if w_arr.shape == (n, n):
                return w_arr.astype(float)
            if w_arr.size == 1:
                return float(w_arr.item()) * np.eye(n)

            raise ValueError(
                f"Incompatible weight shape {w_arr.shape} for block size {n} "
                f"(expected scalar, ({n},) or ({n},{n}))"
            )

        # ===== Global / regularization terms (stable order) =====
        # if "min_joint_torque" in param.active_costs:
        #     expr = self.tau(cx, cu[:nv], cu[nv:]) if param.free_flyer else self.tau(cx, cu[:nv])
        #     _push("min_joint_torque", expr, _as_W(param.weights["min_joint_torque"], nv, param.variables_w))

        if "min_joint_torque" in param.active_costs:
            tau_expr = self.tau(cx, cu[:nv], cu[nv:]) if param.free_flyer else self.tau(cx, cu[:nv])

            if getattr(param, "groups_joint_torques", {"all": True}).get("all", True):
                 
                if param.free_flyer:
                # EXCLUDE the free-flyer (first 6)
                    expr_all = tau_expr[6:nv]#tau_expr[6:nv]
                    _push("min_joint_torque",
                        expr_all,
                        _as_W(param.weights["min_joint_torque"], (nv - 6), param.variables_w))
                else:
                    expr_all = tau_expr
                    _push("min_joint_torque",
                        expr_all,
                        _as_W(param.weights["min_joint_torque"], (nv), param.variables_w))
                
            else:
                # One block per group, with correct indices using idx_v
                for gname, jlist in param.groups_joint_torques.items():
                    if gname == "all":
                        continue  # flag, not a group
                    group_terms = []
                    for jname in jlist:
                        
                        

                        jid = self.model.getJointId(jname)
                        j = self.model.joints[jid]
                        
                        i0 = j.idx_v                    # start index in v/tau
                        nj = j.nv                       # number of DoFs for this joint
                        # Skip free-flyer DoFs if any (safety)
                        if i0 < 6:
                            i0 = 6 if i0 + nj > 6 else i0  # ensure no FF indices enter
                        group_terms.append(tau_expr[i0:i0+nj])
                    group_expr = casadi.vertcat(*group_terms)
                    w_spec = param.weights["min_joint_torque"][gname]
                    Wg = _as_W(w_spec, int(group_expr.shape[0]), param.variables_w)
                    _push(f"min_joint_torque/{gname}", group_expr, Wg)
            
        
        if "min_torque_change" in param.active_costs:
            expr = self.dtau(cx, cu[:nv])                             
            _push("min_torque_change", expr, _as_W(param.weights["min_torque_change"], nv, param.variables_w))
        
        if "min_com_deviation" in param.active_costs:
            expr = param.com_hum[0, 0] - self.com(cx, cu[:nv])[0]    # 1-dim
            _push("min_com_deviation", expr, _as_W(param.weights["min_com_deviation"], 1, param.variables_w))

        if "min_com_velocity" in param.active_costs:
            expr = self.vcom(cx, cu[:nv])                            # 3-dim
            _push("min_com_velocity", expr, _as_W(param.weights["min_com_velocity"], 3, param.variables_w))

        if "min_joint_acc" in param.active_costs:
            if param.free_flyer:
                expr = cu[6:nv]                                          # (nv-6)
                _push("min_joint_acc", expr, _as_W(param.weights["min_joint_acc"], nv-6, param.variables_w))
            else:
                expr = cu[:nv]                                           
                _push("min_joint_acc", expr, _as_W(param.weights["min_joint_acc"], nv, param.variables_w))
 
        if "min_joint_vel" in param.active_costs:
            if param.free_flyer:
                expr = cx[nq+6:]                                         # (nv-6)
                _push("min_joint_vel", expr, _as_W(param.weights["min_joint_vel"], nv-6, param.variables_w))
            else:
                expr = cx[nq:]                                         # (nv-6)
                _push("min_joint_vel", expr, _as_W(param.weights["min_joint_vel"], nv, param.variables_w))
                
        # Joint position regularization (fixed)
        _push("q_reg", cx[:nq], 5e-3 * np.eye(nq))

        # ===== Contact regularization =====
        if param.free_flyer:
            _push("ext_forces", cu[nv:], 1e-3 * np.eye(12))
            _push("acc_lf", self.acc_cstr_lf(cx, cu[:nv]), 1e3 * np.eye(6))
            _push("acc_rf", self.acc_cstr_rf(cx, cu[:nv]), 1e3 * np.eye(6))
            _push("pose_lf", self.pos_error_lf(cx, cu[:nv]), 1e1 * np.eye(6))
            _push("pose_rf", self.pos_error_rf(cx, cu[:nv]), 1e1 * np.eye(6))

        # ===== Targets (always last) =====
        axes_map = {"x":[0], "y":[1], "z":[2], "xy":[0,1], "xz":[0,2], "yz":[1,2], "xyz":[0,1,2]}
        for i, foi_id in enumerate(param.FOI_to_set_Id):
            block6 = self.cost_to_target(cx, cu[:nv])[i*6:(i+1)*6]
            ax = axes_map[param.FOI_axes[i]]
            expr = block6[ax]
            name = f"target/{i}"
            # Note: target weights may be scalar or per-axis scalar; use same policy
            _push(name, expr, _as_W(param.weights["target"][i], len(ax), param.variables_w))
            target_slices.append(slice_map[name])

        return y_terms, W_blocks, slice_map, target_slices

    
    def _add_con(self,name, expr, lb, ub, idx):
        """
        Append one constraint block.
        expr : CasADi vector (m x 1)
        lb,ub: array-like length m (or scalars broadcastable to m)
        Returns:
        new_idx (int)
        """
        m = int(expr.shape[0])
        self.con_terms.append(expr)

        lb_arr = np.atleast_1d(lb).astype(float).reshape(m,)
        ub_arr = np.atleast_1d(ub).astype(float).reshape(m,)
        self.lh_blocks.append(lb_arr)
        self.uh_blocks.append(ub_arr)

        self.con_slice_map[name] = slice(idx, idx + m)
        return idx + m  # updated index
    
    
    def update_path_weights_absolute(self, ocp_solver, ocp, slice_map, param, ramp_targets=True):
        
        # uncomment to check if all joint torques are in group lists
        tau_idx_from_groups = []
        for gname, jlist in param.groups_joint_torques.items():
            if gname == "all":
                continue
            for jname in jlist:
                print(jname)
                jid = self.model.getJointId(jname)
                j   = self.model.joints[jid]
                i0, nj = j.idx_v, j.nv
                if i0 < 6:
                    continue  # skip free-flyer completely
                tau_idx_from_groups += list(range(i0, i0+nj))

        print("Missing:", sorted(set(range(6, self.nv)) - set(tau_idx_from_groups)))
        print("Duplicated:", [i for i in set(tau_idx_from_groups) if tau_idx_from_groups.count(i) > 1])
        
        """
        After-build path-cost weight update (absolute assignment).
        - Non-targets: scalar/per-window/per-stage → val*I ; (n,) → diag ; (n,n) → as-is
        - Targets: same as above, multiplied by triangular alpha[k] if ramp_targets=True
        """
        W_base = ocp.cost.W.copy()  # keeps untouched blocks as compiled
        N      = int(ocp.solver_options.N_horizon)
        nb_w   = int(getattr(param, "nb_w", 1))

        # target slices present in cost
        target_keys   = sorted([k for k in slice_map if k.startswith("target/")],
                            key=lambda s: int(s.split("/")[1]))
        target_blocks = [(slice_map[k], int(k.split("/")[1])) for k in target_keys]

        # triangular ramp (0→1→0), peak at FOI_sample, forced fall
        if ramp_targets and N > 0:
            s = param.FOI_sample[0] if isinstance(param.FOI_sample, (list, tuple, np.ndarray)) else param.FOI_sample
            peak = int(np.clip(int(s), 0, max(N - 1, 0)))
            if N > 1 and peak >= N - 1:
                peak = N - 2
            alpha = np.ones(N) if N <= 1 else np.r_[np.linspace(0,1,peak+1), np.linspace(1,0,N-peak)]
        else:
            alpha = np.ones(N)

        for k in range(N):
            Wk = W_base.copy()
            wi = min(nb_w - 1, (k * nb_w) // max(N, 1)) if nb_w > 0 else 0  # window index

            # # --- Non-targets ---
            # for name in getattr(param, "active_costs", []):
            #     if name not in slice_map or name not in param.weights:
            #         continue
            #     sl = slice_map[name]; n = sl.stop - sl.start
            #     spec = param.weights[name]

            #     if np.isscalar(spec):
            #         W_block = float(spec) * np.eye(n)
            #     else:
            #         arr = np.asarray(spec)
            #         if arr.ndim == 1 and arr.size == N:       # per-stage scalars
            #             W_block = float(arr[k]) * np.eye(n)
            #         elif arr.ndim == 1 and arr.size == nb_w:  # per-window scalars
            #             W_block = float(arr[wi]) * np.eye(n)
            #         elif arr.shape == (n,):                   # per-axis
            #             W_block = np.diag(arr.astype(float))
            #         elif arr.shape == (n, n):                 # full matrix
            #             W_block = arr.astype(float)
            #         else:                                     # fallback
            #             W_block = float(arr.ravel()[0]) * np.eye(n)

            #     Wk[sl, sl] = W_block

            # --- Non-targets (including grouped torque) ---
            for name, sl in slice_map.items():
                if name.startswith("target/"):
                    continue

                n = sl.stop - sl.start

                # pick weight spec
                if name.startswith("min_joint_torque/"):
                    gname = name.split("/", 1)[1]
                    if "min_joint_torque" not in param.weights:
                        continue
                    spec = param.weights["min_joint_torque"][gname]
                elif name in getattr(param, "active_costs", []) and name in param.weights:
                    spec = param.weights[name]
                else:
                    continue  # not configured → leave compiled value

                # resolve absolute block W for this stage k (not scaling the compiled base)
                if np.isscalar(spec):
                    W_block = float(spec) * np.eye(n)
                else:
                    arr = np.asarray(spec)
                    if arr.ndim == 1 and arr.size == N:       # per-stage scalars
                        W_block = float(arr[k]) * np.eye(n)
                    elif arr.ndim == 1 and arr.size == nb_w:  # per-window scalars
                        W_block = float(arr[wi]) * np.eye(n)
                    elif arr.shape == (n,):                   # per-axis
                        W_block = np.diag(arr.astype(float))
                    elif arr.shape == (n, n):                 # full matrix
                        W_block = arr.astype(float)
                    else:
                        W_block = float(arr.ravel()[0]) * np.eye(n)

                Wk[sl, sl] = W_block
            
            # --- Targets ---
            for sl, i in target_blocks:
                n = sl.stop - sl.start
                w_spec = param.weights["target"][i]
                if np.isscalar(w_spec):
                    W_block = (alpha[k] * float(w_spec)) * np.eye(n)
                else:
                    arr = np.asarray(w_spec)
                    if arr.shape == (n,):
                        W_block = np.diag(alpha[k] * arr.astype(float))
                    elif arr.shape == (n, n):
                        W_block = alpha[k] * arr.astype(float)
                    else:
                        W_block = (alpha[k] * float(arr.ravel()[0])) * np.eye(n)
                Wk[sl, sl] = W_block

            ocp_solver.cost_set(k, "W", Wk)
    
    
        
    def halfspace_from_segment(self,P0, P1, toward_point):
        """
        P0, P1, toward_point: (2,) or (3,) numpy arrays in world; we use XY.
        Returns (n, c) with n (2,), c scalar for inequality n^T p - c >= 0 (inside).
        """
        p0 = np.asarray(P0)[:2]; p1 = np.asarray(P1)[:2]; tp = np.asarray(toward_point)[:2]
        e = p1 - p0
        # outward normal candidates (perp of segment):
        n_cw  = np.array([ e[1], -e[0] ])
        n_ccw = -n_cw
        # choose the one pointing toward 'tp'
        n = n_cw
        if np.dot(n, tp - p0) < 0.0:
            n = n_ccw
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([1.0, 0.0])  # fallback
            n_norm = 1.0
        n = n / n_norm
        c = np.dot(n, p0)
        return n, c
 
        
        
            
    def tau_free_eval(self,xs,cu,param):
        tau=self.tau_free(xs,cu)
        return tau 
    
    def transport_effort(self,R,p,tau_A):
        #transport 6D effort from point A to frame B defined by its R,p pose
        p=p.reshape(3,)
       
        F=(R  @ tau_A[:3]).reshape(3,)
        M=(R @ tau_A[3:6]).reshape(3,)+ np.cross(p, F)
        tau_B=np.concatenate((F,M))
        return tau_B



    def calc(self,cx,cu,cf):
            ### WARNING REMEMBER TO HANDLE TEH NO FREEFLEYER CASE !!!!
        self.J=[]
        # if self.param.individual_joint_torques_cost==1:
        #     for j in range(6,self.nv):
        #         self.J.append(self.tau(cx,cu,cf)[j].T@self.tau(cx,cu,cf)[j]) # min torque #C1
        # else:
        for cost in  self.param.active_costs:
            if cost=="min_joint_torque":
                if self.param.groups_joint_torques["all"]==True:
                    self.J.append(self.tau(cx,cu,cf)[6:].T@self.tau(cx,cu,cf)[6:]) # min torque #C1
                else:
                    for group_name, joint_list in self.param.groups_joint_torques.items():
                        if group_name == "all":
                            continue  # skip the 'all' key
                        for joint_name in joint_list:
                            joint_id = self.model.getJointId(joint_name)
                            self.J.append(self.tau(cx,cu,cf)[joint_id].T@self.tau(cx,cu,cf)[joint_id])
                            
            if cost=="min_joint_vel":
                self.J.append(cx[self.nv+6:].T@cx[self.nv+6:]) # min joint velocity, ie NO freeflyer #C2
        
            
            if cost=="min_joint_acc":
                self.J.append(cu[self.nv+6:].T@cu[self.nv+6:]) # min joint accleration, ie NO freeflyer #C3
        #self.J.append( (self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1])/10 ) # min cartesian vel #C4
        #self.J.append( (self.dtau(cx,cu,cf).T@self.dtau(cx,cu,cf)) ) # min torque change #C5
        #self.J.append((self.energy(cx,cu,cf)) ) # min energy #C6
        #self.J.append((self.geodesic(q,dq,ddq))/4) # min geodesic #C7
        
        return self.J 
  
    
    
    def symbolic_log3(self,R):
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
    
    def approx_log6(self,id, M,cx,cu):
        # Update data symbolically inside the function
        cpin.forwardKinematics(self.cmodel, self.cdata, cx[:self.nq], cx[self.nq:],cu)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        cpin.updateFramePlacements(self.cmodel, self.cdata)                   # Update the frames placement to symbolic expressions in data
       
        tran_error = self.cdata.oMf[id].translation - M.translation
        #rot_error = casadi.diag(self.cdata.oMf[id].rotation.T @ M.rotation)-casadi.SX.ones(3)
        
        R_err = self.cdata.oMf[id].rotation.T @ M.rotation
        rot_error = self.symbolic_log3(R_err)  # returns a 3D vector in so(3)
        
        return(casadi.vertcat(tran_error, rot_error))      
    
   
 

    
    
    
    
    
    
    
    
    
    
    
    