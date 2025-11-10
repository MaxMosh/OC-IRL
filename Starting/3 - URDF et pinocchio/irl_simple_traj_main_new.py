import casadi
import numpy as np
import time
from  tools.create_model_scene import create_double_pendulum_model_and_scene

from tools.irl_utils import CostsModelDoublePendulum, DocDoublePendulum, rmse_function, generate_non_optimal_traj, evaluate_costs, IRL_step_calculation
 
import matplotlib.pyplot as plt; 

## Load optimal (observed) trajectories denoted with _opt to be reproduced
# Load also the param dictonary containing simulation parameters
from tools.parameters import param, q_opt, dq_opt, ddq_opt
param["nb_samples"]=len(q_opt)

### Load and display pendulum model using meshcat 
# robot,viz,viz_hum, param = create_double_pendulum_model_and_scene(param)
robot, param = create_double_pendulum_model_and_scene(param)
model = robot.model

## 1. 
## Generate non-optimal trajectories 
print(q_opt.shape)
input()
q_nopt, dq_nopt, ddq_nopt=generate_non_optimal_traj(q_opt,dq_opt,ddq_opt, param)
#np.savez("q_dq_ddq_nopt.npz", arr1=q_nopt,arr2=dq_nopt,arr3=ddq_nopt )
#loaded_nopt=np.load('q_dq_ddq_nopt.npz')
#q_nopt=loaded_nopt['arr1'] 
#dq_nopt=loaded_nopt['arr2'] 
#ddq_nopt=loaded_nopt['arr3'] 

# plt.figure(1)
# plt.subplot(611)
# plt.plot(np.rad2deg( q_nopt[:,0,:] )   ,'k' )
# plt.plot(np.rad2deg( q_opt[:,0] )   ,'r' )
# plt.ylabel("q1 [deg]")

# plt.subplot(612)
# plt.plot(np.rad2deg( q_nopt[:,1,:] )   ,'k' )
# plt.plot(np.rad2deg( q_opt[:,1] )   ,'g' )
# plt.ylabel("q2 [deg]")

# plt.subplot(613)
# plt.plot(np.rad2deg( dq_nopt[:,0,:] )   ,'k' )
# plt.plot(np.rad2deg( dq_opt[:,0] )   ,'r' )
# plt.ylabel("dq1 [deg.s^-1]")

# plt.subplot(614)
# plt.plot(np.rad2deg( dq_nopt[:,1,:] )   ,'k' )
# plt.plot(np.rad2deg( dq_opt[:,1] )   ,'g' )
# plt.ylabel("dq2 [deg.s^-1]")

# plt.subplot(615)
# plt.plot(np.rad2deg( ddq_nopt[:,0,:] )   ,'k' )
# plt.plot(np.rad2deg( ddq_opt[:,0] )   ,'r' )
# plt.ylabel("ddq1 [deg.s^-2]")

# plt.subplot(616)
# plt.plot(np.rad2deg( ddq_nopt[:,1,:] )   ,'k' )
# plt.plot(np.rad2deg( ddq_opt[:,1] )   ,'g' )
# plt.ylabel("ddq2 [deg.s^-2]")
#plt.show()



## 2. 
## Evaluate the set of cost functions
## 2.1 Using optimal (observed) trajectories

runningModels = CostsModelDoublePendulum(model,param) 
## Evaluate the optimal set of cost functions, ie using human demonstration    
   
J_opt=evaluate_costs(q_opt,dq_opt,ddq_opt,runningModels,param)  
        

## 2.1 Using non-optimal trajectories
  
#J_nopt=evaluate_costs(q_nopt,dq_nopt,ddq_nopt,runningModels,param)  

## Evaluate the non optimal set of cost functions    
if param["variables_w"]:
    J_nopt=np.empty((param["nb_cost"],param["nb_nopt"], param["nb_w"] )) 
    
    for i in range(param["nb_nopt"]):
        J=evaluate_costs(q_nopt[:,:,i],dq_nopt[:,:,i],ddq_nopt[:,:,i],runningModels,param)   
        J_nopt[:,i,:]=J         
else:
    J_nopt=np.empty( (param["nb_cost"], param["nb_nopt"]) )      
    for i in range(param["nb_nopt"]):
        J=evaluate_costs(q_nopt[:,i],dq_nopt[:,i],ddq_nopt[:,i],runningModels,param)     
        J_nopt[:,i,:]=J


##3.
## Solve IRL Problem 
 
# IPopts parameters 
p_opts = {'ipopt.print_level':1, 'print_time': 0, 'ipopt.sb': 'yes'} 
s_opts = {"max_iter": 10000}

J_opt_prev=[]
x_opt=[]

if param["variables_w"]==1:
    w_norm=np.empty(( param["nb_cost"], param["nb_w"] ))
    for i in range( param["nb_w"] ):
        w_norm[:,i]=param["w0"] # initial values of the weigths

else:
    w_norm=param["w0"] # initial values of the weigths


val_rmse_min=10
val_rmse=100
w_norm0=w_norm
no_add_J_opt_prev=0

iter=0
val_rmse_prec=100

val_rmse_nopt=np.zeros(param["nb_nopt"])
for i in range(param["nb_nopt"]):
    val_rmse_nopt[i]=rmse_function( np.array([q_opt[:,0],q_opt[:,1]]).reshape(2*(param["nb_samples"])), np.array([[q_nopt[:,0,i],q_nopt[:,1,i]]]).reshape(2*(param["nb_samples"]) ) )
 
lambda_bretl=10

while (np.rad2deg(val_rmse)>1 and iter <param["MAXITER"]): # continue to run the algo if RMSE(q_opt,q_est)>2deg or MAXITER
#for iter in range(20):
    
    
   # add the previous iteration "optimal trajectory" to the set of non-optimal ones
    if iter>0 and no_add_J_opt_prev==0:
        
        if param["variables_w"]:
            J_nopt=np.concatenate((J_nopt, J_opt_prev.reshape(param["nb_cost"],1, param["nb_w"] )), axis=1)
        else:
            J_nopt=np.hstack( [J_nopt,J_opt_prev.reshape(param["nb_cost"],1)] ) 
       
        q_nopt=np.concatenate((q_nopt, q_opt_s.reshape((param["nb_samples"],2,1))), axis=2)   
        param["nb_nopt"]=param["nb_nopt"]+1
    no_add_J_opt_prev=0
    
    val_rmse_nopt=np.zeros(param["nb_nopt"])
    for i in range(param["nb_nopt"]):
        val_rmse_nopt[i]=rmse_function( np.array([q_opt[:,0],q_opt[:,1]]).reshape(2*(param["nb_samples"])), np.array([[q_nopt[:,0,i],q_nopt[:,1,i]]]).reshape(2*(param["nb_samples"]) ) )
    
    
 
    w_norm=IRL_step_calculation(w_norm0,val_rmse_nopt,lambda_bretl,J_opt,J_nopt,param) 

    
    doc_problem = DocDoublePendulum(model,w_norm,runningModels, param) 

    q_opt_s, dq_opt_s, ddq_opt_s=doc_problem.solve_doc(model, [], param)
    
    
    ## 3.3
    ## Analyse DOC generated new trajectoy and calcualte RMSE     
    val_rmse=rmse_function( np.array([q_opt[:,0],q_opt[:,1]]).reshape(2*(param["nb_samples"])), np.array([[q_opt_s[:,0],q_opt_s[:,1]]]).reshape(2*(param["nb_samples"]) ) )
    
    
    
    #print( np.rad2deg(val_rmse_nopt) )
    
    if val_rmse<val_rmse_min:
        val_rmse_min=val_rmse
        q_opt_opt=q_opt_s
        w_opt=x_opt
        
    print("iter ",iter ," rmse ",np.rad2deg(val_rmse))
    print("Best rmse ",np.rad2deg(val_rmse_min))
    
    # print(np.abs(np.rad2deg(val_rmse-val_rmse_prec)))
    treshold=0.1
     
    # if  (val_rmse-val_rmse_prec)>=0: # remove teh previously added nopt
    #     print("IRL stuck remove prev traj ")

    #     ind_rmse=np.where(val_rmse_prec==val_rmse_nopt)[0]# 0 for access to the first occurrence
    #     print(ind_rmse)
    #     val_rmse_nopt=np.delete(val_rmse_nopt,ind_rmse)
    #     J_nopt=np.delete(J_nopt,ind_rmse, axis=1)
    #     q_nopt=np.delete(q_nopt,ind_rmse, axis=2)#=np.empty( (param["nb_samples"],param["nb_joint"],param["nb_nopt"]) )
    #     param["nb_nopt"]=param["nb_nopt"]-1
    
    if np.abs(np.rad2deg(val_rmse-val_rmse_prec))<=treshold:# or  val_rmse>=np.min( val_rmse_nopt ):
        lambda_bretl=lambda_bretl/(1+0.05)
        if lambda_bretl<0.05:
            lambda_bretl=0.05
        print("IRL stuck update heat to "+str(1/lambda_bretl))
        
        
        
        # ind_rmse_max=np.argmax(val_rmse_nopt)
        # val_rmse_nopt=np.delete(val_rmse_nopt,ind_rmse_max)
        # J_nopt=np.delete(J_nopt,ind_rmse_max, axis=1)
        # q_nopt=np.delete(q_nopt,ind_rmse_max, axis=2)#=np.empty( (param["nb_samples"],param["nb_joint"],param["nb_nopt"]) )
        # param["nb_nopt"]=param["nb_nopt"]-1
        
        
        # A=finite_difference_matrix(param["nb_samples"], 1)
        # A0=np.zeros(param["nb_samples"])
        # A0[0]=1
        # A=np.vstack((A0,A))
        # A0[0]=0
        # A0[-1]=1

        # A=np.vstack((A,A0))
        # R=np.matmul(A.T,A)
        # Rinv=np.linalg.inv(R)

        # for i in range(param["nb_samples"]):
        #     Rinv[:,i]=Rinv[:,i]*1/(2*param["nb_samples"])
        
        # covariance_matrix =  Rinv  
        
        # # Perform Cholesky decomposition to correlate the samples
        # L = np.linalg.cholesky(covariance_matrix.T)

        # for j in range(param["nb_joints"]):
                
        #     q_opt_s[:,j]=q_opt_s[:,j]+ np.random.normal(0, 2*param["noise_std"], param["nb_samples"]) @ L.T # arbirtray choose parameters for noise 0.5 for position, 1 for velocity and 50 for acc
        #     dq_opt_s[:,j]=dq_opt_s[:,j]+ np.random.normal(0, 5*param["noise_std"], param["nb_samples"]) @ L.T
        #     ddq_opt_s[:,j]=ddq_opt_s[:,j]+ np.random.normal(0, 10*param["noise_std"], param["nb_samples"]) @ L.T 
    
    if param["variables_w"]==1:
        J_opt_prev=np.empty(( param["nb_cost"], param["nb_w"] ))
    else:
        J_opt_prev=np.empty(param["nb_cost"])



    t_prev=-1
    J_opt_prev_temp=np.zeros(param["nb_cost"])
    k=0
    for t in range(param["nb_samples"]):
        runningModels.calc(q_opt_s[t,:],dq_opt_s[t,:],ddq_opt_s[t,:])
        if param["variables_w"]==1:
           
            J_opt_prev[:,k]=J_opt_prev_temp
            if int(param["nb_samples"]/param["nb_w"])==(t-t_prev):
                t_prev=t
                if k<param["nb_w"]-1:
                    k=k+1
                    J_opt_temp=np.zeros(param["nb_cost"])

            J_opt_prev_temp=np.array(runningModels.J ).reshape(param["nb_cost"])
        else:
            J_opt_prev+=np.array(runningModels.J ).reshape(param["nb_cost"])
    
        
        
        
        
    
        
        
        
    iter+=1
    val_rmse_prec=val_rmse
        
    plt.figure(3)
    plt.clf()
    plt.subplot(311)
    plt.plot(np.rad2deg( q_nopt[:,0,:] )   ,'k' )
    plt.plot(np.rad2deg( q_opt[:,0] )   ,'r' )
    plt.plot(np.rad2deg( q_opt_s[:,0] )   ,'--b' )
    
    plt.ylabel("q1 [deg]")

    plt.subplot(312)
    plt.plot(np.rad2deg( q_nopt[:,1,:] )   ,'k' )
    plt.plot(np.rad2deg( q_opt[:,1] )   ,'g' )
    plt.plot(np.rad2deg( q_opt_s[:,1] )   ,'--b' )

    plt.ylabel("q2 [deg]")
    plt.subplot(313)
    plt.plot(np.rad2deg(val_rmse_nopt))
    plt.plot(np.argmin(val_rmse_nopt),np.rad2deg( val_rmse_nopt[np.argmin(val_rmse_nopt)]),'xr')
    
    # plt.subplot(613)
    # plt.plot(np.rad2deg( dq_nopt[:,0,:] )   ,'k' )
    # plt.plot(np.rad2deg( dq_opt[:,0] )   ,'r' )
    # plt.plot(np.rad2deg( dq_opt_s[:,0] )   ,'--r' )

    # plt.ylabel("dq1 [deg.s^-1]")

    # plt.subplot(614)
    # plt.plot(np.rad2deg( dq_nopt[:,1,:] )   ,'k' )
    # plt.plot(np.rad2deg( dq_opt[:,1] )   ,'g' )
    # plt.plot(np.rad2deg( dq_opt_s[:,1] )   ,'--g' )

    # plt.ylabel("dq2 [deg.s^-1]")

    # plt.subplot(615)
    # plt.plot(np.rad2deg( ddq_nopt[:,0,:] )   ,'k' )
    # plt.plot(np.rad2deg( ddq_opt[:,0] )   ,'r' )
    # plt.plot(np.rad2deg( ddq_opt_s[:,0] )   ,'--r' )

    # plt.ylabel("ddq1 [deg.s^-2]")

    # plt.subplot(616)
    # plt.plot(np.rad2deg( ddq_nopt[:,1,:] )   ,'k' )
    # plt.plot(np.rad2deg( ddq_opt[:,1] )   ,'g' )
    # plt.plot(np.rad2deg( ddq_opt_s[:,1] )   ,'--r' )

    # plt.ylabel("ddq2 [deg.s^-2]")
    plt.pause(0.05)   
    
# Finalize the plot
plt.ioff()  # Disable interactive mode
plt.show()  # Keep the plot open
 
print("Best rmse %f",np.rad2deg(val_rmse_min))
rmse_q1=rmse_function( q_opt[:,0], q_opt_opt[:,0]) 
rmse_q2=rmse_function( q_opt[:,1], q_opt_opt[:,1])

#print('Press Enter to start animation')
#x=input()

# for i in range(param["nb_samples"]):
#     viz.display(q_opt[i,:])
#     viz_hum.display(q_opt_opt[i,:])
#     time.sleep(0.01)


#plot_identified_weigths(w_norm,np.mean(np.rad2deg(val_rmse_min)), param)
#plt.show()

    
plt.figure(2)
plt.subplot(211)
#plt.plot(np.rad2deg( q_nopt[:,0,:] )   ,'k' )
plt.plot(np.rad2deg( q_opt[:,0] )   ,'r' )
#plt.plot(np.rad2deg( q_opt_s[:,0] )   ,'b' )
plt.plot(np.rad2deg( q_opt_opt[:,0] )   ,'g' )
plt.legend(["hum","est"])
plt.ylabel("q1 [deg]")

plt.subplot(212)
#plt.plot(np.rad2deg( q_nopt[:,1,:] )   ,'k' )
plt.plot(np.rad2deg( q_opt[:,1] )   ,'r' )
#plt.plot(np.rad2deg( q_opt_s[:,1] )   ,'b' )
plt.plot(np.rad2deg( q_opt_opt[:,1] )   ,'g' )

plt.ylabel("q2 [deg]")
plt.show()
    
# ### SAVE
# #np.save(open(WARMSTART,'wb'),[xs_sol,us_sol])
# np.save('optimal_recovery_traj.csv',[q_opt,q_opt_opt])
# np.save('optimal_recovery_weigths.csv',[w_opt])

#print("tip PX opt: %f", runningModels.tip(q_opt[-1,:])[0])
print("tip PY est: %f ",runningModels.tip(q_opt[-1,:])[1])
#print("tip PX opt: %f", runningModels.tip(q_opt[-1,:])[0])
print("tip PY est: %f ",runningModels.tip(q_opt_opt[-1,:])[1])

print(param["w0"])
print(w_opt[:,0])


