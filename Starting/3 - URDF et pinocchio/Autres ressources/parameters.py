import numpy as np
from tools.irl_utils import low_pass_filter_data
import os 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# init: 0 100
# end -55 90

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)


subject="S03"
trial="Trial30"

poids= 66 # subject weights

param={"Mt":poids,
       "dt" : 0.01, # sampling time
       "cut_off": 10, # cut of frequency of the real filter
        "nb_samples": [], # number of samples of a single trajectory
        "qdi":np.deg2rad([0,90]),# initial desired joint position
        "dqdi":[0,0],# initial desired joint velocity
        "ddqdi":[0,0],# initial desired joint acceleration
        "nb_traj":1,# number of human trajectories to load for IRL
        "numero_traj":0,# if several traj choose a individual one among the five demonstrations
       "IK":0, # Solve IK for real human data
       "SIM":2,#0 solve IRL for a single (real) human optimal trajectory # 1  solve IRL for a single syntetic optimal trajectory# 2 generate optimal trajectory using DOC simulation  # 3 solve IRL for mulitiple human optimal trajectory 
       "variables_w":1, # use of variable weights 0: NO, 1: YES
        "nb_w":4,# number of windows of different weigths in the trajectory
        "MAXITER":2,
        "optimal_control":0,
        "type_P":0, # one of the 5 types of motion described in Berret 2012 P[:,0]=[0,90], P[:,1]=[-90,90], P[:,2]=[-120,120], P[:,3]=[-90,30], P[:,4]=[-80,140]
        "MAX":0,
       }
 
human_data={"q_opt":[0,0],
            "dq_opt":[0,0],
            "ddq_opt":[0,0]}

#segment lengths for syntetic data only
L=np.zeros(2)
L[0]=0.3369
L[1]=0.2327

if param["SIM"]==1:
    
    # Get optimal trajectory
    data=np.loadtxt('optimal_trajectories_all.csv', delimiter=',')

    q_opt=data[:,0:2]
    dq_opt=data[:,2:4]
    ddq_opt=data[:,4:6]
    
    param.update({"pxf": L[0]*np.cos(q_opt[-1,0])+L[1]*np.cos(q_opt[-1,0]+q_opt[-1,1]),#final desired Cartesian X position correspoding to 85% of L1+L2 !  
                 "nb_samples":len(q_opt),# number of samples of a single trajectory
                 "qdi":q_opt[0,:],})
    
elif param["SIM"]==2: # generate DOC optimal data
    param.update({"nb_samples":50,
                 "pxf": (L[0]+L[1])*0.85,
                 "pyf": 0})
    
elif param["SIM"]==0: 
    L=np.loadtxt(os.path.join(parent_directory,"data_collection/data_berret_2011/IK_results/"+subject+"_model.csv"), delimiter=',')

    # Get human optimal trajectory
    
    

    if param["MAX"]==1:
        data=np.load(os.path.join(parent_directory,"data_collection/data_maxime/all_elaborated_data_sb_Maxime2.npz"))

        # time_original = np.arange(0, int(data['len_q_opt'][0])*0.04, 0.04)
        # time_resampled = np.arange(0, time_original[-1], 0.01) 
        
        # q_opt_original=np.empty( (int(data['len_q_opt'][0]),2) )
        # q_opt_max=np.empty( (len(time_resampled),2) )
        # dq_opt_max=np.empty( (len(time_resampled)-1,2) )
        # ddq_opt_max=np.empty( (len(time_resampled)-2,2) )
        
        # for i in range(2):# 2 since 2 dof
        #     # Original data
            
        #     data_original = data['q_opt'][0:int(data['len_q_opt'][0]),i,0,0]

 
            

 
        #     interpolator = interp1d(time_original, data_original, kind='cubic')
        #     data_resampled = interpolator(time_resampled)

        #     q_opt_original[:,i]=data['q_opt'][0:int(data['len_q_opt'][0]),i,0,0]
        #     q_opt_max[:,i]=data_resampled
        #     dq_opt_max[:,i]=np.diff(q_opt_max[:,i])/param["dt"]
        #     ddq_opt_max[:,i]=np.diff(dq_opt_max[:,i])/param["dt"]
        
        q_opt=np.empty( (int(data['len_q_opt'][0]),2) )
        dq_opt=np.empty( (int(data['len_q_opt'][0])-1,2) )
        ddq_opt=np.empty( (int(data['len_q_opt'][0])-2,2) )
        
        q_opt_no_filt=np.empty( (int(data['len_q_opt'][0]),2) )
        
        for i in range(2):# 2 since 2 dof
           
            q_opt[:,i]=low_pass_filter_data(data['q_opt'][0:int(data['len_q_opt'][0]),i,0,0],param["dt"],param["cut_off"],5)
            q_opt_no_filt[:,i]=data['q_opt'][0:int(data['len_q_opt'][0]),i,0,0]#low_pass_filter_data(data['q_opt'][0:int(data['len_q_opt'][0]),i,0,0],param["dt"],param["cut_off"],1)

            dq_opt[:,i]=np.diff(q_opt[:,i])/param["dt"]
            ddq_opt[:,i]=np.diff(dq_opt[:,i])/param["dt"]
    
        # print("data_collection/data_berret_2011/IK_results/"+subject+"/"+trial+".csv")
        
        # file_path=os.path.join(parent_directory,"data_collection/data_berret_2011/IK_results/"+subject+"/"+trial+".csv")
        # data=np.loadtxt(file_path, delimiter=',')
        # data=data.T
   
        # q_opt=np.empty( (len(data),2) )
        # dq_opt=np.empty( (len(data)-1,2) )
        # ddq_opt=np.empty( (len(data)-2,2) )

    
        # for i in range(2):# 2 since 2 dof
        #     q_opt[:,i]=low_pass_filter_data(data[:,i],param["dt"],param["cut_off"],5)
        #     dq_opt[:,i]=np.diff(q_opt[:,i])/param["dt"]
        #     ddq_opt[:,i]=np.diff(dq_opt[:,i])/param["dt"]
    
    
    
    else: 
        
        print("data_collection/data_berret_2011/IK_results/"+subject+"/"+trial+".csv")
        
        file_path=os.path.join(parent_directory,"data_collection/data_berret_2011/IK_results/"+subject+"/"+trial+".csv")
        data=np.loadtxt(file_path, delimiter=',')
        data=data.T

        q_opt_no_filt   =np.empty( (len(data),2) )

        q_opt=np.empty( (len(data),2) )
        dq_opt=np.empty( (len(data)-1,2) )
        ddq_opt=np.empty( (len(data)-2,2) )

    
        for i in range(2):# 2 since 2 dof
            q_opt[:,i]=low_pass_filter_data(data[:,i],param["dt"],param["cut_off"],5)
            dq_opt[:,i]=np.diff(q_opt[:,i])/param["dt"]
            ddq_opt[:,i]=np.diff(dq_opt[:,i])/param["dt"]
    
    
    
    q_opt=q_opt[:-2,:]
    dq_opt=dq_opt[:-1,:]
    
    Px=0.3*np.cos( q_opt[:,0]) + 0.3*np.cos( q_opt[:,0]+ q_opt[:,1]) 
    dPx=np.abs(np.diff(Px)/param["dt"])
    ind_to_rm=np.argwhere(np.array(dPx)<8e-3)
    ind_max=np.argmax(dPx)
    
    
    
    ind_first=ind_to_rm[np.argmax(np.array(ind_to_rm)>ind_max)-1][0]

    ind_last=ind_to_rm[np.argmax(np.array(ind_to_rm)>ind_max)][0]
    
    #q_opt=q_opt[ind_first:ind_last,:]
    #dq_opt=dq_opt[ind_first:ind_last,:]
    #ddq_opt=ddq_opt[ind_first:ind_last,:]
    
    pxf=((L[0]*np.cos(q_opt[ind_last,0])+L[1]*np.cos(q_opt[ind_last,0]+q_opt[ind_last,1])))
    pyf=((L[0]*np.sin(q_opt[ind_last,0])+L[1]*np.sin(q_opt[ind_last,0]+q_opt[ind_last,1])))

    #pxf_max=((L[0]*np.cos(q_opt_max[-1,0])+L[1]*np.cos(q_opt_max[-1,0]+q_opt[-1,1])))
    #pyf_max=((L[0]*np.sin(q_opt_max[-1,0])+L[1]*np.sin(q_opt_max[-1,0]+q_opt[-1,1])))


    print("pxf:"+str(pxf)+"  pyf:"+str(pyf) )
    #print("pxf_max:"+str(pxf_max)+"  pyf_max:"+str(pyf_max) )
    
    q_opt=q_opt[ind_first:ind_last,:]
    dq_opt=dq_opt[ind_first:ind_last,:]
    ddq_opt=ddq_opt[ind_first:ind_last,:]

    
    param.update({"pxf": pxf,
                  "pyf": pyf,
                  "qdi":q_opt[0,:],
                  "dqf":0*dq_opt[-1,:],})#final desired Cartesian X position correspoding to 85% of L1+L2 !  



    #   # Create a figure and axis
    # fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # axs[0].plot(np.rad2deg(q_opt[:,0]), label='BB')
    # axs[0].plot(np.rad2deg(q_opt_no_filt[:,0]), label='max',color='black')
    # #axs[0].plot(np.rad2deg(q_opt_original[:,0]), label='max',color='blue')
    # axs[1].plot(np.rad2deg(q_opt[:,1]), label='BB',color='red')
    # axs[1].plot(np.rad2deg(q_opt_no_filt[:,1]), label='max',color='black')

    # #axs[1].plot(np.rad2deg(q_opt_max[:,1]), label='max',color='black')
    # #axs[1].plot(np.rad2deg(q_opt_original[:,1]), label='max',color='blue')

    # # axs[0].plot(time_resampled,np.rad2deg(q_opt[:,0]), label='px')
    # # axs[0].plot(time_original,np.rad2deg(q_opt_original[:,0]), label='px',color='black')

    # # axs[1].plot(time_resampled,np.rad2deg(q_opt[:,1]), label='dpx',color='red')
    # # axs[1].plot(time_original,np.rad2deg(q_opt_original[:,1]), label='dpx',color='black')
   
    # axs[0].set_title('q1')
    # axs[1].set_title('q2')

    # axs[0].legend()
    # axs[1].legend()

    # plt.tight_layout()
    # plt.show()
    
    # 
    # # Create a figure and axis
    # fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    # axs[0].plot(Px, label='px')
    # axs[0].plot(ind_first,Px[ind_first], 'x')
    # axs[0].plot(ind_last,Px[ind_last], 'x')
    # axs[1].plot(dPx, label='dpx',color='red')
    # axs[1].plot(ind_first,dPx[ind_first], 'x')
    # axs[1].plot(ind_last,dPx[ind_last], 'x')

    # axs[0].set_title('px')
    # axs[1].set_title('dpx')

    # axs[0].legend()
    # axs[1].legend()

    # plt.tight_layout()
    # #plt.show()
  
    
elif param["SIM"]==3:
    
  
    
    L=np.loadtxt(os.path.join(parent_directory,"data_collection/data_berret_2011/IK_results/"+subject+"_model.csv"), delimiter=',')

    data=np.load(os.path.join(parent_directory,"data_collection/data_berret_2011/all_elaborated_data_sb_"+subject+".npz"))
    
    
    if param["MAX"]==1:
        subject="Fadi"
        
        data=np.load(os.path.join(parent_directory,"data_collection/data_Vincent/"+str(subject)+"/trials/all_elaborated_data_sb_"+str(subject)+".npz"))

        #data=np.load(os.path.join(parent_directory,"data_collection/data_Vincent/"+str(subject)+"/training/raw/all_elaborated_data_sb_"+str(subject)+".npz"))
        L=np.loadtxt(os.path.join(parent_directory,"data_collection/data_Vincent/"+str(subject)+"/"+str(subject)+"_model.csv"), delimiter=',')

       # print(data.files)
        # L=np.loadtxt(os.path.join(parent_directory,"data_collection/data_maxime/Maxime3_model.csv"), delimiter=',')
        # data=np.load(os.path.join(parent_directory,"data_collection/data_maxime/all_elaborated_data_sb_Maxime3.npz"))
        # w0=data['weigths']
    
    q_opt=np.empty( (200,2,param["nb_traj"]) ) 
    dq_opt=np.empty( (200,2,param["nb_traj"]) ) 
    ddq_opt=np.empty( (200,2,param["nb_traj"]) ) 
    nb_samples_list=[]
    pxf_list=[]
    qdi_list=[]
    qdf_list=[]
    dqdi_list=[]
    ddqdi_list=[]

    for num in range(param["nb_traj"]):
            nb_samples=int(data['len_q_opt'][num,param['type_P']])
            q_opt[0:nb_samples,:,num]=data['q_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#q_filt    
            dq_opt[0:nb_samples,:,num]=data['dq_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#dq_filt
            ddq_opt[0:nb_samples,:,num]=data['ddq_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#ddq_filt
    
            
            
            print(param["nb_traj"])
            
            if param["nb_traj"]==1:
                num=param["numero_traj"] #collect another trajectory
                start=0
                stop=0
               
                #nb_samples=int(data['len_q_opt'][num])
                nb_samples=int(data['len_q_opt'][num,param['type_P']])
                
                q_opt[0:nb_samples-start-stop,:,0]=data['q_opt'][0+start:-stop+int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#q_filt    
                dq_opt[0:nb_samples-start-stop,:,0]=data['dq_opt'][0+start:-stop+int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#dq_filt
                ddq_opt[0:nb_samples-start-stop,:,0]=data['ddq_opt'][0+start:-stop+int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#ddq_filt
                nb_samples=nb_samples-start-stop
                
                pxf_list.append((L[0]*np.cos(q_opt[nb_samples-1,0,0])+L[1]*np.cos(q_opt[nb_samples-1,0,0]+q_opt[nb_samples-1,1,0])))
                print(pxf_list)
                num=0# set back index to zero as we have only one traj
            else:
                #nb_samples=int(data['len_q_opt'][num])
                nb_samples=int(data['len_q_opt'][num,param['type_P']])
                q_opt[0:nb_samples,:,num]=data['q_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#q_filt    
                dq_opt[0:nb_samples,:,num]=data['dq_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#dq_filt
                ddq_opt[0:nb_samples,:,num]=data['ddq_opt'][0:int(data['len_q_opt'][num,param['type_P']]),:,num,param['type_P']]#ddq_filt
                
                pxf_list.append((L[0]*np.cos(q_opt[nb_samples-1,0,num])+L[1]*np.cos(q_opt[nb_samples-1,0,num]+q_opt[nb_samples-1,1,num])))
            
            nb_samples_list.append(nb_samples)

            qdi_list.append(q_opt[0,:,num])
            qdf_list.append(q_opt[nb_samples-1,:,num])
            dqdi_list.append(dq_opt[0,:,num])
            ddqdi_list.append(ddq_opt[0,:,num])
      
 
    human_data.update({"q_opt": q_opt, 
                 "dq_opt": dq_opt, 
                 "ddq_opt": ddq_opt,
                # "w0":w0,
                 })

 
    print(pxf_list[0])
        
    param.update({"pxf_list": pxf_list,#final desired Cartesian X position correspoding to 85% of L1+L2 !  
                 "pxf":pxf_list[0],# set pxf to first desired value just for visual representation
                 "nb_samples_list":nb_samples_list,
                 "qdi_list":qdi_list,
                 "qdf_list":qdf_list,
                 "dqdi_list":dqdi_list,
                 "ddqdi_list":ddqdi_list})


  

param.update({"nb_joints":2, # number of joints in the model
              "nb_cost":7, # number of cost functions
    "nb_nopt":50, # number of random non optimal trajectories
    "noise_std":np.deg2rad(20), # noise standard deviation in radian
    "q_min":np.deg2rad([-150,10]),
    "q_max":np.deg2rad([90,170]),
    "L":[L[0],L[1]],#[0.186 *  param["Lt"], (0.146 + 0.108) *  param["Lt"]],
    "M":[0.028*param["Mt"],0.022*param["Mt"]],
    "I":[0.028*param["Mt"] * (L[0] * 0.322)**2,0.022*param["Mt"]* (L[1] * 0.468)**2],
    "K":[0.436*L[0], 0.682*L[1]],#[0.436*0.186 * param["Lt"], 0.682*(0.146 + 0.108) * param["Lt"]],
    "g":-9.81, # gravity in m.s-2
    "tau_max":[92,77], # in N.m
    "dq_lim":[-60.14,60.14],# in rad
    #"w0":[0.1,0.05,0.1,0.1,0.1,0.1,0.01]/np.linalg.norm([0.1,0.05,0.1,0.1,0.1,0.1,0.01]), # initial or DOC weights
    #"w0":[0.01,0.05,0.01,0.001,0.1,0.1,0.01]/np.linalg.norm([0.01,0.05,0.01,0.001,0.1,0.1,0.01]), # initial or DOC weights
   
    })  
  
 