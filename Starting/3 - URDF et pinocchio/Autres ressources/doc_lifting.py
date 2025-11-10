import pandas as pd 
import numpy as np 
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
from utils.model_utils import setup_lifting_model_and_viz, model_scaling_from_dict
from utils.irl_utils import CostsModel5Dofs, Doc5DofsLifting
from utils.vizutils import place, display_force, display_collisions, color_segment
from utils.irl_utils import low_pass_filter_data, evaluate_costs
from utils.data_utils import format_numpy_to_list_of_dict, calculate_distance_collisions
import time
import matplotlib.pyplot as plt

subject = 'jony'

model, pin_data, viz, collision_model, visual_model = setup_lifting_model_and_viz(subject)

# Adds markers viz objects 
ori_base = np.array([[-1,0,0],[0,0,1],[0,1,0]])
marker_names = ['BACK', 'FARDL', 'FARDR', 'FARL', 'FARR', 'FARUR', 'NEADR', 'NEAR', 'RANK', 'RELB', 'RGTR', 'RHEE', 'RKNE', 'RSHOE', 'RTOE', 'RWRI']
nb_markers = len(marker_names)

mks_data = pd.read_csv('data_collection/lifting/normallifting1/'+subject+'/'+subject+'_normallifting1.csv',skiprows=6)
mks_data_cleaned = mks_data.loc[:, ~mks_data.columns.str.contains('^Unnamed')].to_numpy()
if mks_data_cleaned.shape[1] != nb_markers*3+2:
    print('Error: the number of markers is not correct for clement, reshaping the data')
    print('Original shape: ', mks_data_cleaned.shape)
    mks_data_cleaned = mks_data_cleaned[:,:nb_markers*3+2]

dict_mks = format_numpy_to_list_of_dict(mks_data_cleaned, marker_names)

segment_lengths = {}
segment_lengths['lowerleg']= np.linalg.norm(dict_mks[0]['RKNE']-dict_mks[0]['RANK'])
segment_lengths['upperleg']= np.linalg.norm(dict_mks[0]['RGTR']-dict_mks[0]['RKNE'])  
segment_lengths['trunk']= np.linalg.norm(dict_mks[0]['RSHOE']-dict_mks[0]['RGTR'])
segment_lengths['upperarm']= np.linalg.norm(dict_mks[0]['RELB']-dict_mks[0]['RSHOE'])
segment_lengths['lowerarm']= np.linalg.norm(dict_mks[0]['RWRI']-dict_mks[0]['RELB'])
print("Segment lengths for subject : ", segment_lengths)
model, pin_data = model_scaling_from_dict(model, pin_data, segment_lengths)

# turn background to white 
import gepetto as gep
viz.viewer.gui.setBackgroundColor1("python-pinocchio", gep.color.Color.white)
viz.viewer.gui.setBackgroundColor2("python-pinocchio", gep.color.Color.white)
viz.viewer.gui.addLight("light", "python-pinocchio", 360, gep.color.Color.white)

# Adds a second visualizer and color it in green
viz2 = GepettoVisualizer(model, collision_model,visual_model)
viz2.initViewer(viz.viewer)
viz2.loadViewerModel(rootNodeName = "pinocchio2")

nodes = viz2.viewer.gui.getNodeList()
filtered_nodes = [node for node in nodes if "world/pinocchio2/visuals" in node]
std = None
color_segment(viz2, std, filtered_nodes, color=[0,1,0,1])

param={}
param['nb_joints']=model.nq
param['cut_off']=10
dt = param['dt']=1/20
param["q_min"]= np.array([np.deg2rad(30), 0, np.deg2rad(-150), np.deg2rad(-180), np.deg2rad(0)])
param["q_max"]= np.array([np.deg2rad(110),np.deg2rad(150),np.deg2rad(30), np.deg2rad(0), np.deg2rad(110)])
param["dq_lim"] = [-6.14,6.14]
param["tau_lim"] =  [200*2,200*2,200*2,200*2,200*2] # full five dof limits
param["cop_lim"] = [-0.05,0.30]
param["nb_w"]= 3
param['noise_std'] = np.deg2rad(2)
param["MAXITER"]=8
param["repeat"]=5
param["nb_cost"] = 10 # standard is 7
param["variables_w"]=1
param['collision_epsilon'] = 0.05
param['nb_traj'] = 1

param["w0"] = np.random.rand(param["nb_cost"])
# param["w0"] = np.array([1, 1, 1, 1,
#         1, 1, 1, 1]) 
# param["w0"] = param["w0"]/param["w0"].sum()

# With qf
# In [3]: param['w0']
# Out[3]: 
# array([0.13269917, 0.0279143 , 0.13817647, 0.19726762, 0.18637425,
#        0.13391261, 0.18365558])

# with pzf only but without COP
# In [2]: param['w0']
# Out[2]: 
# array([0.18992925, 0.1501619 , 0.03636082, 0.15197684, 0.00623284,
#        0.16246433, 0.30287401])


filip_data = pd.read_csv('data_collection/lifting/normallifting1/'+subject+'/'+subject+'_q.csv',delimiter=',')

# Reshape filip_data to get one lifting cycle 
# To be read from filip files, gives cycle of lifting and descending tasks
if subject == 'clement':
    # For clement, the lifting cycle is from 0 to 1218
    first_index = 0
    last_index = 1218
elif subject == 'jony':
    # For jony, the lifting cycle is from 231 to 1770
    first_index = 231
    last_index = 1770

filip_data = filip_data.iloc[first_index:last_index, :]

data = np.zeros((int(len(filip_data)/5),param['nb_joints']))

i_mod_5 = 0
for index, row in filip_data.iterrows():
    if index%5==0 and i_mod_5!=int(len(filip_data)/5):
        data[i_mod_5,:] = np.array([row['q0'], row['q1'], row['q2'], row['q3'], row['q4']])
        i_mod_5+=1

q_opt=np.empty( (len(data),param['nb_joints']) )
dq_opt=np.empty( (len(data)-1,param['nb_joints']) )
ddq_opt=np.empty( (len(data)-2,param['nb_joints']) )

for i in range(param['nb_joints']): 
    q_opt[:,i]=low_pass_filter_data(data[:,i],param["dt"],param["cut_off"],5)
    dq_opt[:,i]=np.diff(q_opt[:,i])/param["dt"]
    ddq_opt[:,i]=np.diff(dq_opt[:,i])/param["dt"]

q_opt=q_opt[:-2,:]
dq_opt=dq_opt[:-1,:]

# fig, ax = plt.subplots(5,1)
# ax[0].plot(q_opt[:,0])
# ax[1].plot(q_opt[:,1])
# ax[2].plot(q_opt[:,2])
# ax[3].plot(q_opt[:,3])
# ax[4].plot(q_opt[:,4])
# plt.show()

# for ii in range(q_opt.shape[0]):
#     print('ii = ',ii)
#     q_ii = pin.neutral(model)
#     q_ii[:] = q_opt[ii,:]
#     viz.display(q_ii)
#     input()

# Takes only the phase of lifting 
if subject == 'clement':
    # For clement 
    start_lifting = 70
    end_lifting= 117
elif subject == 'jony':
    # For jony 
    start_lifting = 45
    end_lifting= 110

q_opt = q_opt[start_lifting:end_lifting,:]
dq_opt = dq_opt[start_lifting:end_lifting,:]
ddq_opt = ddq_opt[start_lifting:end_lifting,:]

# Adds COP, COM display 
R0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
viz.viewer.gui.addXYZaxis("world/R0", [1, 0, 0, 1], 0.01, 0.1)
place(viz, "world/R0", R0)
viz.viewer.gui.addSphere("world/COP", 0.01, [1, 1, 0, 1])
viz.viewer.gui.addSphere("world/COM", 0.01, [1, 0, 0, 1])

if param["variables_w"]==1:
    w_norm=np.empty(( param["nb_cost"], param["nb_w"] ))
    for i in range( param["nb_w"] ):
        w_norm[:,i]=param["w0"] # initial values of the weigths
else:
    w_norm=param["w0"] # initial values of the weigths

# w_norm = np.array([[1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#        [0.0, 0.0, 0.0, 0.0, 0, 0],
#        [0.0, 0.0, 0, 0., 0., 0.],
#        [0.0, 0.0, 0, 0., 0., 0.],
#        [0.0, 0.0, 0, 0., 0., 0.],
#        [0.0, 0.0, 0, 0, 0, 0],
#        [0.0, 0.1, 0.1, 0.1, 0.1, 0.0]])

w_norm = np.array([[9.99028407e-06, 1.85768948e-02, 9.99015801e-06],
       [1.00024473e-05, 9.99541365e-06, 9.99371127e-06],
       [1.87381748e+00, 1.00059737e-05, 9.99140731e-06],
       [9.99016181e-06, 9.99014443e-06, 9.99060343e-06],
       [9.99022939e-06, 9.99007691e-06, 9.99025139e-06],
       [9.99026840e-06, 9.99016812e-06, 9.99195722e-06],
       [9.99037963e-06, 9.99189449e-06, 9.99142836e-06],
       [5.38082963e-05, 1.74282782e-03, 1.78722841e-03],
       [9.99060060e-06, 9.99104508e-06, 9.99872712e-06],
       [9.99028561e-06, 9.99023835e-06, 9.99013516e-06]])


print('w_norm = ', w_norm)
# input()
param["nb_samples"]= len(q_opt)

pin.framesForwardKinematics(model, pin_data, q_opt[-1,:])
param['rf'] = pin_data.oMf[model.getFrameId('box')].rotation
param['angle_y'] = np.sum(q_opt[-1])
param['pf'] = pin_data.oMf[model.getFrameId('box')].translation

runningModel = CostsModel5Dofs(model, segment_lengths, param['pf'], dt)

## Evaluate the optimal set of cost functions, ie using human demonstration    
J_opt= evaluate_costs(q_opt,dq_opt,ddq_opt,runningModel,param)      

print('J_opt = ', J_opt)
# input()

# ### Check the collision circles
# circles = runningModel.circles
# q_test = pin.neutral(model)
# q_test[:] = q_opt[-1,:]

# viz.display(q_test)
# display_collisions(viz, circles, q_test)
    
doc_problem = Doc5DofsLifting(model, w_norm, runningModel, param)
q_opt_s, dq_opt_s, ddq_opt_s, _ = doc_problem.solve_doc(q_opt[0,:], dq_opt[0,:], ddq_opt[0,:])

COP_list = []
COP_opt_list = []
q_list = []
dq_list = []
tau_list = []
tau_opt = np.zeros((q_opt.shape[0],param['nb_joints']))

# input()

for i in range(param["nb_samples"]):
    q_opt_ii = q_opt[i,:]
    dq_opt_ii = dq_opt[i,:]
    ddq_opt_ii = ddq_opt[i,:]
    viz2.display(q_opt_ii)

    COP_opt = np.array(runningModel.cop(q_opt_ii,dq_opt_ii,ddq_opt_ii))
    COP_opt_list.append(COP_opt)
    tau_opt[i,:] = runningModel.tau(q_opt_ii,dq_opt_ii,ddq_opt_ii)[0]

    q_ii = q_opt_s[i,:]
    q_list.append(q_ii)
    dq_ii = dq_opt_s[i,:]
    dq_list.append(dq_ii)
    ddq_ii = ddq_opt_s[i,:]
    viz.display(q_ii)
    display_collisions(viz, runningModel.circles, q_ii)

    collisions_dict = calculate_distance_collisions(runningModel.circles, q_ii)
    print('Collisions distances: ', collisions_dict)

    tau_list.append(runningModel.tau(q_ii,dq_ii,ddq_ii))
    
    #COP
    phi_ii = pin.Force(np.array(runningModel.phi_ankle(q_ii,dq_ii,ddq_ii)))
    print('Phi is ', phi_ii)
    COP = np.array(runningModel.cop(q_ii,dq_ii,ddq_ii))
    COP_list.append(COP)
    print('Cop is ', COP)
    COP_p = pin.SE3(np.eye(3), np.array([COP[0][0], COP[1][0], 0]))
    place(viz, "world/COP", COP_p)
    display_force(viz, phi_ii, COP_p)

    # COM 
    COM = np.array(runningModel.com(q_ii,dq_ii,ddq_ii))
    print('Com is ', COM)
    COM_p = pin.SE3(np.eye(3), np.array([COM[0][0], COM[1][0], COM[2][0]]))
    place(viz, "world/COM", COM_p)
    # time.sleep(param['dt'])
    input()


# Plots 
import matplotlib.pyplot as plt

COP_array = np.array(COP_list)
COP_opt_array = np.array(COP_opt_list)
COP_array = COP_array.reshape(param['nb_samples'],2)
COP_lim = np.array(param['cop_lim'])

plt.plot(COP_lim[1]*np.ones(param['nb_samples']),'k')
plt.plot(COP_lim[0]*np.ones(param['nb_samples']),'k')
plt.plot(COP_array[:,0],'r', label = 'doc')
plt.plot(COP_opt_array[:,0],'k', label = 'opt')
plt.title('COP x [m]')
plt.show()

q_array = np.array(q_list)
q_array = q_array.reshape(param['nb_samples'],param['nb_joints'])
q_lim = np.array([param['q_min'],param['q_max']])

fig, ax = plt.subplots(param['nb_joints'],1)
for i in range(param['nb_joints']):
    ax[i].plot(q_array[:,i],'r', label='doc')
    ax[i].plot(q_opt[:,i],'k', label='opt')
    ax[i].plot(q_lim[0,i]*np.ones(param['nb_samples']),'b')
    ax[i].plot(q_lim[1,i]*np.ones(param['nb_samples']),'b')
    ax[i].set_title('q'+str(i)+' [rad]')
    ax[i].legend()

plt.show()

dq_array = np.array(dq_list)
dq_array = dq_array.reshape(param['nb_samples'],param['nb_joints'])
dq_lim = np.array(param['dq_lim'])

fig, ax = plt.subplots(param['nb_joints'],1)
for i in range(param['nb_joints']):
    ax[i].plot(dq_array[:,i],'r', label='doc')
    ax[i].plot(dq_opt[:,i],'k', label='opt')
    ax[i].plot(dq_lim[0]*np.ones(param['nb_samples']),'b')
    ax[i].plot(dq_lim[1]*np.ones(param['nb_samples']),'b')
    ax[i].set_title('dq'+str(i)+' [rad/s]')
    ax[i].legend()

plt.show()

tau_array = np.array(tau_list)
tau_array = tau_array.reshape(param['nb_samples'],param['nb_joints'])
tau_lim = np.array(param['tau_lim'])

fig, ax = plt.subplots(param['nb_joints'],1)
for i in range(param['nb_joints']):
    ax[i].plot(tau_array[:,i],'r', label='doc')
    ax[i].plot(tau_opt[:,i],'k', label='opt')
    ax[i].plot(tau_lim[i]*np.ones(param['nb_samples']),'b')
    ax[i].plot(-tau_lim[i]*np.ones(param['nb_samples']),'b')
    ax[i].set_title('tau'+str(i)+' [N.m]')
    ax[i].legend()

plt.show()