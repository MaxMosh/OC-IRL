import pinocchio as pin
from tools.model_utils_motif import Robot
import meshcat
import numpy as np
import hppfcl
from tools.vizutils import addViewerBox, addViewerSphere, applyViewerConfiguration
import os

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Go one folder back
parent_directory = os.path.dirname(script_directory)

# conda-activate aws
# meshcat-server
# recuper le numero du server zmq

def create_double_pendulum_model_and_scene(param):
  
    meshloader=hppfcl.MeshLoader()
    # Adaptable file path relative to the script's directory
    urdf_name = "model/human_urdf/urdf/2dof_human_arm.urdf"
    urdf_path = os.path.join(parent_directory, urdf_name)
    urdf_meshes_path = os.path.join(parent_directory, "model")

    robot=Robot(urdf_path,urdf_meshes_path)

    robot.model.jointPlacements[1].translation = np.array([0,0,0]) # soulder
    robot.model.jointPlacements[2].translation = np.array([param["L"][0],0,0]) # elbow
    robot.model.frames[robot.model.getFrameId("hand")].placement.translation = np.array([param["L"][1],0,0]) # frame de la main

    model=robot.model
    visual_model = robot.visual_model
   
    # Modify the inertial parameters
    model.frames[model.getFrameId('upperarm')].inertia.mass=param["M"][0] #.lever .
    model.frames[model.getFrameId('lowerarm')].inertia.mass=param["M"][1] #.lever .
    model.frames[model.getFrameId('hand')].inertia.mass=0 #.lever .
    

    
    # To do add inertia term
    # model.frames[model.getFrameId('upperarm')].inertia.lever=np.array([param["K"][0],0,0]) 
    # model.frames[model.getFrameId('lowerarm')].inertia.lever=np.array([param["K"][1],0,0]) 
    # model.frames[model.getFrameId('hand')].inertia.lever=np.array([0,0,0])  
    
   
    #model.frames[model.getFrameId('upperarm')].inertia.inertia=np.eye(6) 
  
    # modify color for animation
    for i in range(len(visual_model.geometryObjects)):
        visual_model.geometryObjects[i].meshColor = np.array([0.05, 0.8, 0.2, 1.0])       
     

    #print(model.frames.tolist()) # use to display the model
     
    # add a XYZ frame at the the hand frame
    FIDX = robot.model.getFrameId("hand")
    JIDX = robot.model.frames[FIDX].parentJoint

    X = pin.utils.rotate("y", np.pi / 2)
    Y = pin.utils.rotate("x", -np.pi / 2)
    Z = np.eye(3)

    hand_p = robot.model.frames[FIDX].placement.translation# np.array([0, 0, 0])
    
    
    visual_model.addGeometryObject(pin.GeometryObject("axis_x", JIDX, pin.SE3(X, hand_p), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/X_arrow.stl",np.array([0.0025,0.0025,0.0025]) ))
    visual_model.geometryObjects[-1].meshColor = np.array([1, 0, 0, 1.0])

    visual_model.addGeometryObject(pin.GeometryObject("axis_y", JIDX, pin.SE3(Y, hand_p), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.0025,0.0025,0.0025]) ))
    visual_model.geometryObjects[-1].meshColor = np.array([0, 1, 0, 1.0])

    visual_model.addGeometryObject(pin.GeometryObject("axis_z", JIDX, pin.SE3(Z, hand_p), meshloader.load(urdf_meshes_path+"/human_urdf/meshes/Z_arrow.stl"),urdf_meshes_path+"/human_urdf/meshes/Y_arrow.stl",np.array([0.0025,0.0025,0.0025]) ))
    visual_model.geometryObjects[-1].meshColor = np.array([0, 0, 1, 1.0])       
                    
    viz = pin.visualize.MeshcatVisualizer(model, robot.collision_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.loadViewerModel()
      
    # modify color for animation
    for i in range(4,len(visual_model.geometryObjects)-3):
        visual_model.geometryObjects[i].meshColor = np.array([0.88, 0.05, 0.1, 0.5])   
    # Display another robot.
    viz_hum = pin.visualize.MeshcatVisualizer(model, robot.collision_model, visual_model)
    viz_hum.initViewer(viz.viewer)
    viz_hum.loadViewerModel(rootNodeName="pinocchio2")
    q = robot.q0.copy()
    q[1] = 1.0
    viz_hum.display(q)

    
    if param["IK"]==1:
        robot.model.addFrame(pin.Frame('mk_elbow_est',model.getJointId("elbow_Z"),model.getFrameId("upperarm"),pin.SE3(np.eye(3),np.array([0,0,0.0])),pin.OP_FRAME))# frames are in the elbow joint
        robot.model.addFrame(pin.Frame('mk_hand_est',model.getJointId("elbow_Z"),model.getFrameId("upperarm"),pin.SE3(np.eye(3),np.array([param["L"][1],0,0.0])),pin.OP_FRAME))# frames are in the hand joint
        sphere_color = [0, 1, 0, 1]
        radius = 0.01
        addViewerSphere(viz, "mk_shoulder_est", radius, sphere_color)
        addViewerSphere(viz, "mk_elbow_est", radius, sphere_color)
        addViewerSphere(viz, "mk_hand_est", radius, sphere_color)
    
    
    # Add objects
    
    print(param["pxf"])
    box_color = [0.5, 0.5, 0.6, 1]
    box_side = 0.1
    addViewerBox(viz, "box", box_side, box_side*5, box_side*5, box_color)
    applyViewerConfiguration(viz, "box", [param["pxf"]+box_side/2, 0.1, 0.1, 0, 0, 0, 1]) # 85% of the arm's length as in berret 2012

    viz.display(robot.q0)
     
    viz.display( np.array([0.3,np.pi/3]) )
    return robot, viz, viz_hum, param




#viz.display( np.array([0.3,np.pi/2]) )


# model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([1,0,0])


# print(model.frames[model.getFrameId('hand_fixed')].placement.translation)

# model.frames[model.getFrameId('hand_fixed')].placement.translation=np.array([2,0,0])

# print(model.frames[model.getFrameId('hand_fixed')].inertia.mass)

# model.frames[model.getFrameId('hand_fixed')].inertia.mass=0.1 #.lever .


# print(model.frames[model.getFrameId('hand_fixed')].inertia.mass)


#visual_model.visual_modeletryObjects.tolist()[visual_model.getvisual_modeletryId('hand')].placement.translation=np.array([2,0,0]) 

#visual_model.visual_modeletryObjects.tolist()[visual_model.getvisual_modeletryId('hand')].meshScale=np.array([0.0060, 0.0060, 0.0060]) 
#visual_model.visual_modeletryObjects.tolist()[visual_model.getvisual_modeletryId('hand_1')].meshColor=np.array([0, 1, 1, 0.5]) 

#frames[model.getFrameId('hand_fixed')].placement.translation=np.array([2,0,0])


#print(model.frames[model.getFrameId([-1])].placement.translation)



#input()