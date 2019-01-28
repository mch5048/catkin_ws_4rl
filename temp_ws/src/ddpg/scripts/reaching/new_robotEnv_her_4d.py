#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from intera_core_msgs.msg import JointCommand
from intera_core_msgs.msg import EndpointState
from gazebo_msgs.msg import ContactsState
from tf import TransformListener
from intera_io import IODeviceInterface
import intera_interface
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from string import Template
import time
import intera_interface
from ddpg.msg import GoalObs

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from gazebo_msgs.srv import (
    GetModelState
)

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)   

# Merge IK service for 4DoF implementation


from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from sawyer_sim_examples.msg import * 

from get_model_gazebo_pose import GazeboModel

#


base_dir = os.path.dirname(os.path.realpath(__file__))

# minimized 

fixed_orientation = Quaternion(
                         x=-0.00142460053167,
                         y=0.999994209902,
                         z=-0.00177030764765,
                         w=0.00253311793936)


N_STEP_RETURN = 10
GAMMA = 0.97

class robotEnv():
    def __init__(self, max_steps=700, isdagger=False):
        #init code
        rospy.init_node("robotEnv")
        self.isdagger = isdagger
        self._limb = intera_interface.Limb("right")
        self._tip_name = 'right_gripper_tip'
        self._gripper = intera_interface.Gripper()
        self.currentDist = 1
        self.previousDist = 1
        self.reached = False
        self.tf = TransformListener()
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        # self._limb.set_command_timeout(10.0)


        self.joint_names_4d = ['right_j0', 'right_j1', 'right_j2', 'right_j3']
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        
        self.joint_speeds = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_positions = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_velocities = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_efforts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.gripper_position_command = 100.0

        self.right_gripper_force = 0.0
        self.grasp_count = 0
        self.learn_grasp = False
        self.right_endpoint_position = [0,0,0]
        self.bridge = CvBridge()
        self.max_steps = max_steps
        self.done = False
        self.reward = 0
        self.reward_rescale = 1.0 # gradient is not that noticible
        self.isResetting = False
        self.isDemo = False


        self.joint_command = JointCommand()
        self.gripper =  intera_interface.Gripper("right")
    
        self.pub = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, tcp_nodelay=True, queue_size=1)
        self.pub3 = rospy.Publisher('/dagger/restart', JointCommand, queue_size=1)
        self.pub4 = rospy.Publisher('/robot/limb/right/joint_command_timeout', Float64, latch=True, queue_size=10)
        self.pub5 = rospy.Publisher('/ddpg/epi0', JointCommand, queue_size=1)
        self.pub6 = rospy.Publisher('/ddpg/reset2/', Bool, queue_size=1)

         #  /camera/color/image_raw</imageTopicName>
         #  /camera/depth/camera_info</cameraInfoTopicName>
         # /camera/depth/image_raw</depthImageTopicName>
         #  /camera/depth/camera_info</depthImageInfoTopicName>
         #  /camera/depth/points</pointCloudTopicName>

        
        # rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_ImgCB)

        # create Kinect raw image subscriber
        # Schunk gripper

        self.destPos = np.array([0.7, 0.15, -0.12+0.025])
        self.destObj = np.array([0.7, 0.10, 0])

        self.color_obs_list = []
        self.depth_obs_list = []

        # self.resize_factor1 = 100/640.0
        # self.resize_factor2 = 100/480.0
        self.resize_factor = 100/400.0
        self.resize_factor_real = 100/650.0

        self.gripper_length = 0.176
        self.distance_threshold = 0.1 # 5 cm, used for sparse reward
        self.position = [0.0, 0.0, 0.0]
        self.terminateCount = 0
        self.successCount = 0
        self.color_image = np.ones((400,400,3))
        self.squared_sum_eff = 0.0
        self.isReset = False
        self.daggerPosAction = [0.0,0.0,0.0]


        self.joint_0_record = []
        self.joint_1_record = []
        self.joint_2_record = []
        self.joint_3_record = []
        self.joint_4_record = []
        self.joint_5_record = []
        self.joint_6_record = []

        self.cmd_0_record = []
        self.cmd_1_record = []
        self.cmd_2_record = []

        
        self.joint_vel_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cartesian_command = [0.0, 0.0, 0.0]
        self.isInit = True
        # used for per-step elapsed time measurement
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0


        self.starting_joint_angles = {'right_j0': -0.041662954890248294,
                                 'right_j1': -1.0258291091425074,
                                 'right_j2': 0.0293680414401436,
                                 'right_j3': 1.37518162913313,
                                 'right_j4':  -0.06703022873354225,
                                 'right_j5': 0.7968371433926965,
                                 'right_j6': 1.7659649178699421}




        rospy.Subscriber('/robot/joint_states', JointState , self.jointStateCB)
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState , self.endpoint_positionCB)
        # rospy.Subscriber('/robot/limb/right/endpoint_state',  , self.endpoint_positionCB)
        rospy.Subscriber('/teacher/fin', JointCommand , self.doneCB)

        # subscribers for kinect images 
        rospy.Subscriber("/dynamic_objects/camera/raw_image", Image, self.rgb_ImgCB)
        # rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_ImgCB) # for Kinect
        # rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_ImgCB)
        # subscriber for joint command
        rospy.Subscriber("/robot/limb/right/joint_command", JointCommand, self.vel_CommandCB)
        rospy.Subscriber("/ddpg/reset/", Float64, self.resetCB)
        rospy.Subscriber("/teacher/pos_cmd_pub/", PosCmd, self.posCmdCB)
    

        # gz_model_obj = GazeboModel('sawyer')
        # print gz_model_obj



    def posCmdCB(self, poscmd):

        cmd_arr = poscmd.goal_cart_pos
        self.daggerPosAction = cmd_arr

    def resetCB(self, isReset):
        if isReset:
            self.isReset = True


    def vel_CommandCB(self, data):
        '''
            self.joint_command.mode = 2 # velocity control mode
            self.joint_command.names = self.joint_names
            self.joint_command.velocity =self.joint_speeds
            self.joint_command.header.stamp = rospy.Time.now()
        '''
        _temp_vel = list(data.velocity)
        if len(_temp_vel)>0 and abs(_temp_vel[0]) >0.0:
            self.joint_vel_command = _temp_vel
        # print list(self.joint_vel_command)

    def doneCB(self, data):
        print "IT's DONE!!!!!!!!!!!!!!!!"
        self.done = True

    def rgb_ImgCB(self, data):
        self.rcvd_color = data  ## ROS default image
        self.cimg_tstmp = rospy.get_time()
        self.color_image = self.bridge.imgmsg_to_cv2(self.rcvd_color, "bgr8") # 640 * 480

    def depth_ImgCB(self, data):
        data.encoding = "mono16"
        self.rcvd_depth = data  ## ROS default image
        self.dimg_tstmp = rospy.get_time()
        self.depth_image2 = self.bridge.imgmsg_to_cv2(self.rcvd_depth2, "mono16")  # 640 * 480

    def getObjPose(self, isReal=False):
        if not isReal:
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                object_state = object_state_srv("block", "world")
                self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.884])
            except rospy.ServiceException, e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))

        return self.destPos

    def depth_ImgCB(self, data):
        pass
        # self.rcvd_depth = data  ## ROS default image
        # self.dimg_tstmp = rospy.get_time()

        # self.depth_image = self.bridge.imgmsg_to_cv2(self.rcvd_depth, "bgr8") # 640 * 480


    def jointStateCB(self,msg): # callback function for joint state readings
        global reset_bool

        # name = msg.name[1:8] first three are head, finger 1, 2
        # position = msg.position[3:10]
        # velocity = msg.velocity[3:10]
        # effort = msg.effort[3:10]



        self.joint_positions = [self._limb.joint_angle('right_j0'),
        self._limb.joint_angle('right_j1'),
        self._limb.joint_angle('right_j2'),
        self._limb.joint_angle('right_j3'),
        self._limb.joint_angle('right_j4'),
        self._limb.joint_angle('right_j5'),
        self._limb.joint_angle('right_j6')]

        self.joint_velocities = [self._limb.joint_velocity('right_j0'),
        self._limb.joint_velocity('right_j1'),
        self._limb.joint_velocity('right_j2'),
        self._limb.joint_velocity('right_j3'),
        self._limb.joint_velocity('right_j4'),
        self._limb.joint_velocity('right_j5'),
        self._limb.joint_velocity('right_j6')]

        self.joint_efforts = [self._limb.joint_effort('right_j0'),
        self._limb.joint_effort('right_j1'),
        self._limb.joint_effort('right_j2'),
        self._limb.joint_effort('right_j3'),
        self._limb.joint_effort('right_j4'),
        self._limb.joint_effort('right_j5'),
        self._limb.joint_effort('right_j6')]

        self.squared_sum_eff = (self.joint_efforts[1])**2 +(self.joint_efforts[3])**2


        # if abs(self._limb.joint_effort('right_j3'))>=15:
        #     # print 'OOPS, excessive torque command on 4th joint'
        #     # print 'Joint vel command is'
        #     print self.joint_speeds
        self._limb.set_command_timeout(1.0)
        # command for 100kHz, self.joint_speeds are updated automatically
        # if not self.isResetting: # during resetting, stop vel command
        #     self.joint_command.mode = 2 # velocity control mode
        #     self.joint_command.names = self.joint_names
        #     _j3_pos = self.joint_positions[3]
        #     _j3_vel = self.joint_speeds[3]
        #     if _j3_pos >=1.70 and _j3_vel >=0: # Safety controller
        #         self.joint_speeds[3] = 0
        #     self.joint_command.velocity =self.joint_speeds
        #     self.joint_command.header.stamp = rospy.Time.now()
        #     self.pub.publish(self.joint_command)

        # for this implementation
        if not self.isResetting and not self.isInit and not self.isdagger: # during resetting, stop vel command
            pass
            # self.solveIK(self.cartesian_command)


    def solveIK(self, cartesian):
        cartesian = cartesian


        ik_pose = Pose()
        ik_pose.position.x = cartesian[0]
        ik_pose.position.y = cartesian[1]
        ik_pose.position.z = cartesian[2] 
        ik_pose.orientation.x = fixed_orientation.x
        ik_pose.orientation.y = fixed_orientation.y
        ik_pose.orientation.z = fixed_orientation.z
        ik_pose.orientation.w = fixed_orientation.w

        self._servo_to_pose(ik_pose)



    # for HER implementation # for HER implementation # for HER implementation # for HER implementation # for HER implementation
    # for HER implementation # for HER implementation # for HER implementation # for HER implementation # for HER implementation

    # for HER implementation # for HER implementation # for HER implementation # for HER implementation # for HER implementation

    def endpoint_positionCB(self,msg):
        self.right_endpoint_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]


#     def set_robot_homepose(self):

#         self.move_to_start(self.starting_joint_angles)



#     def move_to_start(self, start_angles=None):
#         print("Moving the {0} arm to start pose...".format('right'))
#         if not start_angles:
#             start_angles = dict(zip(self._joint_names, [0]*7))
#         self._guarded_move_to_joint_position(start_angles)
#         # self.gripper_open()

#     def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
#         if rospy.is_shutdown():
#             return
#         if joint_angles:
#             self._limb.move_to_joint_positions(joint_angles,timeout=timeout)
#         else:
#             rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")



# # for 4DoF IKservice implementation
#     def getHomogenMat(self, current_Pose=None):
#         # Note that 'current_pose' is from Sawyer's base to 7th axis frame

#         # position array of 3 DoF  (X,Y,Z)
#         position[0] = X, position[1] = Y, position[2] = Z in camera coordinate
#         currentURPose = [self.th2.ur5.x, self.th2.ur5.y, self.th2.ur5.z]
#         # Transformation btwn cam_coord to tool_coord
#         r_x = 0.0  # Roll rotation
#         # t_x = self.offsetRobToCam1[0]
#         # t_x = -32.0
#         t_x = -34.0
#         # t_y = self.offsetRobToCam1[1]
#         # t_y = -70.0
#         t_y = -80.0
#         # t_z = self.offsetRobToCam1[2]
#         t_z = -193.0

#         objectPose_toolFrame = PyKDL.Vector(position[0], position[1], position[2])

#         rotation_matrix = PyKDL.Rotation.RotX(rad(r_x))  # 3*3
#         translation_matrix = PyKDL.Vector(t_x, t_y, t_z)  # 3*1./
#         homegen_tf_matrix = PyKDL.Frame(rotation_matrix, translation_matrix)
#         objectPose_robotFrame1 = homegen_tf_matrix * objectPose_toolFrame  # 3*1
#         return objectPose_robotFrame1

    def _servo_to_pose(self, pose, time=1.0, steps=1.0):
        ''' An *incredibly simple* linearly-interpolated Cartesian move '''
        r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
        current_pose = self._limb.endpoint_pose()


        ik_delta = Pose()

        # ik_delta.position.x = (current_pose['position'].x - pose.position.x) / steps
        # ik_delta.position.y = (current_pose['position'].y - pose.position.y) / steps
        # ik_delta.position.z = (current_pose['position'].z - pose.position.z) / steps
        # ik_delta.orientation.x = (current_pose['orientation'].x - pose.orientation.x) / steps
        # ik_delta.orientation.y = (current_pose['orientation'].y - pose.orientation.y) / steps
        # ik_delta.orientation.z = (current_pose['orientation'].z - pose.orientation.z) / steps
        # ik_delta.orientation.w = (current_pose['orientation'].w - pose.orientation.w) / steps





        ik_delta.position.x =   pose.position.x  / steps + current_pose['position'].x 
        ik_delta.position.y =   pose.position.y  / steps + current_pose['position'].y 
        ik_delta.position.z =   pose.position.z  / steps + current_pose['position'].z 


        ik_delta.orientation.x = ( pose.orientation.x) / steps 
        ik_delta.orientation.y = ( pose.orientation.y) / steps
        ik_delta.orientation.z = ( pose.orientation.z) / steps
        ik_delta.orientation.w = ( pose.orientation.w) / steps


        # print '==================================='
        # print ik_delta
        # print '==================================='
        # print '==================================='
        # print ik_delta
        # print '==================================='


        
        joint_angles = self._limb.ik_request(ik_delta, self._tip_name)
        if joint_angles:
            self._limb.set_joint_positions(joint_angles)


        # optimization is required
        
        # for d in range(int(steps), -1, -1):
        #     if rospy.is_shutdown():
        #         return
        #     ik_step = Pose()
        #     ik_step.position.x = d*ik_delta.position.x + current_pose['position'].x 
        #     ik_step.position.y = d*ik_delta.position.y + current_pose['position'].y
        #     ik_step.position.z = d*ik_delta.position.z + current_pose['position'].z
        #     ik_step.orientation.x = pose.orientation.x # assume theres's no change in orientation
        #     ik_step.orientation.y = pose.orientation.y
        #     ik_step.orientation.z = pose.orientation.z
        #     ik_step.orientation.w = pose.orientation.w
        #     joint_angles = self._limb.ik_request(ik_step, self._tip_name)
        #     if joint_angles:
        #         self._limb.set_joint_positions(joint_angles)
        #     # else:
        #         # rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
        #     r.sleep()


# for 4DoF IKservice implementation







    def getColor_observation(self):
        return self.color_image

    def getAction_Dagger(self):
        return  self.daggerPosAction
        # return self.joint_positions
        # return self.joint_efforts
        # return self.endpoint_pose
        # return self.joint_vel_command

    # def preprocess(observation, last_observation): # If DQN
    #     processed_observation = np.maximum(observation, last_observation)
    #     processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    #     return np.reshape(processed_observation, (1fFRAME_WIDTH, FRAME_HEIGHT))

    def getDepth_observation(self):
        return self.depth_image

    def getCurrentJointValues(self):
        # rospy.sleep(0.05)
        return self.joint_positions, self.joint_velocities, self.joint_efforts

    def getCurrentPose(self):
        return self.right_endpoint_position


    def setCartAction(self, action):
        self.cartesian_command = action



    def setJointValues(self,tjv):
        self.joint_speeds = tjv
        # rospy.sleep(0.1) # for ctrl period
        # self.joint_command.mode = 2 # velocity control mode
        # self.joint_command.names = self.joint_names
        # self.joint_command.velocity =self.joint_speeds
        # self.joint_command.header.stamp = rospy.Time.now()

        # self.pub.publish(self.joint_command)

        # command_values = tjv
        # self.joint_command.mode = 3 # Vel control mode
        # self.joint_command.names = self.joint_names
        # self.joint_command.effort = command_values
        # self.pub.publish(self.joint_command)
        # self.joint_command.mode = 3 # Vel control mode
        # self.joint_command.names = self.joint_names
        # self.joint_command.effort = command_values
        # self.pub.publish(self.joint_command)
        # self.pub.publish(self.joint_command)
        return True

    def getDist(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            object_state = object_state_srv("block", "world")
            self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.9+0.025-0.0375])
        except rospy.ServiceException, e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        
        self.position = self.getCurrentPose()
        currentPos = np.array((self.position[0],self.position[1],self.position[2]))
        
        return np.linalg.norm(currentPos-self.destPos)

# DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger

    def reset_teaching(self):
        self.isResetting = True
        if self.isInit:
            self.isInit =False
        self.done = False
        self.successCount =0
        self.terminateCount =0

        self.joint_command.mode = 2 # velocity control mode
        self.joint_command.names = self.joint_names
        self.joint_command.velocity = self.joint_speeds
        # print self.joint_command
        print 'All trajectories are collected for this EPISODE'

        rospy.set_param('dagger_reset',"true") # param_name, param_value
        # print 'Reset param published'

        # rospy.wait_for_message("/teacher/start",JointCommand)
        
        print 'Waiting for new episode to start'
        
        while not rospy.is_shutdown():
            if rospy.has_param('epi_start'):
                break                    

        rospy.delete_param('epi_start')

        color_obs = self.getColor_observation()
        self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()
        # cv2.imwrite('/home/irobot/Downloads/Cheolhui/abc.png', color_obs)

        self.color_obs = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
        # resized_obsS_depth = cv2.resize(self.depth_image,None,fx=self.resize_factor,fy=self.resize_factor)

        #     if not np.array_equal(self.color_obs_list[-1], self.resized_obs) and not np.array_equal(self.depth_obs_list[-1], self.resized_obs_depth) : 
        #         self.color_obs_list.append(self.resized_obs)
        #         count += 1
        #     else:
        #         pass
        # print ('Now %d frames for both color and depth acquired' %len(self.color_obs_list))
        self.learn_grasp = False
        self.grasp_count = 0
        self.isResetting = False
        

        # return self.color_obs, self.joint_pos, self.joint_vels, self.joint_effos
        return self.color_obs, self.joint_pos, self.joint_vels, self.joint_effos

 #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger
    def getGoalObs_teaching(self):
        # goal_pose = rospy.wait_for_message('/teacher/goal_obs',Pose) # dtype== ROS std_msg Pose
        # self._limb.set_joint_position_speed(0.3)
        # self.move_to_goal(goal_pose)
        gs_pos, gs_vel, goal_obs = self.get_goal_observation() # state for Critic,& obs for Actor
        print 'Goal obs acquired'
        rospy.sleep(0.2)
        print 'Now goes to next episode'
        return gs_pos, gs_vel, goal_obs # first two for critic, third one for actor



    def checkForTermination(self):

        if self.position[0] < 0.30 or abs(self.position[1])>0.5 or self.position[2]<-0.2  or self.position[2]>0.55 or self.position[0] > 0.75:
            self.terminateCount +=1
            # self.reward -=10
        if self.terminateCount == 50:
            self.terminateCount =0
            return True
        else:
            return False

    def checkForSuccess(self):
        curDist = self.getDist()
        # curDist = 1
        # print self.successCount
        if curDist < self.distance_threshold/1.0:
            self.successCount +=1
            self.reward +=10
        if self.successCount == 400:
            self.successCount =0
            return True
        else:
            return False

    def checkForEpiFin(self):
        curDist = self.getDist()
        # curDist = 1
        # print self.successCount
        if curDist < self.distance_threshold*1.2:
            return True
        else:
            return False

    def move_to_goal(self, pose=None):
        pose = pose
        print("Moves right arm to desired goal pose, solved by inverse kinematics position")
        self._servo_to_goal(pose)
        print("reached position to acquire goal info. ")

    def _servo_to_goal(self, pose, time=4.0, steps=400.0):
        ''' An *incredibly simple* linearly-interpolated Cartesian move '''
        r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
        current_pose = self._limb.endpoint_pose()
        ik_delta = Pose()
        ik_delta.position.x = (current_pose['position'].x - pose.position.x) / steps
        ik_delta.position.y = (current_pose['position'].y - pose.position.y) / steps
        ik_delta.position.z = (current_pose['position'].z - pose.position.z) / steps
        ik_delta.orientation.x = (current_pose['orientation'].x - pose.orientation.x) / steps
        ik_delta.orientation.y = (current_pose['orientation'].y - pose.orientation.y) / steps
        ik_delta.orientation.z = (current_pose['orientation'].z - pose.orientation.z) / steps
        ik_delta.orientation.w = (current_pose['orientation'].w - pose.orientation.w) / steps
        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            ik_step = Pose()
            ik_step.position.x = d*ik_delta.position.x + pose.position.x
            ik_step.position.y = d*ik_delta.position.y + pose.position.y
            ik_step.position.z = d*ik_delta.position.z + pose.position.z
            ik_step.orientation.x = d*ik_delta.orientation.x + pose.orientation.x
            ik_step.orientation.y = d*ik_delta.orientation.y + pose.orientation.y
            ik_step.orientation.z = d*ik_delta.orientation.z + pose.orientation.z
            ik_step.orientation.w = d*ik_delta.orientation.w + pose.orientation.w
            joint_angles = self._limb.ik_request(ik_step, "right_gripper_tip")
            if joint_angles:
                self._limb.set_joint_positions(joint_angles)
            else:
                rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            r.sleep()

    def get_goal_observation(self, isReal=False):
        # reach the goal, get the image @ resets  w/ IK
        # get states(fully observable)
        # joint values @ target position & image of reached manipulator
        # goal_state_pos = self.joint_positions
        # goal_state_vel = self.joint_velocities # could be near zero
        # _goal_obs = self.getColor_observation()
        # goal_obs = cv2.resize(_goal_obs, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_CUBIC)
        # # cv2.imwrite('/home/irobot/catkin_ws/src/ddpg/scripts/goal_obs.png', goal_obs)

        # return goal_state_pos, goal_state_vel, goal_obs

        goal_state_pos = self.joint_positions
        goal_state_vel = self.joint_velocities # could be near zero
        _goal_obs = self.getColor_observation()
        if isReal:
            goal_obs = cv2.resize(_goal_obs, None, fx=self.resize_factor_real, fy=self.resize_factor_real, interpolation=cv2.INTER_CUBIC)
        else:
            goal_obs = cv2.resize(_goal_obs, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('/home/irobot/catkin_ws/src/ddpg/scripts/goal_obs.png', goal_obs)

        return goal_state_pos, goal_state_vel, goal_obs


    # def get_goal_state(self):
    #     if self.has_object:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    #         goal += self.target_offset
    #         goal[2] = self.height_offset
    #         if self.target_in_the_air and self.np_random.uniform() < 0.5:
    #             goal[2] += self.np_random.uniform(0, 0.45)
    #     else:
    #         goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    #     return goal.copy()

    def reset(self, isReal=False, isStart=False):
        self.isResetting = True
        if self.isInit:
            self.isInit =False
        self.successCount =0
        self.terminateCount =0
        self.reward = 0
        # self._limb.exit_control_mode()
        # self._limb.set_command_timeout(15)
        if not isReal:
            if not isReal: 
                self.joint_command.mode = 2 # velocity control mode
                self.joint_command.names = self.joint_names
                self.joint_command.velocity = self.joint_speeds
                rospy.set_param('ddpg_reset',"true") # param_name, param_value
                print 'Reset param published'
                # else: # for zeroth episode
                    # self.pub5.publish(self.joint_command)
                    # pass
                # rospy.sleep(1)
            else:
                pass

            goal_pose = rospy.wait_for_message('/ddpg/goal_obs',Pose) # dtype== ROS std_msg Pose
            print 'Goal obs acquired'
            self._limb.set_joint_position_speed(0.3)
            self.move_to_goal(goal_pose)
            gs_pos, gs_vel, goal_obs = self.get_goal_observation(isReal=isReal) # state for Critic,& obs for Actor
            rospy.sleep(0.5)
            print 'Now moves to start position'

            # should reset after acquiring goal pose

            color_obs = self.getColor_observation()

            # cv2.imwrite('/home/irobot/Downloads/Cheolhui/abc.png', color_obs)
            a = Bool()
            self.pub6.publish(a)

            while not rospy.is_shutdown():
                if self.isReset:
                    self.isReset = False
                    break
            print 'Environment has been reset'
        else:
            color_obs = self.getColor_observation()
            goal_pose = self.getColor_observation()
            gs_pos, gs_vel, goal_obs = self.get_goal_observation(isReal=isReal) # state for Critic,& obs for Actor


        if isReal:
            self.color_obs = cv2.resize(color_obs,None,fx=self.resize_factor_real,fy=self.resize_factor_real,interpolation=cv2.INTER_CUBIC)

        else:
            self.color_obs = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)

        self.learn_grasp = False
        self.grasp_count = 0
        self.isResetting = False
        # rospy.set_param('ddpg_reset',"false")
        self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()
        while not self.joint_positions:
            print 'waiting joint vals'
            self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()

        # print '!!!!!!'
        return self.color_obs, self.joint_pos, self.joint_vels, self.joint_effos, gs_pos, gs_vel, goal_obs

    def get_observation_shape(self):
        gs_pos, gs_vel, goal_obs = self.get_goal_observation() # state for Critic,& obs for Actor
        return [(1,100,100,3), (1,7), (1,7), (1,7), (1,3), (1,7), (1,7), (1,100,100,3), (1,7), (1,7), (1,100,100,3)] # last three for substitute observations
                   # 0           1      2     3       4       5     6       7               8   9       10



    def step_teaching(self,step):
        self.prev_tic = self.tic # save previous step's tic in another variable
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        # joint velocities with OU noise are given
        if step == self.max_steps:
            self.done = True
            rospy.set_param('demo_success','true')


        curDist = self.getDist()
        # curDist = 1

            # reward =-curDist    #-0.0001*self.gripper.get_force()*(1/curDist)
        # self.reward =-curDist #- 0.001*self.squared_sum_eff    #-0.0001*self.gripper.get_force()*(1/curDist)
        self.reward = (curDist <= self.distance_threshold).astype(np.float32) #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)

        # self.reward = -(curDist > self.distance_threshold).astype(np.float32)
            # reward = -curDist # -(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.05 ) else 0)
            #if(curDist<0.05):
            #    done = True
            #    reward = 100
            # b =-0.0001*self.left_gripper_force*self.right_gripper_force*(1/curDist)
            #a = np.log10(np.linalg.norm(self.destObj-self.destPos)+0.9)
            #if(np.linalg.norm(self.destObj-self.destPos)<0.05):
            #    done = True
            #    reward = 100
            #a = -self.left_gripper_force+self.right_gripper_force

        # if self.checkForSuccess():
        #     print ('==================================================================================')
        #     print ('Terminates current Episode : SUCCEEDED')
        #     print ('==================================================================================')
        #     # reward +=5
        #     rospy.set_param('demo_success','true')
        #     self.done = True

        # # if self.checkForEpiFin():
        # #     rospy.set_param('demo_success','true')
        # #     self.done = True


        # if self.checkForTermination():
        #     print ('==================================================================================')
        #     print ('Terminates Episode current Episode : OUT OF BOUNDARY')
        #     print ('==================================================================================')
        #     self.done = True

        # if rospy.has_param('reached'):
        #     self.done = True
        #     rospy.delete_param('reached')
        #     print ('==================================================================================')
        #     print ('Reached target object')
        #     print ('==================================================================================')

        joint_pos_step, joint_vels_step, joint_efforts_step  = self.getCurrentJointValues()
        while not joint_pos_step:
            print 'waiting joint vals'
            joint_pos_step, joint_vels_step, joint_efforts_step = self.getCurrentJointValues()
        positions = self.getCurrentPose()
        color_obs = self.getColor_observation()
        self.color_obs1 = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)

        ach_pos, ach_vel = self.get_achieved_goal_Critic()
        _ach_color_obs = self.get_achieved_goal_Actor()
        ach_color_obs = cv2.resize(_ach_color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)


        if np.mod(step, 10)==0:
            print("PER STEP ELAPSED : ", self.elapsed)
            print("DISTANCE : ", curDist)
            print("SPARSE REWARD : ", self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
        # dist , observation, reward, doneself.joint_pos, self.joint_vels  = self.getCurrentJointValues()self.joint_pos, self.joint_vels  = self.getCurrentJointValues()self.joint_pos, self.joint_vels  = self.getCurrentJointValues()ff
        # return curDist,[tjv+positions+self.destPos.tolist()+self.destObj.tolist()+[self.gripper.get_force()], reward, done
        # return curDist, self.color_obs1, joint_pos_step, joint_vels_step, joint_efforts_step, self.reward_rescale*self.reward, self.done
        return curDist, self.color_obs1, joint_pos_step, joint_vels_step, joint_efforts_step, ach_pos, ach_vel, ach_color_obs, self.reward_rescale*self.reward, self.done



    def act_for_ctrl_period(self, vals):
        pass
        # tjv = vals.flatten().tolist() 

        # status = self.setJointValues(tjv)


    # for HER # for HER # for HER # for HER # for HER # for HER # for HER # for HER # for HER 

    def get_achieved_goal_Critic(self):
        return self.joint_positions, self.joint_velocities
 
    def get_achieved_goal_Actor(self):
        return self.color_image

    def get_substitute_goal(self, transition_list):
        # len(list) == 12
        # st, st1, at, discount_r, rt, s_t_1, acvhd_obs_t, achvd_state_t, achvd_state_critic , achvd_obs_actor, dn, isDem

        _, _, _, _, _, _, achvd_state_t1, acvhd_obs_t1, achvd_state_tN, achvd_obs_tN,_ ,_ = transition_list[-1]

        # st, st1, at, discount_r, rt, s_t_1, goal_state_critic, goal_obs_actor, dn, isDem
        # achieved goal @ t+1 & t+N
        return achvd_state_t1, acvhd_obs_t1, achvd_state_tN, achvd_obs_tN

    # for HER # for HER # for HER # for HER # for HER # for HER # for HER # for HER # for HER

    def fk_service_client(self, joint_pose, limb = "right"):
        # returned value contains PoseStanped
        # std_msgs/Header header
        # geometry_msgs/Pose pose

          ns = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
          fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
          fkreq = SolvePositionFKRequest()
          joints = JointState()
          joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                         'right_j4', 'right_j5', 'right_j6']
          # joints.position = [0.763331, 0.415979, -1.728629, 1.482985,t
          #                    -1.135621, -1.674347, -0.496337]
          joints.position = joint_pose
          # Add desired pose for forward kinematics
          fkreq.configuration.append(joints)
          # Request forward kinematics from base to "right_hand" link
          # fkreq.tip_names.append('right_hand')
          fkreq.tip_names.append('right_gripper_tip')

          try:
              rospy.wait_for_service(ns, 5.0)
              resp = fksvc(fkreq)
          except (rospy.ServiceException, rospy.ROSException), e:
              rospy.logerr("Service call failed: %s" % (e,))
              return False

          # Check if result valid
          if (resp.isValid[0]):
              # rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
              # rospy.loginfo("\nFK Cartesian Solution:\n")
              # rospy.loginfo("------------------")
              # rospy.loginfo("Response Message:\n%s", resp)
              # print resp.pose_stamp[0].pose
              return resp.pose_stamp[0].pose
          else:
              # rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
              return False

    def compute_substitute_reward(self, achieved_goal, substitute_goal, isNstep=False):
        # requires joint position list
        # compute distance between two goals from joint positions

        # print '=== Computing substitute reward...'


        _pos_achvd = self.fk_service_client(achieved_goal.tolist())
        _pos_sbsttd = self.fk_service_client(substitute_goal.tolist())

        while not _pos_achvd and not _pos_sbsttd:
            _pos_achvd = self.fk_service_client(achieved_goal.tolist())
            _pos_sbsttd = self.fk_service_client(substitute_goal.tolist())
            rospy.sleep(0.01)
        achvd = np.array([_pos_achvd.position.x, _pos_achvd.position.y, _pos_achvd.position.z])
        sbsttd = np.array([_pos_sbsttd.position.x, _pos_sbsttd.position.y, _pos_sbsttd.position.z])
        if not isNstep:
            return (np.linalg.norm(achvd-sbsttd) <= self.distance_threshold).astype(np.float32) # return onestep reward
        else:
            _r = (np.linalg.norm(achvd-sbsttd) <= self.distance_threshold).astype(np.float32)
            discount_r = _r
            for idx in range(0,N_STEP_RETURN): # N step states from the last assumed to be the same
                discount_r += _r * GAMMA ** (idx + 1)
            return discount_r

    def set_learning_phase(self):
        self.isResetting = True

    def unset_learning_phase(self):
        self.isResetting = False


    def step(self,vals, step, isReal=False):
        self.prev_tic = self.tic # save previous step's tic in another variable
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        # joint velocities with OU noise are given
        # For HER implementation, additionally return 1) Achieved goal: Critic; current vel, current pos // Actor; current color obs
        # 

        done = False
        if step == self.max_steps:
            done = True

        if not isReal:
            prevDist = self.getDist()

        # action to joint velocities

        # print vals
        # tjv = vals.flatten().tolist() 
        # self.gripper_position_command = (tjv[-1] +1)*50
        # squared_sum_jvs = sum(i*i for i in tjv)
        act = vals.flatten().tolist()

        # self.setCartAction(act)

        # appy action
        self.solveIK(act)

        # status = self.setJointValues(tjv)
        if not isReal:
            curDist = self.getDist()
            # curDist = 10
            # reward =-curDist    #-0.0001*self.gripper.get_force()*(1/curDist)
            # self.reward =-curDist #- 0.001*self.squared_sum_eff     #-0.0001*self.gripper.get_force()*(1/curDist)

            self.reward = (curDist <= self.distance_threshold).astype(np.float32) #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
            # self.reward = -curDist #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
        
        # reward = -(curDist > self.distance_threshold).astype(np.float32) #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
        # reward =10
            # if self.checkForSuccess():
            #     print ('==================================================================================')
            #     print ('Terminates current Episode : SUCCEEDED')
            #     print ('==================================================================================')

            #     # reward +=5
            #     done = True

            if self.checkForTermination():
                print ('==================================================================================')
                print ('Terminates Episode current Episode : OUT OF BOUNDARY')
                print ('==================================================================================')
                done = True

        # print '!!!!!'
        joint_pos_step, joint_vels_step, joint_efforts_step = self.getCurrentJointValues()
        while not joint_pos_step:
            print 'waiting joint vals'
            joint_pos_step, joint_vels_step, joint_efforts_step = self.getCurrentJointValues()
        positions = self.getCurrentPose()
        color_obs = self.getColor_observation()
        
        if isReal:
            self.color_obs1 = cv2.resize(color_obs,None,fx=self.resize_factor_real,fy=self.resize_factor_real,interpolation=cv2.INTER_CUBIC)
        else:        
            self.color_obs1 = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
        # status = False

        # self.color_obs1 = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
            # for HER # for HER # for HER # for HER # for HER # for HER # for HER # for HER
        if not isReal:
            ach_pos, ach_vel = self.get_achieved_goal_Critic()
            _ach_color_obs = self.get_achieved_goal_Actor()
            ach_color_obs = cv2.resize(_ach_color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
        else: # real
            ach_pos, ach_vel = self.get_achieved_goal_Critic()
            _ach_color_obs = self.get_achieved_goal_Actor()
            ach_color_obs = cv2.resize(_ach_color_obs,None,fx=self.resize_factor_real,fy=self.resize_factor_real,interpolation=cv2.INTER_CUBIC)
 
        if np.mod(step, 10)==0:
            print(" ")
            if not isReal:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale*self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
            print("Actions: ", act)


        # dist , observation, reward, done
        # return curDist,[tjv+positions+self.destPos.tolist()+self.destObj.tolist()+[self.gripper.get_force()], reward, done
        if isReal:
            curDist = 0

        # cv2.imwrite('/home/irobot/catkin_ws/src/ddpg/scripts/achvd_obs.png', ach_color_obs)
        # cv2.imwrite('/home/irobot/catkin_ws/src/ddpg/scripts/color_obs.png', self.color_obs1)




        return curDist, self.color_obs1, joint_pos_step, joint_vels_step, joint_efforts_step, ach_pos, ach_vel, ach_color_obs, self.reward_rescale*self.reward, done

    def done(self):
        self.sub.unregister()
        self.sub2.unregister()
        rospy.signal_shutdown("done")

    # def preprocess(observation, last_observation):
    #     processed_observation = np.maximum(observation, last_observation)
    #     processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    #     return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

    def append_joint_cmds_to_list(self):
        self.cmd_0_record.append(self.daggerPosAction[0])
        self.cmd_1_record.append(self.daggerPosAction[1])
        self.cmd_2_record.append(self.daggerPosAction[2])


    def append_joint_angles_to_list(self):

        # self.joint_0_record.append(self.joint_efforts[0])
        # self.joint_1_record.append(self.joint_efforts[1])
        # self.joint_2_record.append(self.joint_efforts[2])
        # self.joint_3_record.append(self.joint_efforts[3])
        # self.joint_4_record.append(self.joint_efforts[4])
        # self.joint_5_record.append(self.joint_efforts[5])
        # self.joint_6_record.append(self.joint_efforts[6])

        self.joint_0_record.append(self.joint_velocities[0])
        self.joint_1_record.append(self.joint_velocities[1])
        self.joint_2_record.append(self.joint_velocities[2])
        self.joint_3_record.append(self.joint_velocities[3])
        self.joint_4_record.append(self.joint_velocities[4])
        self.joint_5_record.append(self.joint_velocities[5])
        self.joint_6_record.append(self.joint_velocities[6])

    def save_joint_cmds_to_csv(self):

        print 'saves the Cmd to csv file'

        self.cmd_0_record_np = np.array(self.cmd_0_record)
        self.cmd_1_record_np = np.array(self.cmd_1_record)
        self.cmd_2_record_np = np.array(self.cmd_2_record)

        self.save_data_cmd = np.column_stack((self.cmd_0_record,
                                          self.cmd_1_record,
                                          self.cmd_2_record
                                          ))
        if os.path.isfile("4d_pos_cmd.csv"):
            os.remove("4d_pos_cmd.csv")
        np.savetxt("4d_pos_cmd.csv", self.save_data_cmd, delimiter=',', fmt='%.3e')

        self.joint_0_record = []
        self.joint_1_record = []
        self.joint_2_record = []

        print 'Successfully saved the Cmd data'



    def save_joint_values_to_csv(self):
         
        print 'Saves the joint values to csv file'


        self.joint_0_record_np = np.array(self.joint_0_record)
        self.joint_1_record_np = np.array(self.joint_1_record)
        self.joint_2_record_np = np.array(self.joint_2_record)
        self.joint_3_record_np = np.array(self.joint_3_record)
        self.joint_4_record_np = np.array(self.joint_4_record)
        self.joint_5_record_np = np.array(self.joint_5_record)
        self.joint_6_record_np = np.array(self.joint_6_record)
        self.save_data = np.column_stack((self.joint_0_record_np,
                                          self.joint_1_record_np,
                                          self.joint_2_record_np,
                                          self.joint_3_record_np,
                                          self.joint_4_record_np,
                                          self.joint_5_record_np,
                                          self.joint_6_record_np
                                          ))


        if os.path.isfile("7d_velocity.csv"):
            os.remove("7d_velocity.csv")
        np.savetxt("7d_velocity.csv", self.save_data, delimiter=',', fmt='%.3e')

        self.joint_0_record = []
        self.joint_1_record = []
        self.joint_2_record = []
        self.joint_3_record = []
        self.joint_4_record = []
        self.joint_5_record = []
        self.joint_6_record = []
        print 'Successfully saved the joint demo data'




if __name__ == "__main__":
            r = robotEnv()
            # print r.getCurrentJointValues()
