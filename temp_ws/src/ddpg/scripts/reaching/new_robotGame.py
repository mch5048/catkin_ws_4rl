#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import baxter_interface
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

from gazebo_msgs.srv import (
    GetModelState
)


base_dir = os.path.dirname(os.path.realpath(__file__))




class robotGame():
    def __init__(self):
        #init code
        rospy.init_node("robotGame")
        self._limb = intera_interface.Limb("right")
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
        self._limb.set_command_timeout(10.0)


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
        self.max_steps = 2000
        self.done = False
        self.reward = 0
        self.reward_rescale = 0.1
        self.isResetting = False
        self.isDemo = False


        self.joint_command = JointCommand()
        self.gripper =  intera_interface.Gripper("right")
    
        self.pub = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, tcp_nodelay=True, queue_size=1)
        self.pub2 = rospy.Publisher('/ddpg/reset', JointCommand, queue_size=1)
        self.pub3 = rospy.Publisher('/dagger/restart', JointCommand, queue_size=1)
        self.pub4 = rospy.Publisher('/robot/limb/right/joint_command_timeout', Float64, latch=True, queue_size=10)
        self.pub5 = rospy.Publisher('/ddpg/epi0', JointCommand, queue_size=1)

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
        self.gripper_length = 0.176
        self.distance_threshold = 0.1 # 5 cm, used for sparse reward
        self.position = [0.0, 0.0, 0.0]
        self.terminateCount = 0
        self.successCount = 0
        self.color_image = np.ones((400,400,3))
        self.squared_sum_eff = 0.0

        self.joint_0_record = []
        self.joint_1_record = []
        self.joint_2_record = []
        self.joint_3_record = []
        self.joint_4_record = []
        self.joint_5_record = []
        self.joint_6_record = []

        self.joint_vel_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rospy.Subscriber('/robot/joint_states', JointState , self.jointStateCB)
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState , self.endpoint_positionCB)
        # rospy.Subscriber('/robot/limb/right/endpoint_state',  , self.endpoint_positionCB)
        rospy.Subscriber('/teacher/fin', JointCommand , self.doneCB)

        # subscribers for kinect images 
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_ImgCB)
        # subscriber for joint command
        rospy.Subscriber("/robot/limb/right/joint_command", JointCommand, self.vel_CommandCB)

    def vel_CommandCB(self, data):
        '''
            self.joint_command.mode = 2 # velocity control mode
            self.joint_command.names = self.joint_names
            self.joint_command.velocity =self.joint_speeds
            self.joint_command.header.stamp = rospy.Time.now()
        '''
        _temp_vel = list(data.velocity)
        if len(_temp_vel)>0 and abs(_temp_vel[0]) >=0.0:
            self.joint_vel_command = _temp_vel
        # print list(self.joint_vel_command)

    def doneCB(self, data):
        print "IT's DONE!!!!!!!!!!!!!!!!"
        self.done = True

    def rgb_ImgCB(self, data):
        self.rcvd_color = data  ## ROS default image
        self.cimg_tstmp = rospy.get_time()
        self.color_image = self.bridge.imgmsg_to_cv2(self.rcvd_color, "bgr8") # 640 * 480


    def getObjPose(self, isReal=False):
        if not isReal:
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                object_state = object_state_srv("block", "world")
                self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.9+0.025])
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

        # command for 100kHz, self.joint_speeds are updated automatically
        if not self.isResetting: # during resetting, stop vel command
            self.joint_command.mode = 2 # velocity control mode
            self.joint_command.names = self.joint_names
            self.joint_command.velocity =self.joint_speeds
            self.joint_command.header.stamp = rospy.Time.now()
            self.pub.publish(self.joint_command)
            # print'learning action'

        #
        # print '!!!!!!!!!!!!!'
        # print name
        # print position
        # print '!!!!!!!!!!!!!'

        # temp_dict_pos = dict(zip(msg.name, msg.position))
        # temp_dict_vel = dict(zip(msg.name, msg.velocity))
        # temp_dict_eff = dict(zip(msg.name, msg.effort))

        # print temp_dict_pos
  
        # self.joint_positions = [temp_dict_pos[x] for x in self.joint_names]
        # self.joint_velocities = [temp_dict_vel[x] for x in self.joint_names]
        # self.joint_efforts = [temp_dict_eff[x] for x in self.joint_names]

        # self.joint_positions = position
        # self.joint_velocities = velocity
        # self.joint_efforts = effort
        # # print self.joint_velocities
        # self.joint_velocities = [temp_dict_vel[x] for x in self.joint_names]
        

        # self.joint_command.effort =self.joint_efforts
        # if(self.isResetting == False):
            # self.gripper.set_position(self.gripper_position_command)

    def endpoint_positionCB(self,msg):
        self.right_endpoint_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def getColor_observation(self):
        return self.color_image

    def getAction_Dagger(self):
        # return self.joint_positions
        # return self.joint_efforts
        return self.joint_vel_command

    # def preprocess(observation, last_observation): # If DQN
    #     processed_observation = np.maximum(observation, last_observation)
    #     processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    #     return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

    def getDepth_observation(self):
        return self.depth_image

    def getCurrentJointValues(self):
        # rospy.sleep(0.05)
        return self.joint_positions, self.joint_velocities, self.joint_efforts

    def getCurrentPose(self):
        return self.right_endpoint_position

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
            self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.9+0.025])
        except rospy.ServiceException, e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        
        self.position = self.getCurrentPose()
        currentPos = np.array((self.position[0],self.position[1],self.position[2]))
        return np.linalg.norm(currentPos-self.destPos)

# DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger

    def reset_teaching(self):
        self.isResetting = True
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
        
        print 'Waiting for new epsiode to start'
        
        while not rospy.is_shutdown():
            if rospy.has_param('epi_start'):
                break                    

        rospy.delete_param('epi_start')

        color_obs = self.getColor_observation()
        self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()
        # cv2.imwrite('/home/cheolhui/Downloads/Cheolhui/abc.png', color_obs)

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

        return self.color_obs, self.joint_pos, self.joint_vels, self.joint_effos

 #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger #DAgger

    def checkForTermination(self):

        if self.position[0] < 0.2 or abs(self.position[1])>0.5 or self.position[2]<-0.15  or self.position[2]>0.55 or self.position[0] > 0.75:
            self.terminateCount +=1
            self.reward -=10
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
        if self.successCount == 50:
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

    def reset(self, isReal=False, isStart=False):
        self.isResetting = True
        self.successCount =0
        self.terminateCount =0
        self.reward = 0
        # self._limb.exit_control_mode()
        # self._limb.set_command_timeout(15)
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

        rospy.wait_for_message("/ddpg/reset2/",Float64)
        print 'Environment has been reset'

        color_obs = self.getColor_observation()
        self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()
        while not self.joint_positions:
            print 'waiting joint vals'
            self.joint_pos, self.joint_vels, self.joint_effos  = self.getCurrentJointValues()
        # cv2.imwrite('/home/cheolhui/Downloads/Cheolhui/abc.png', color_obs)

        self.color_obs = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
        # resized_obs_depth = cv2.resize(self.depth_image,None,fx=self.resize_factor,fy=self.resize_factor)

        #     if not np.array_equal(self.color_obs_list[-1], self.resized_obs) and not np.array_equal(self.depth_obs_list[-1], self.resized_obs_depth) : 
        #         self.color_obs_list.append(self.resized_obs)
        #         count += 1
        #     else:
        #         pass
        # print ('Now %d frames for both color and depth acquired' %len(self.color_obs_list))

        
        self.learn_grasp = False
        self.grasp_count = 0
        self.isResetting = False
        # rospy.set_param('ddpg_reset',"false")

        # print '!!!!!!'
        return self.color_obs, self.joint_pos, self.joint_vels, self.joint_effos

    def step_teaching(self,step):

        # joint velocities with OU noise are given
        if step == self.max_steps:
            self.done = True

        curDist = self.getDist()
        # curDist = 1

            # reward =-curDist    #-0.0001*self.gripper.get_force()*(1/curDist)
        # self.reward =-curDist #- 0.001*self.squared_sum_eff    #-0.0001*self.gripper.get_force()*(1/curDist)

        self.reward = -(curDist > self.distance_threshold).astype(np.float32)
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

        if self.checkForSuccess():
            print ('==================================================================================')
            print ('Terminates current Episode : SUCCEEDED')
            print ('==================================================================================')
            # reward +=5
            rospy.set_param('demo_success','true')
            self.done = True

        # if self.checkForEpiFin():
        #     rospy.set_param('demo_success','true')
        #     self.done = True


        if self.checkForTermination():
            print ('==================================================================================')
            print ('Terminates Episode current Episode : OUT OF BOUNDARY')
            print ('==================================================================================')
            self.done = True

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

        if np.mod(step, 10)==0:
            print(" ")
            print("DISTANCE : ", curDist)
            print("SPARSE REWARD : ", self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
        # dist , observation, reward, doneself.joint_pos, self.joint_vels  = self.getCurrentJointValues()self.joint_pos, self.joint_vels  = self.getCurrentJointValues()self.joint_pos, self.joint_vels  = self.getCurrentJointValues()ff
        # return curDist,[tjv+positions+self.destPos.tolist()+self.destObj.tolist()+[self.gripper.get_force()], reward, done
        return curDist, self.color_obs1, joint_pos_step, joint_vels_step, joint_efforts_step, self.reward_rescale*self.reward, self.done

    def act_for_ctrl_period(self, vals):
        pass
        # tjv = vals.flatten().tolist() 

        # status = self.setJointValues(tjv)
        
    def step(self,vals, step, isReal=False):

        # joint velocities with OU noise are given
        done = False
        if step == self.max_steps:
            done = True

        if not isReal:
            prevDist = self.getDist()

        # action to joint velocities

        # print vals
        tjv = vals.flatten().tolist() 
        self.gripper_position_command = (tjv[-1] +1)*50
        # squared_sum_jvs = sum(i*i for i in tjv)


        status = self.setJointValues(tjv)
        if status:                                         
            if not isReal:
                curDist = self.getDist()
                # curDist = 10
                # reward =-curDist    #-0.0001*self.gripper.get_force()*(1/curDist)
                # self.reward =-curDist #- 0.001*self.squared_sum_eff     #-0.0001*self.gripper.get_force()*(1/curDist)

                self.reward = -(curDist > self.distance_threshold).astype(np.float32) #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
                # self.reward = -curDist #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
            
            # reward = -(curDist > self.distance_threshold).astype(np.float32) #-(1. if self.position[0] < 0.2 else 0) -(1. if (self.position[2] <-0.13 or self.position[2] >0.05) else 0)
            # reward =10
                if self.checkForSuccess():
                    print ('==================================================================================')
                    print ('Terminates current Episode : SUCCEEDED')
                    print ('==================================================================================')

                    # reward +=5
                    done = True

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
            self.color_obs1 = cv2.resize(color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
            status = False

        if np.mod(step, 10)==0 and not isReal:
            print(" ")
            print("DISTANCE : ", curDist)
            print("SPARSE REWARD : ", self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
        # dist , observation, reward, done
        # return curDist,[tjv+positions+self.destPos.tolist()+self.destObj.tolist()+[self.gripper.get_force()], reward, done
        if isReal:
            curDist = 0

        return curDist, self.color_obs1, joint_pos_step, joint_vels_step, joint_efforts_step, self.reward_rescale*self.reward, done

    def done(self):
        self.sub.unregister()
        self.sub2.unregister()
        rospy.signal_shutdown("done")

    # def preprocess(observation, last_observation):
    #     processed_observation = np.maximum(observation, last_observation)
    #     processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    #     return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


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


        if os.path.isfile("torque_dagger.csv"):
            os.remove("torque_dagger.csv")
        np.savetxt("torque_dagger.csv", self.save_data, delimiter=',', fmt='%.3e')

        self.joint_0_record = []
        self.joint_1_record = []
        self.joint_2_record = []
        self.joint_3_record = []
        self.joint_4_record = []
        self.joint_5_record = []
        self.joint_6_record = []
        print 'Successfully saved the joint demo data'






if __name__ == "__main__":
            r = robotGame()
            # print r.getCurrentJointValues()
