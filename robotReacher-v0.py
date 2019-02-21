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
from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from sawyer_sim_examples.msg import * 
from get_model_gazebo_pose import GazeboModel

base_dir = os.path.dirname(os.path.realpath(__file__))

fixed_orientation = Quaternion(
                         x=-0.00142460053167,
                         y=0.999994209902,
                         z=-0.00177030764765,
                         w=0.00253311793936)

N_STEP_RETURN = 10
GAMMA = 0.97

from gym import seeding
from gym import spaces
# register(
#         id='FetchReach-v0',
#         entry_point='openai_ros:task_envs.fetch_reach.fetch_reach.FetchReachEnv',
#         timestep_limit=1000,
#     )
ACTION_DIM = 3 # Cartesian
OBS_DIM = (100,100,3)      # POMDP
STATE_DIM = 24        # MDP
 
class robotEnv():
    def __init__(self, max_steps=700, isdagger=False, isPOMDP=False, train_indicator=0):
        """An implementation of OpenAI-Gym style robot reacher environment
        """
        rospy.init_node("robotEnv")
        # for compatiability
        self.action_space = spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype='float32')
        self.observation_space = spaces.Dict(dict(

            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))



        self.train_indicator = train_indicator # 0: Train 1:Test
        self.isdagger = isdagger
        self.isPOMDP = isPOMDP
        self._limb = intera_interface.Limb("right")
        if not train_indicator:
            self._tip_name = 'right_gripper_tip'
        else:
            self._tip_name = 'right_hand'

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
        self.bridge = CvBridge()

        self.joint_names_4d = ['right_j0', 'right_j1', 'right_j2', 'right_j3']
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self.joint_speeds = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_positions = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_velocities = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.joint_efforts = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.right_endpoint_position = [0,0,0]
        self.max_steps = max_steps
        self.done = False
        self.reward = 0
        self.reward_rescale = 1.0
        self.isDemo = False
        self.reward_type = 'sparse'


        self.joint_command = JointCommand()
        self.gripper =  intera_interface.Gripper("right")
    
        self.pub = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, tcp_nodelay=True, queue_size=1)
        self.pub3 = rospy.Publisher('/dagger/restart', JointCommand, queue_size=1)
        self.pub4 = rospy.Publisher('/robot/limb/right/joint_command_timeout', Float64, latch=True, queue_size=10)
        self.pub5 = rospy.Publisher('/ddpg/epi0', JointCommand, queue_size=1)
        self.resetPub = rospy.Publisher('/ddpg/reset2/', Bool, queue_size=1)

        self.destPos = np.array([0.7, 0.15, -0.12+0.025])
        self.destObj = np.array([0.7, 0.10, 0])

        self.color_obs_list = []
        self.depth_obs_list = []

        self.resize_factor = 100/400.0
        self.resize_factor_real = 100/650.0

        self.gripper_length = 0.176
        self.distance_threshold = 0.1
        self.position = [0.0, 0.0, 0.0]
        self.terminateCount = 0
        self.successCount = 0
        self.color_image = np.ones((400,400,3))
        self.squared_sum_eff = 0.0
        self.isReset = False
        self.daggerPosAction = [0.0,0.0,0.0]
        
        self.joint_vel_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cartesian_command = [0.0, 0.0, 0.0]
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

        self._action_scale = 1.0

        rospy.Subscriber('/robot/joint_states', JointState , self.jointStateCB)
        rospy.Subscriber('/robot/limb/right/endpoint_state', EndpointState , self.endpoint_positionCB)
        rospy.Subscriber('/teacher/fin', JointCommand , self.doneCB)

        if not self.train_indicator: # train
            rospy.Subscriber("/dynamic_objects/camera/raw_image", Image, self.rgb_ImgCB)
        else:
            rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_ImgCB)

        rospy.Subscriber("/robot/limb/right/joint_command", JointCommand, self.vel_CommandCB)
        rospy.Subscriber("/ddpg/reset/", Float64, self.resetCB)
        rospy.Subscriber("/teacher/pos_cmd_pub/", PosCmd, self.posCmdCB)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def doneCB(self, data):
        print ("Done")
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

    def depth_ImgCB(self, data):
        pass

    def jointStateCB(self,msg): # callback function for joint state readings

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
        self.squared_sum_vel = np.linalg.norm(np.array(self.joint_velocities))

        self.joint_efforts = [self._limb.joint_effort('right_j0'),
        self._limb.joint_effort('right_j1'),
        self._limb.joint_effort('right_j2'),
        self._limb.joint_effort('right_j3'),
        self._limb.joint_effort('right_j4'),
        self._limb.joint_effort('right_j5'),
        self._limb.joint_effort('right_j6')]
        self.squared_sum_eff = np.linalg.norm(np.array(self.joint_efforts))

        self._limb.set_command_timeout(1.0)

    def endpoint_positionCB(self,msg):
        self.right_endpoint_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def apply_action(self, cartesian):
        cartesian = cartesian
        ik_pose = Pose()
        ik_pose.position.x = cartesian[0]
        ik_pose.position.y = cartesian[1]
        ik_pose.position.z = cartesian[2] #- 0.14 if isReal else 0
        ik_pose.orientation.x = fixed_orientation.x
        ik_pose.orientation.y = fixed_orientation.y
        ik_pose.orientation.z = fixed_orientation.z
        ik_pose.orientation.w = fixed_orientation.w
        self._servo_to_pose(ik_pose)

    def _servo_to_pose(self, pose, time=1.0, steps=1.0):
        ''' An *incredibly simple* linearly-interpolated Cartesian move '''
        r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
        current_pose = self._limb.endpoint_pose()
        ik_delta = Pose()
        ik_delta.position.x =   pose.position.x  / steps + current_pose['position'].x 
        ik_delta.position.y =   pose.position.y  / steps + current_pose['position'].y 
        ik_delta.position.z =   pose.position.z  / steps + current_pose['position'].z #s - 0.14 if isReal else 0
        ik_delta.orientation.x = ( pose.orientation.x) / steps 
        ik_delta.orientation.y = ( pose.orientation.y) / steps
        ik_delta.orientation.z = ( pose.orientation.z) / steps
        ik_delta.orientation.w = ( pose.orientation.w) / steps
        joint_angles = self._limb.ik_request(ik_delta, self._tip_name)
        if joint_angles:
            self._limb.set_joint_positions(joint_angles)

    def getColor_observation(self):
        return self.color_image

    def getAction_Dagger(self):
        return  self.daggerPosAction

    def getDepth_observation(self):
        return self.depth_image

    def getCurrentJointValues(self):
        return self.joint_positions, self.joint_velocities, self.joint_efforts

    def getCurrentPose(self):
        return self.right_endpoint_position

    def setCartAction(self, action):
        self.cartesian_command = action

    def setJointValues(self,jvals):
        self.joint_speeds = jvals
        return True

    def getDist(self):
        DIST_OFFSET = -0.9+0.025-0.0375
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            object_state = object_state_srv("block", "world")
            self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z + DIST_OFFSET])
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))        
        self.position = self.getCurrentPose()
        currentPos = np.array((self.position[0],self.position[1],self.position[2]))        
        return np.linalg.norm(currentPos-self.destPos)

    def checkForTermination(self):
        """Termination triggers done=True
        """
        X_RANGE = range(0.3,0.75)
        Y_RANGE = range(-0.5,0.5)
        Z_RANGE = range(-0.2,0.55)

        if not self.position[0] in X_RANGE or not self.position[1] in Y_RANGE or not self.position[2] in Z_RANGE:
            self.terminateCount +=1

        if self.terminateCount == 50:
            self.terminateCount =0
            return True
        else:
            return False

    def checkForSuccess(self):
        """Success triggers done=True
        """
        curDist = self.getDist()
        if curDist < self.distance_threshold:
            self.successCount +=1
            self.reward +=1
        if self.successCount == 50:
            self.successCount =0
            return True
        else:
            return False

    def _pend_epi_transition(self):
        if self.isDagger:
            print ('All demo trajectories are collected for this EPISODE')
            rospy.set_param('dagger_reset',"true") # param_name, param_value        
            print ('Waiting for new episode to start')        
            while not rospy.is_shutdown():
                if rospy.has_param('epi_start'):
                    break                    
            rospy.delete_param('epi_start')   
            print ('Now starts new eisode')
        else:
            rospy.set_param('ddpg_reset',"true") # param_name, param_value
            print ('Reset param published')
            print ('Now moves to start position')
            _color_obs = self.getColor_observation()
            resetMsg = Bool()
            self.resetPub.publish(resetMsg)
            while not rospy.is_shutdown():
                if self.isReset:
                    self.isReset = False
                    break
            print ('Now starts new eisode')

    def _get_color_obs(self):         
        _color_obs = self.getColor_observation()
        if self.isReal:
            self.color_obs = cv2.resize(_color_obs,None,fx=self.resize_factor_real,fy=self.resize_factor_real,interpolation=cv2.INTER_CUBIC)
        else:
            self.color_obs = cv2.resize(_color_obs,None,fx=self.resize_factor,fy=self.resize_factor,interpolation=cv2.INTER_CUBIC)
        return self.color_obs

    def _get_joint_obs(self):
        joint_pos, joint_vels, joint_effos  = self.getCurrentJointValues()
        while not joint_pos:
            print ('waiting joint vals')
            joint_pos, joint_vels, joint_effos  = self.getCurrentJointValues()
        return joint_pos, joint_vels, joint_effos

    def _get_target_obj_obs(self):
        """Return target object pose. Experimentally supports only position info."""
        if not self.isReal:
            rospy.wait_for_service('/gazebo/get_model_state')
            try:
                object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                object_state = object_state_srv("block", "world")
                self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z-0.884])
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        return self.destPos

    def reset(self):
        """OpenAI Gym style reset function."""
        self.done = False
        self.successCount =0
        self.terminateCount =0
        self.reward = 0
        color_obs = self._get_color_obs()
        joint_pos, joint_vels, joint_effos = self._get_joint_obs()
        obj_pos = self._get_target_obj_obs()
        if self.isPOMDP: # Partially observable
            obs = [color_obs, joint_pos, joint_vels, joint_effos, obj_pos]
        else: # Fully observable
            obs = [joint_pos, joint_vels, joint_effos, obj_pos]
        return obs

    def reset_teaching(self):
        """OpenAI Gym style reset function.
           Will be used for demo data acquisition."""
        self.done = False
        self.successCount =0
        self.terminateCount =0
        self.reward = 0

        color_obs = self._get_color_obs()
        joint_pos, joint_vels, joint_effos  = self._get_joint_obs()
        obj_pos = self._get_target_obj_obs()
        if self.isPOMDP: # Partially observable
            obs = [color_obs, joint_pos, joint_vels, joint_effos]
        else: # Fully observable
            obs = [joint_pos, joint_vels, joint_effos]
        return obs

    def compute_reward(self):
        """ Reward computation for non-goalEnv.
        """
        curDist = self.getDist()
        if self.reward_type == 'sparse':
            return (curDist <= self.distance_threshold).astype(np.float32) # 1 for success else 0
        else:
            return -curDist -self.squared_sum_vel # -L2 distance -l2_norm(joint_vels)

    def compute_HER_rewards(env, achieved_goal, desired_goal):
        """Re-computed rewards for substituted goals. Only supports sparse reward setting.
        Computes batch array of rewards"""
        batch_dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (batch_dist <= self.distance_threshold).astype(np.float32)

    def step(self,_act, step):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done
        """
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        self.done = False
        if step == self.max_steps:
            self.done = True

        act = _act.flatten().tolist()
        self.apply_action(act)
        if not self.isReal:
            self.reward = self.compute_reward()
            if self.checkForTermination():
                print ('======================================================')
                print ('Terminates Episode current Episode : OUT OF BOUNDARY')
                print ('======================================================')
                self.done = True
        color_obs = self._get_color_obs()
        joint_pos, joint_vels, joint_effos = self._get_joint_obs()
        obj_pos = self._get_target_obj_obs()
        
        if np.mod(step, 10)==0:
            if not isReal:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale*self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
            print("Actions: ", act)

        if self.isPOMDP: # Partially observable
            obs = [color_obs, joint_pos, joint_vels, joint_effos]
        else: # Fully observable
            obs = [joint_pos, joint_vels, joint_effos]

        return obs, self.reward_rescale*self.reward, self.done

    def step_teaching(self,step):
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        self.done = False
        if step == self.max_steps:
            self.done = True
            rospy.set_param('demo_success','true')
        curDist = self.getDist()
        if not self.isReal:
            self.reward = self.compute_reward()
        color_obs = self._get_color_obs()
        joint_pos, joint_vels, joint_effos = self._get_joint_obs()
        obj_pos = self._get_target_obj_obs()

        if np.mod(step, 10)==0:
            print("PER STEP ELAPSED : ", self.elapsed)
            print("DISTANCE : ", curDist)
            print("SPARSE REWARD : ", self.reward)
            print("Current EE pos: " ,self.right_endpoint_position)
        return obs, self.reward_rescale*self.reward, self.done



    def close(self):        
        rospy.signal_shutdown("done")

