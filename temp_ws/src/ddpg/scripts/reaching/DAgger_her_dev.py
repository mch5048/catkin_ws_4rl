#!/usr/bin/env python


import random
import numpy as np
import cv2
import os
import pickle
from collections import deque
import rospy
from new_robotEnv_her import robotEnv

from std_msgs.msg import *
from std_srvs.srv import *
import math
from math import degrees as deg
from math import radians as rad

from random import *
import intera_interface

from intera_core_msgs.msg import JointCommand

from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
from collections import deque

import PyKDL
from intera_interface import Limb
from running_mean_std import RunningMeanStd

from intera_interface import CHECK_VERSION
from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
from intera_core_msgs.msg import (
    DigitalIOState,
    DigitalOutputCommand,
    IODeviceStatus
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from std_msgs.msg import Header
from intera_motion_msgs.msg import (
    Trajectory,
    TrajectoryOptions,
    Waypoint
)

from intera_motion_interface import (

    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
import time



# from ddpg import DAgger
ACTION_DIM = 7
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_POSE_STATE = 7
CHANNELS = 3 # RGB Kinect-v1

# It might be student?!

## Hyperparameter for DAgger Actor
learning_rate = 1e-3
img_dim = [100, 100, 3]
n_action = 7        # 7 DoF Robot Arm control
DAGGER_STEPS = 1000        # maximum step for a game
batch_size = 32     # for collecting imitation data
n_epoch = 100      # for training the model
n_episode = 20       # for retrain
memory = 10000
episode_count = 20
max_steps = 1000
N_STEP_RETURN = 10
total_time = 0
GAMMA = 0.97

Imgpath = "./trainImg" 
parent_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(parent_path, 'trainedData')

# Define RL variables 

memory = deque()
done = False
images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0, ACTION_DIM))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

joint_0_record = []
joint_1_record = []
joint_2_record = []
joint_3_record = []
joint_4_record = []
joint_5_record = []
joint_6_record = []

joint_0_vel_record = []
joint_1_vel_record = []
joint_2_vel_record = []
joint_3_vel_record = []
joint_4_vel_record = []
joint_5_vel_record = []
joint_6_vel_record = []


temp_goal_obs_crit = np.zeros((1,14)) # temp data to be stored in transition before real goal obs acquired
temp_goal_obs_actor = np.zeros((1,100,100,3))



isDemo = True

ACTION_LOW_BOUND = -1.0
ACTION_HIGH_BOUND = 1.0

# get Sawyer's robot values

# rospy.init_node("dagger")

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def normalize_action(action_arr):
    lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
    hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)
    _norm_action = lb_array + (action_arr+1.0*np.ones(action_arr.shape))*0.5*(hb_array - lb_array)
    _norm_action = np.clip(_norm_action, lb_array, hb_array)
    _norm_action = _norm_action.reshape(action_arr.shape)
    return _norm_action



## Implement DAgger here!

#1# First get demo data

if __name__ == '__main__':
    data_aggr_total_list = []
    data_aggr_list = []
    env = robotEnv()
    limb = Limb()
    env.isDemo = True
    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts')



    for e in range(episode_count):
        # if not e == 0:
            # rospy.wait_for_message("/teacher/start", JointCommand)
        data_aggr_list = []
        rospy.loginfo("Now It's EPISODE{0}".format(e))
        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t = env.reset_teaching() # done-> False
        obj_state_t = env.getObjPose()
        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t),  np.array(obj_state_t)]
        # s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t)]



        if e == 0:
            s_t0_rms = RunningMeanStd(shape=(1,) + s_t[0].shape)
            s_t1_rms = RunningMeanStd(shape=(1,) + s_t[1].shape)
            s_t2_rms = RunningMeanStd(shape=(1,) + s_t[2].shape)
            s_t3_rms = RunningMeanStd(shape=(1,) + s_t[3].shape)
            s_t4_rms = RunningMeanStd(shape=(1,) + s_t[4].shape)

        

            # achieved goals have the same shape with that of desired goals
            # achvd_obs_rms = RunningMeanStd(shape=goal_obs_actor.shape)
            # achvd_state0_rms = RunningMeanStd(shape=goal_state_critic[0].shape)
            # achvd_state1_rms = RunningMeanStd(shape=goal_state_critic[1].shape)

        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])
        s_t[0] = np.reshape(s_t[0],(-1,100,100,3))

        _rms_s_t = s_t[:]

        s_t[0] = normalize(s_t[0], s_t0_rms)
        s_t[1] = normalize(s_t[1], s_t1_rms)
        s_t[2] = normalize(s_t[2], s_t2_rms)
        s_t[3] = normalize(s_t[3], s_t3_rms)
        s_t[4] = normalize(s_t[4], s_t4_rms)

        


        # print s_t[0]
        # normalized obs
        # s_t[1] = (s_t[1] - s_t[1].mean())/s_t[1].std()
        # s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        # s_t[2] = (s_t[2] - s_t[2].mean())/s_t[2].std()
        # s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        # s_t[3] = (s_t[3] - s_t[3].mean())/s_t[3].std()
        # s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        # s_t[4] = (s_t[4] - s_t[4].mean())/s_t[4].std()
        # s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])
        step = 0
        done = False
        for s in range(max_steps):
            start_time = time.time()
            rospy.sleep(0.01)
            a_t = np.array(env.getAction_Dagger())

            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, achvd_pos, achvd_vel, achvd_color_obs, r_t, done = env.step_teaching(step)
            obj_state_t_1 = env.getObjPose()

            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]
            # s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1)]
            s_t_1[0] = np.reshape(s_t_1[0],(-1,100,100,3))
            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])

            _rms_s_t_1 = s_t_1[:]
     
            s_t_1[0] = normalize(s_t_1[0], s_t0_rms)
            s_t_1[1] = normalize(s_t_1[1], s_t1_rms)
            s_t_1[2] = normalize(s_t_1[2], s_t2_rms)
            s_t_1[3] = normalize(s_t_1[3], s_t3_rms)
            s_t_1[4] = normalize(s_t_1[4], s_t4_rms)

            # achvd_obs_actor = normalize(achvd_obs_actor, achvd_obs_rms)
            # achvd_state_critic[0] = normalize(achvd_state_critic[0], achvd_state0_rms)
            # achvd_state_critic[1] = normalize(achvd_state_critic[1], achvd_state1_rms)



            # achvd_obs_actor = np.reshape(achvd_obs_actor,(-1,100,100,3))
            # memory.append((s_t, , a_t[0], r_t))
            # a_t[0] = normalize_action(a_t[0]) # Action normalizations

            memory.append((s_t, s_t_1, a_t, r_t, done))
            # memory.append((s_t, a_t[0], r_t))

            if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                st, st1, at, discount_r, dn = memory.popleft()
                rt = discount_r
                for idx, (si, s1i, ai, ri, di) in enumerate(memory):
                    discount_r += ri * GAMMA ** (idx + 1)

                # s_t, a_t, discount_r = memory.popleft()
                # for idx, (si, ai, ri) in enumerate(memory):
                #     discount_r += ri * GAMMA ** (idx + 1)
                # buff.add(st, st1, at, discount_r, rt, s_t_1, _goal_critic, goal_obs_actor, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)

                # self.buffer.add((s_mem, a_mem, discount_r, s_, 1 if not done else 0))
                data_aggr_list.append([st, st1, at, discount_r, rt, s_t_1, temp_goal_obs_crit, temp_goal_obs_actor, dn, isDemo])      #A s_t , s_t+N, a_t+N, disc_r


                # achvd_obs_rms.update(achvd_obs_actor)
                # achvd_state0_rms.update(achvd_state_critic[0])
                # achvd_state1_rms.update(achvd_state_critic[1])

            elapsed_time = time.time() - start_time

            total_time +=elapsed_time

            s_t0_rms.update(_rms_s_t_1[0])
            s_t1_rms.update(_rms_s_t_1[1])
            s_t2_rms.update(_rms_s_t_1[2])
            s_t3_rms.update(_rms_s_t_1[3])
            s_t4_rms.update(_rms_s_t_1[4])

            if np.mod(s, 10) == 0:
                    print("Episode", e, "Step", step, "Action", a_t ,"10step-Time", total_time)
                    env.append_joint_angles_to_list()
                    total_time = 0
            if done:
                print 'Episode done'
                break
            s_t = s_t_1[:]
            step += 1            
        # end of epsiode ->put in goal obs
        print ('End of episode'+str(e)+', now saves the goal observations')   

        goal_state_pos, goal_state_vel, goal_obs = env.getGoalObs_teaching()
        goal_obs_actor = np.array(goal_obs)
        goal_state_critic = [np.array(goal_state_pos), np.array(goal_state_vel)]
        goal_obs_actor = np.reshape(goal_obs_actor,(-1,100,100,3))
        norm_goal_state_critic = goal_state_critic[:]

        if e == 0:
            goal_obs_rms = RunningMeanStd(shape=(1,) + goal_obs_actor.shape)
            goal_state0_rms = RunningMeanStd(shape=(1,) + goal_state_critic[0].shape)
            goal_state1_rms = RunningMeanStd(shape=(1,) + goal_state_critic[1].shape)
        # goal_obs_actor = normalize(goal_obs_actor, s_t0_rms)
        # goal_state_critic[0] = normalize(goal_state_critic[0], s_t1_rms)
        # goal_state_critic[1] = normalize(goal_state_critic[1], s_t2_rms)


        
        print 'reshaping goal observations...'
    


        print 'replacing goal observations'
        for _ in data_aggr_list:

            norm_goal_obs_actor = normalize(goal_obs_actor, goal_obs_rms)
            norm_goal_obs_actor = np.reshape(norm_goal_obs_actor,(-1,100,100,3))

            norm_goal_state_critic[0] = normalize(goal_state_critic[0], goal_state0_rms)
            norm_goal_state_critic[1] = normalize(goal_state_critic[1], goal_state1_rms)
            _g = np.concatenate((norm_goal_state_critic[0],norm_goal_state_critic[1]))
            _norm_goal_critic = np.reshape(_g,(-1,14))

            goal_obs_rms.update(goal_obs_actor)
            goal_state0_rms.update(goal_state_critic[0])
            goal_state1_rms.update(goal_state_critic[1])



            _[6] = _norm_goal_critic # replace 7th and 8th data of data_aggr_list [6]-> critic, [7] -> actor
            _[7] = norm_goal_obs_actor # replace 7th and 8th data of data_aggr_list [6]-> critic, [7] -> actor
            # goal observations also should be normalized
            # print _[7]
            # print _[7].shape
        print 'extends trajectories to total list'
        data_aggr_total_list.extend(data_aggr_list)



        # for s in range(max_steps):
    # for e in range(episode_count):


    print 'Now saves the collected trajectory in pickle format'
    # if not os.path.exists(data_path):
    #        os.makedirs(data_path)
    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts')
    env.save_joint_values_to_csv()
    with open ('traj_dagger.bin', 'wb') as f:             
        pickle.dump(data_aggr_total_list, f)


