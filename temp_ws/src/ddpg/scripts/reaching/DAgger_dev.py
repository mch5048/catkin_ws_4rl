#!/usr/bin/env python


import random
import numpy as np
import cv2
import os
import pickle
from collections import deque
import rospy
from new_robotGame import robotGame

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
learning_rate = 1e-4
img_dim = [100, 100, 3]
n_action = 7        # 7 DoF Robot Arm control
DAGGER_STEPS = 1000        # maximum step for a game
batch_size = 32     # for collecting imitation data
n_epoch = 100      # for training the model
n_episode = 5       # for retrain
memory = 10000
episode_count = 30
max_steps = 500
N_STEP_RETURN = 5
total_time = 0
GAMMA = 0.99

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

isDemo = True

# get Sawyer's robot values

# rospy.init_node("dagger")

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std



## Implement DAgger here!

#1# First get demo data

if __name__ == '__main__':

    data_aggr_list = []
    env = robotGame()
    limb = Limb()
    env.isDemo = True



    for e in range(episode_count):
        # if not e == 0:
            # rospy.wait_for_message("/teacher/start", JointCommand)
        rospy.loginfo("Now It's EPISODE{0}".format(e))
        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t = env.reset_teaching() # done-> False
        obj_state_t = env.getObjPose()

        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t),  np.array(obj_state_t)]
        # s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t)]

        if e == 0:
            s_t0_rms = RunningMeanStd(shape=s_t[0].shape)
            s_t1_rms = RunningMeanStd(shape=s_t[1].shape)
            s_t2_rms = RunningMeanStd(shape=s_t[2].shape)
            s_t3_rms = RunningMeanStd(shape=s_t[3].shape)
            s_t4_rms = RunningMeanStd(shape=s_t[4].shape)



        s_t[0] = normalize(s_t[0], s_t0_rms)
        s_t[1] = normalize(s_t[1], s_t1_rms)
        s_t[2] = normalize(s_t[2], s_t2_rms)
        s_t[3] = normalize(s_t[3], s_t3_rms)
        s_t[4] = normalize(s_t[4], s_t4_rms)

        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        # s_t[2] = (s_t[2] - s_t[2].mean())/s_t[2].std()
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        # s_t[3] = (s_t[3] - s_t[3].mean())/s_t[3].std()
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        # s_t[4] = (s_t[4] - s_t[4].mean())/s_t[4].std()
        s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])

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

            s_t[0] = np.reshape(s_t[0],(-1,100,100,3))


            # rospy.sleep(0.09) # to meet the delta_T

            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, r_t, done = env.step_teaching(step)
            obj_state_t_1 = env.getObjPose()
            a_t = np.array(env.getAction_Dagger())

            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]
            # s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1)]

            # a_t = np.array([s_t_1[2]])

            # normalized obs
            # s_t_1[1] = (s_t_1[1] - s_t_1[1].mean())/s_t_1[1].std()
            # s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            # s_t_1[2] = (s_t_1[2] - s_t_1[2].mean())/s_t_1[2].std()
            # s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            # s_t_1[3] = (s_t_1[3] - s_t_1[3].mean())/s_t_1[3].std()
            # s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            # s_t_1[4] = (s_t_1[4] - s_t_1[4].mean())/s_t_1[4].std()
            # s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])           
            s_t_1[0] = normalize(s_t_1[0], s_t0_rms)
            s_t_1[1] = normalize(s_t_1[1], s_t1_rms)
            s_t_1[2] = normalize(s_t_1[2], s_t2_rms)
            s_t_1[3] = normalize(s_t_1[3], s_t3_rms)
            s_t_1[4] = normalize(s_t_1[4], s_t4_rms)


            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])





            s_t_1[0] = np.reshape(s_t_1[0],(-1,100,100,3))

            # memory.append((s_t, , a_t[0], r_t))

            memory.append((s_t, s_t_1, a_t, r_t, done))
            # memory.append((s_t, a_t[0], r_t))

            if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                st, st1, at, discount_r, dn = memory.popleft()
                rt = discount_r
                for idx, (si, s1i, ai, ri, di) in enumerate(memory):
                    discount_r += ri * GAMMA ** (idx + 1)

                # self.buffer.add((s_mem, a_mem, discount_r, s_, 1 if not done else 0))
                # if pretrain_count>=pretrain_steps: # for some timesteps, just train with demo data
                # buffer_added = True

                # s_t, a_t, discount_r = memory.popleft()
                # for idx, (si, ai, ri) in enumerate(memory):
                #     discount_r += ri * GAMMA ** (idx + 1)

                # self.buffer.add((s_mem, a_mem, discount_r, s_, 1 if not done else 0))
                data_aggr_list.append((st, st1, at, discount_r, rt, s_t_1, dn, isDemo))      #A s_t , s_t+N, a_t+N, disc_r
                s_t0_rms.update(s_t[0])
                s_t1_rms.update(s_t[1])
                s_t2_rms.update(s_t[2])
                s_t3_rms.update(s_t[3])
                s_t4_rms.update(s_t[4])


            elapsed_time = time.time() - start_time

            total_time +=elapsed_time

            if np.mod(s, 10) == 0:
                    env.append_joint_angles_to_list()
                    print("Episode", e, "Step", step, "Action", a_t ,"10step-Time", total_time)
                    total_time = 0
            if done:
                print 'Episode done'
                break
            s_t = s_t_1
            step += 1

    print 'Now saves the collected trajectory in pickle format'
    # if not os.path.exists(data_path):
    #        os.makedirs(data_path)
    # if os.path.isfile()
    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts')
    env.save_joint_values_to_csv()
    with open ('traj_dagger.bin', 'wb') as f:             
        pickle.dump(data_aggr_list, f)


