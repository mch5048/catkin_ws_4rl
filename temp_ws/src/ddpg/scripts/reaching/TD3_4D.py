#!/usr/bin/env python
import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.engine.training import *
import json
import os
from PriorReplayBuffer_her_dev import ReplayBuffer_TD3 as ReplayBuffer
from ActorNetwork_TD3_4D import ActorNetwork # REFACTOR TO REMOVE GOALS
from CriticNetwork_TD3_4D import CriticNetwork # REFACTOR TO REMOVE GOALS
from new_robotEnv_TD3_4D import robotEnv # REFACTOR TO REMOVE GOALS
import time
from tempfile import TemporaryFile
from OU import OU
from Gaussian import Gaussian
from PARAM_Noise import Param_Noise
from SetupSummary import SummaryManager_TD3 as SummaryManager
from subprocess import CalledProcessError

# from BehaviourCloning2 import BehavClone as BC


from running_mean_std import RunningMeanStd
# from running_mean_std import RunningMeanStdMPI as RunningMeanStd
# from Baseline # from Baseline # from Baseline # from Baseline 

from collections import deque
import rospy
import time
import pickle
import random
# for Tensorflow Debugger
from tensorflow.python import debug as tf_debug
import cv2
import random
import tensorflow as tf
import os
import subprocess
import sys
import importlib
from std_srvs.srv import Empty, EmptyRequest

LOSS_COEF_DICT = {'onestep':0.8, 'Nstep':0.2, 'L2coef':0.01}

ACTION_LOW_BOUND = -1.0
ACTION_HIGH_BOUND = 1.0

OBS_SHAPE = (100,100,3)
POS = (7,)
VEL = (7,)
EFF = (7,)
OBJ = (3,)
CRIT_GOAL = (14,)


critic_log_path="/home/irobot/catkin_ws/src/ddpg/scripts/reaching/summary/her_critic"
path = '/home/irobot/catkin_ws/src/ddpg/scripts/'

####################################################################################################
# 
# 
#   Implementation of HER+Asymmetric Actor-Critic on DDPG with Behavioural Cloning and n-step return
# 
# 
####################################################################################################

# monitor mean and stddev changes

# MPI multi-thread learning # MPI multi-thread learning


# MPI multi-thread learning# MPI multi-thread learning

def normalize(ndarray, stats):
    if stats is None:
        return ndarray
    return (ndarray - stats.mean) / stats.std

# def normalize(tensor, stats, sess):
#     """
#     normalize a tensor using a running mean and std

#     :param tensor: (TensorFlow Tensor) the input tensor
#     :param stats: (RunningMeanStd) the running mean and std of the input to normalize
#     :return: (TensorFlow Tensor) the normalized tensor
#     """
#     if stats is None:
#         return tensor

#     norm_tensor = (tensor - stats.mean) / stats.std

#     norm_array = norm_tensor.eval()

#     return norm_array

def denormalize(ndarray, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return ndarray
    denorm_ndarray = ndarray * stats.std + stats.mean

    return denorm_ndarray

def reduce_std(tensor, axis=None, keepdims=False):
    """
    get the standard deviation of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the std over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """
    get the variance of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the variance over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)
def normalize_action(action_arr):
    lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
    hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)
    _norm_action = lb_array + (action_arr+1.0*np.ones(action_arr.shape))*0.5*(hb_array - lb_array)
    _norm_action = np.clip(_norm_action, lb_array, hb_array)
    _norm_action = _norm_action.reshape(action_arr.shape)
    return _norm_action

def denormalize_action(norm_action_arr):
    lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
    hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)

    _denorm_action = 2*(norm_action_arr - lb_array) / (hb_array - lb_array) -1.0*np.ones(action_arr.shape)
    _denorm_action = np.clip(_denorm_action, lb_array, hb_array)
    _denorm_action = _denorm_action.reshape(norm_action_arr.shape)
    return _denorm_action


def setup_summary():

    with tf.variable_scope('training_summary'):

        episode_total_reward = tf.Variable(0.,name='total_reward')
        episode_step = tf.Variable(0.,name='step')

        episode_q = tf.Variable(0.,name='q_val')
        episode_l2_loss = tf.Variable(0.,name='l2_crit_loss')
        episode_1step_loss = tf.Variable(0.,name='1step_loss')
        episode_nstep_loss = tf.Variable(0.,name='nstep_loss')
        # for q2
        episode_q_2 = tf.Variable(0.,name='q_val_2')
        episode_l2_loss_2 = tf.Variable(0.,name='l2_crit_loss_2')
        episode_1step_loss_2 = tf.Variable(0.,name='1step_loss_2')
        episode_nstep_loss_2 = tf.Variable(0.,name='nstep_loss_2')
        # param_noise_distance = tf.Variable(0.,name='param_noise_distance')

        gs = []
        gs.append(tf.summary.scalar('Total_Reward/Episode', episode_total_reward))
        gs.append(tf.summary.scalar('Took_Steps/Episode', episode_step))

        gs.append(tf.summary.scalar('Avg_Q_1/Episode', episode_q))
        gs.append(tf.summary.scalar('L2_Critic_Loss_1/Episode', episode_l2_loss))
        gs.append(tf.summary.scalar('1step_Loss_1/Episode', episode_1step_loss))
        gs.append(tf.summary.scalar('Nstep_Loss_1/Episode', episode_nstep_loss))

        gs.append(tf.summary.scalar('Avg_Q_2/Episode', episode_q_2))
        gs.append(tf.summary.scalar('L2_Critic_Loss_2/Episode', episode_l2_loss_2))
        gs.append(tf.summary.scalar('1step_Loss_2/Episode', episode_1step_loss_2))
        gs.append(tf.summary.scalar('Nstep_Loss_2/Episode', episode_nstep_loss_2))
        # s.append(tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev),family='optional'))    

        # histogram summary for mean stddev monitoring
        # gs.append(tf.histogram_summary('Nstep_Loss/Episode', episode_nstep_loss))


        summary_vars = [episode_total_reward, episode_step, 
                        episode_q, episode_l2_loss, episode_1step_loss, episode_nstep_loss,
                        episode_q_2, episode_l2_loss_2, episode_1step_loss_2, episode_nstep_loss_2
                       ]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge(gs)
        return summary_placeholders, update_ops, summary_op


def color_obs_normalize(img_obs):
    H, W, C = img_obs.shape
    flattened = img_obs.reshape(1, -1)
    
    sample_mean = flattened.mean(axis=1).reshape(-1, 1)
    sample_std = flattened.std(axis=1).reshape(-1, 1)
    
    img_normed = (flattened - sample_mean) / sample_std
    
    return img_normed.reshape(H, W, C)

def randomize_world():

    # We first wait for the service for RandomEnvironment change to be ready
    # rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/dynamic_world_service')
    # rospy.loginfo("Service /dynamic_world_service READY")
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()

    dynamic_world_service_call(change_env_request)

    # Init the FetchClient to move the robot arm
  


    # We generate as many dataset elements as it was indicated.
    # for i in range(number_of_dataset_elements):
        

def DDPGfD(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    # Change batchsize @HERE! # Change batchsize @HERE!
    BATCH_SIZE = 128
    SAMPLE_SIZE = 128
    # Change batchsize @HERE! # Change batchsize @HERE!
    GAMMA = 0.97
    TAU = 0.02     #Target Network HyperParameters
    LRA = 1e-4    #Learning rate for Actor
    LRC = 1e-3    #Lerning rate for Critic
    N_STEP_RETURN = 10

    # =========================Crucial hyperparameters=====================

    action_dim = 3  #num of joints being controlled
    state_dim_rgb = [100,100,3]  #num of features in state refer to 'new_robotGame/reset'
    goal_dim_rgb = [100,100,3]  #num of features in state refer to 'new_robotGame/reset'
    pos_dim_robot = 7  # 7 joint positions + gripper pos (last)
    vel_dim_robot = 7  # 7 joint positions + gripper pos (last)
    eff_dim_robot = 7  # 7 joint positions + gripper pos (last)

    goal_vel_dim = 7  # 7 joint positions + gripper pos (last)
    goal_eff_dim = 7  # 7 joint positions + gripper pos (last)
    
    goal_critic_dim = 14
    full_dim_robot = 21  # 7 joint pos + 7 joint vels + joint efforts
    state_dim_object = 3  #num of features in state refer to 'new_robotGame/eset'
    episode_count = 500 if (train_indicator) else 10
    resultArray=np.zeros((episode_count,2))

    max_steps = 1000
    pretrain_steps = 20000
    TOTAL_TRAIN_STEPS = max_steps*pretrain_steps

    reward = 0
    done = False
    step = 0
    epsilon = 0.3 if (train_indicator) else 0.0
    indicator = 0
    buffer_added = False
    # joint_vel_scale_base = 1.74
    # joint_vel_scale_shoulder_1 = 1.328
    # joint_vel_scale_shoulder_2 = 1.957
    # joint_vel_scale_shoulder_3  = 1.957
    # joint_vel_scale_wrist_1 =3.485
    # joint_vel_scale_wrist_2 =3.485
    # joint_vel_scale_wrist_3 =4.545
    joint_vel_scale_base = 2.0
    joint_vel_scale_shoulder_1 = 2.0
    joint_vel_scale_shoulder_2 = 2.0
    joint_vel_scale_shoulder_3  = 2.0
    joint_vel_scale_wrist_1 =2.0
    joint_vel_scale_wrist_2 =2.0
    joint_vel_scale_wrist_3 =2.0
    # joint_vel_scaling = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    # joint_vel_scaling = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # joint_vel_scaling = [0.1, 0.05, 0.2, 0.3, 0.1, 0.15, 0.15]
    cartesian_scaling = [.15, .15, .15]
    joint_torque_scale_base = 5
    joint_torque_scale_shoulder_1 = 5
    joint_torque_scale_shoulder_2 = 5
    joint_torque_scale_shoulder_3 = 5
    joint_torque_scale_wrist_1 =5
    joint_torque_scale_wrist_2 =5
    joint_torque_scale_wrist_3 =5
    memory = deque()
    show_reward = False
    total_time = 0
    dagger_data = 'traj_dagger.bin'
    isDem = False    
    cur_step_count = 0

    avg_q , l2_reg_loss, l2_loss, one_step_loss, n_step_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    avg_q_2 , l2_reg_loss_2, l2_loss_2, one_step_loss_2, n_step_loss_2, total_loss_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    isStart = False
    action_noise_ratio = 1.0
    # gauss_stddev = 0.005
    gauss_stddev = np.array([0.05,0.05,0.02])
    ou_stddev = np.array([0.005,0.005,0.002])


    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    total_loss = 0


    test_episode = 400
    MAX_UPDATES = 5
    isPretrain = True
    # noise_mean = np.array([-0.0492, -0.0006, 0.005, 0.0343, -0.0021, -0.0073, -0.0089])
    noise_mean = np.array([0.0, 0.0, 0.0])
    # noise_stddev = np.array([(0.05602+0.1648)/2, (0.04885+0.0619)/2, (0.166+0.1195)/2, (0.2008+0.02899)/2, (0.09876+0.07708)/2, (0.07953+0.117)/2, (0.0641+0.1224)/2])
    EXPLORE_CONST =3.0
    noise_stddev = EXPLORE_CONST*np.array([(0.05602+0.1648)/2, (0.04885+0.0619)/2, (0.166+0.1195)/2, (0.2008+0.02899)/2, (0.09876+0.07708)/2, (0.07953+0.117)/2, (0.0641+0.1224)/2])
    param_noise_interval = 50
    delayed_actor_update_interval = 4
    isParamNoise = False

    noise_idx = 0
    # 1 if OU 2 else Gauss 

    noise_type = 'OU' if noise_idx is 1 else 'Gauss'
    # NUM_CPU = 
    isReal= not train_indicator


    # Hyper-parameters # Hyper-parameters # Hyper-parameters # Hyper-parameters # Hyper-parameters
    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess,'cheolhui:7000')
    from keras import backend as K
    K.set_session(sess)
    # if(train_indicator == 1):
    env = robotEnv(train_indicator=train_indicator)

    obs_shape_list = env.get_observation_shape()
    # [self.color_bos.shape, self.joint_pos.shape, self.joint_vels.shape, self.joint_effos.shape, self.destPos, gs_pos.shape, gs_vel.shape, goal_obs.shape, gs_pos.shape, gs_vel.shape, goal_obs.shape] # last three for substitute observations



    K.set_learning_phase(1)


    critic = CriticNetwork(sess, full_dim_robot, state_dim_object, action_dim, BATCH_SIZE, TAU, LRC)


    # actor = ActorNetwork(sess, state_dim_rgb, pos_dim_robot, action_dim, goal_dim_rgb, BATCH_SIZE, TAU, LRA, critic.state_full, critic.state_obj, critic.goal_state_critic, critic.model)
    actor = ActorNetwork(sess, state_dim_rgb, pos_dim_robot, action_dim, BATCH_SIZE, TAU, LRA, critic.state_full, critic.state_obj, full_dim_robot, state_dim_object, action_dim)

    # param noise



    # Setup target network updates here!!
    sess.run(tf.global_variables_initializer())
    # critic.init_network()
    # actor.init_network()
    summary_writer = tf.summary.FileWriter(path + 'reaching/summary/her_4d', sess.graph)

    summary_manager = SummaryManager(sess=sess, obs_shape_list=obs_shape_list, summary_writer=summary_writer)
    summary_manager.setup_state_summary()
    summary_manager.setup_stat_summary()

    if not isParamNoise:
        if noise_type is 'Gauss':
            action_noise = Gaussian(mu=noise_mean, sigma=gauss_stddev)
        else:
            action_noise = OU(mu=noise_mean, sigma=ou_stddev)

    else: # adapt param boise
        actor._setup_param_noise()

    # param_noise =Param_Noise(initial_stddev=float(0.1), desired_action_stddev=float(0.1)) # default 0.1, 0.1wai
    buff = ReplayBuffer(BUFFER_SIZE, TOTAL_TRAIN_STEPS) #Create replay buffer, PER, beta scheduled
    # Generate a Torcs environment


    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # summary_placeholders, update_ops, summary_op = setup_summary(stats_ops, stats_names)
    summary_placeholders, update_ops, summary_op = setup_summary()

    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts/reaching')
    np.random.seed(219)
    ################## CHANGE HERE FOR ADOPTING DAGGER #################### 
    DEMO_USE = True
    #######################################################################
    if os.path.exists(dagger_data) and train_indicator and DEMO_USE:
        print 'initialize the replay buffer with demo data'
        with open(dagger_data, 'rb') as f:
            dagger = pickle.load(f)
            for idx, item in enumerate(dagger):
                buff.add(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7]) 
            print (idx, 'data has retrieved')

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(path+"weights_her_4d/actormodel_100.h5")
        critic.model.load_weights(path+"weights_her_4d/criticmodel_.h5")
        critic.model_2.load_weights(path+"weights_her_4d/criticmodel2_.h5")
        actor.target_model.load_weights(path+"weights_her_4d/actormodel_.h5")
        critic.target_model.load_weights(path+"weights_her_4d/criticmodel_.h5")
        critic.target_model_2.load_weights(path+"weights_her_4d/criticmodel2_.h5")            

        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    pretrain_count = 0 

    # Training loop for each episode starts here! # Training loop for each episode starts here! # Training loop for each episode starts here!
    i = 0
    # for i in range(episode_count) and not rospy.is_shutdown(): # pseudo: for each episode

    # if not train_indicator:
    # env.set_robot_homepose()

    while not rospy.is_shutdown() and i <=episode_count:

        if pretrain_count >= pretrain_steps:
            isPretrain = False
        if i ==0:
            isStart = True
        else:
            isStart = False

        rospy.loginfo("Now It's EPISODE{0}".format(i))
        if isPretrain:
            print 'under pretrain step'
        else:
            print 'non pretrain step'
        ## Pseudo1: Sample a goal and initial state, then render goal observation
        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t = env.reset(isReal=isReal, isStart=isStart) #(self.color_obs, self.joint_values)
        obj_state_t = env.getObjPose(isReal=isReal)

        

        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t), np.array(obj_state_t)]
        
        # goal_obs_actor = np.array(goal_obs)
        # goal_state_critic = [np.array(goal_state_pos), np.array(goal_state_vel)]


        if i == 0:
            if not train_indicator:
                print 'Loads the mean and stddev for test time'
                summary_manager.s_t0_rms.load_mean_std(path+'mean_std0.bin')
                summary_manager.s_t1_rms.load_mean_std(path+'mean_std1.bin')
                summary_manager.s_t2_rms.load_mean_std(path+'mean_std2.bin')
                summary_manager.s_t3_rms.load_mean_std(path+'mean_std3.bin')
                summary_manager.s_t4_rms.load_mean_std(path+'mean_std4.bin')
                # summary_manager.goal_obs_rms.load_mean_std(path+'mean_std5.bin')
                # summary_manager.goal_state0_rms.load_mean_std(path+'mean_std6.bin')
                # summary_manager.goal_state1_rms.load_mean_std(path+'mean_std7.bin')
                # summary_manager.achvd_obs_rms.load_mean_std(path+'mean_std8.bin')
                # summary_manager.achvd_state0_rms.load_mean_std(path+'mean_std9.bin')
                # summary_manager.achvd_state1_rms.load_mean_std(path+'mean_std10.bin')


        # s_t[0]: image, s_t[1]: position, s_t[2]: velocity, s_t[3]: effort, s_t[4]: object pose
        # reshape before normalization
        s_t[0] = np.reshape(s_t[0],(-1,100,100,3))
        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])
        # goal_obs_actor = np.reshape(goal_obs_actor,(-1,100,100,3))
        # goal_state_critic[0] = goal_state_critic[0].reshape(1,goal_state_critic[0].shape[0])
        # goal_state_critic[1] = goal_state_critic[1].reshape(1,goal_state_critic[1].shape[0])

        _rms_s_t = s_t[:]


        # s_t[0] = normalize(s_t[0], summary_manager.s_t0_rms) # DO NOT NORMALIZE VISUAL OBSERVATIONS
        s_t[1] = normalize(s_t[1], summary_manager.s_t1_rms)
        s_t[2] = normalize(s_t[2], summary_manager.s_t2_rms)
        s_t[3] = normalize(s_t[3], summary_manager.s_t3_rms)
        s_t[4] = normalize(s_t[4], summary_manager.s_t4_rms)
        # goal_obs_actor_norm = normalize(goal_obs_actor, summary_manager.goal_obs_rms)
        # goal_obs_actor_norm = goal_obs_actor                    # DO NOT NORMALIZE VISUAL OBSERVATIONS
        # goal_obs_actor_norm = normalize(goal_obs_actor, summary_manager.goal_obs_rms)
        # _norm_goal_state_crit0 = normalize(goal_state_critic[0], summary_manager.goal_state0_rms)
        # _norm_goal_state_crit1 = normalize(goal_state_critic[1], summary_manager.goal_state1_rms)


        # goal_state_critic_norm = [_norm_goal_state_crit0, _norm_goal_state_crit1]

        total_reward = 0.
        step = 0
        transition_store = [] # necessary for second loop



        # First Loop of HER, collect transitions just by doing actions # 1st Loop of HER, collect transitions just by doing actions #
        env.unset_learning_phase()
        start_time = time.time()
        # for j in range(max_steps) and not rospy.is_shutdown():
        j = 0
        while not rospy.is_shutdown() and j <=max_steps:
            if train_indicator: # 
                randomize_world()
            
            a_t = np.zeros([1,action_dim])
            # execute e-Greedy behaviour policy
            ## Pseudo2: Obtain action using behavioural policy


            if actor.param_noise and not isPretrain and isParamNoise: # adaptive param noise
                a_t_original = actor.adaptive_param_noise_actor.predict([s_t[0], s_t[1]]) # state; image & position

                a_t_original[0][0] = cartesian_scaling[0]*a_t_original[0][0]
                a_t_original[0][1] = cartesian_scaling[1]*a_t_original[0][1]
                a_t_original[0][2] = cartesian_scaling[2]*a_t_original[0][2]
                a_t[0] = a_t_original[0]
                
            else: # use OU noise for Gaussian noise
                a_t_original = actor.model.predict([s_t[0], s_t[1]])[0]# state; image & position
                
                # a_t_original[0][0] = cartesian_scaling[0]*a_t_original[0][0]
                # a_t_original[0][1] = cartesian_scaling[1]*a_t_original[0][1]
                # a_t_original[0][2] = cartesian_scaling[2]*a_t_original[0][2]

                _action_noise = action_noise()
                a_t[0] = a_t_original[0] +_action_noise*train_indicator # apply noise only for training

                a_t[0][0] = cartesian_scaling[0]*a_t[0][0]
                a_t[0][1] = cartesian_scaling[1]*a_t[0][1]
                a_t[0][2] = cartesian_scaling[2]*a_t[0][2]
            # a_t[0][3] = joint_vel_scaling[3]*a_t[0][3]
            # a_t[0][4] = joint_vel_scaling[4]*a_t[0][4]
            # a_t[0][5] = joint_vel_scaling[5]*a_t[0][5]
            # a_t[0][6] = joint_vel_scaling[6]*a_t[0][6]

            # action clipping

            a_t[0][0] = np.clip(a_t[0][0], -0.15, 0.15)
            a_t[0][1] = np.clip(a_t[0][1], -0.15, 0.15)
            a_t[0][2] = np.clip(a_t[0][2], -0.15, 0.15)

            a_t[0][0] *=500
            a_t[0][1] *=500
            a_t[0][2] *=500
            # a_t[0]*=10000
            # a_t[0][2] = -0.001

            # print a_t[0]
            # a_t[0][3] = np.clip(a_t[0][3], -0.02, 0.15)
            # a_t[0][4] = np.clip(a_t[0][4], -0.07, 0.1)
            # a_t[0][5] = np.clip(a_t[0][5], -0.1, 0.07)
            # a_t[0][6] = np.clip(a_t[0][6], -0.1, 0.06)
          
            ## Pseudo3: Execute action, receive reward and transition
            # state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo
            # print a_t[0]

            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, r_t, done = env.step(a_t[0], step, isReal=isReal) # 1st
            
            obj_state_t_1 = env.getObjPose(isReal=isReal)
            total_reward += np.average(r_t) # adds up averaged reward for each episode
            # achieved goal is returned every

            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]  
            # reshape observation
            s_t_1[0] = np.reshape(s_t_1[0],(-1,100,100,3))
            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])

            _rms_s_t_1 = s_t_1[:]

            # s_t_1[0] = normalize(s_t_1[0], summary_manager.s_t0_rms) # DO NOT NORMALIZE VISUAL OBSERVATIONS
            s_t_1[1] = normalize(s_t_1[1], summary_manager.s_t1_rms)
            s_t_1[2] = normalize(s_t_1[2], summary_manager.s_t2_rms)
            s_t_1[3] = normalize(s_t_1[3], summary_manager.s_t3_rms)
            s_t_1[4] = normalize(s_t_1[4], summary_manager.s_t4_rms)


            if j == 0: # update @ start of every episode
                summary_manager.update_state_summary(i+1, s_t[0], 
                    s_t[1],
                    s_t[2],
                    s_t[3],
                    s_t[4])

            # a_t[0] = normalize_action(a_t[0]) # Action normalizations
            # summary_manager.s_t0_rms.update(_rms_s_t_1[0])
            summary_manager.s_t1_rms.update(_rms_s_t_1[1])
            summary_manager.s_t2_rms.update(_rms_s_t_1[2])
            summary_manager.s_t3_rms.update(_rms_s_t_1[3])
            summary_manager.s_t4_rms.update(_rms_s_t_1[4])


            memory.append((s_t, s_t_1, a_t[0], r_t, done))
                        #    1    2       3      4            5           6          
            if len(memory) > (N_STEP_RETURN) and not isPretrain: # if buffer has more than 4 memories, execute
            # if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                st, st1, at, discount_r, dn = memory.popleft()
                rt = discount_r
                for idx, ( _, _, _, ri, _) in enumerate(memory): # index & contents
                    discount_r += ri * GAMMA ** (idx + 1)
                buff.add(st, st1, at, discount_r, rt, s_t_1, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                transition_store.append((st, st1, at, discount_r, rt, s_t_1, dn, isDem)) # HER


            s_t = s_t_1[:]  # careful for copying list!!
            step += 1
            pretrain_count += 1
            j  += 1

            if done:
                break 
            # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER
        
        if train_indicator:
            print 'Acquired transition data'
            # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER
            env.set_learning_phase() # stops Sawyer action command for cont' control

            # if len(transition_store) > 0 and not isPretrain: # set the last state as achieved goal and substitute
            #     print 'substitute reward computation starts'
            #     sbsttd_state_t1, sbsttd_obs_t1, sbsttd_state_tN, sbsttd_obs_tN = env.get_substitute_goal(transition_store) # extract achieved goals
            #     # print sbsttd_obs_tN.shape
            #     # Randomly sample trajectories from replay buffer
                


            #     for state, next_state, action, n_reward, reward, nstep_state, achvd_state_t1, acvhd_obs_t1, achvd_state_tN, achvd_obs_tN, dn, isDem in transition_store:
            #         assert sbsttd_state_t1[0].shape == achvd_state_t1[0].shape # assure that goal to be substituted has the same shape with substitute
            #         # print achvd_state_t1.shape

            #         sbsttd_r_1 = env.compute_substitute_reward(achvd_state_t1[0][0:7], sbsttd_state_t1[0][0:7]) # compute 1 step substitute reward
            #         sbsttd_r_n = env.compute_substitute_reward(achvd_state_tN[0][0:7], sbsttd_state_tN[0][0:7], isNstep=True) # compute n step substitute reward
            #         # substitute_reward_n = self.env.compute_reward(achieved_goal, substitute_goal)

            #         buff.add(state, next_state, action, n_reward, reward, nstep_state, sbsttd_state_tN, sbsttd_obs_tN, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
            #         # buff.add(st, st1, at, discount_r, rt, s_t_1, goal_state_critic, goal_obs_actor, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER

                # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER 
            print 'A-C update session starts'
            MAX_UPDATES = step
            DECAY_CONST = 4
            MAX_UPDATES /= DECAY_CONST
            for updates in range(int(MAX_UPDATES)):

                # Buffer index reference
                # state, next_state, action, n_reward, reward, nstep_state, dn, isDem # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                #    0         1        2        3        4         5        6    7   

                # Reference for new PER implementation
                '''
                    experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                '''

                # 'step' argument to beta_schedule executes linear increment of beta value for TD error correction
                batch, indices, weights_array = buff.getBatch(BATCH_SIZE, beta=buff.beta_schedule.value(step)) # just using this varaible might be OK
                weights_array = weights_array.reshape((BATCH_SIZE,1))
                # weights_array_n = weights_array_1
                # find loss

                states_t = ([e[0] for e in batch])
                states_t1 = ([e[1] for e in batch])
                states_tn = ([e[5] for e in batch])
                actions = np.asarray([e[2] for e in batch])
                rewards = np.asarray([e[4] for e in batch])
                rewards_tn = np.asarray([e[3] for e in batch])
                # additional; Goals

                dones = np.asarray([e[6] for e in batch])
                _isDemos = np.asarray([e[7] for e in batch])

                # Check if shape matches ?!
                y_t_n = np.asarray([e[2][0] for e in batch])
                y_t = np.asarray([e[2][0] for e in batch])

                sampled_img_state = np.array([img[0] for img in states_t])
                sampled_img_state = np.reshape(sampled_img_state,(-1,100,100,3))

                sampled_joint_pos_state = np.array([pos[1] for pos in states_t])
                sampled_joint_vel_state = np.array([vel[2] for vel in states_t])

                sampled_joint_eff_state = np.array([eff[3] for eff in states_t])
                sampled_obj_state = np.array([obj[4] for obj in states_t])
                # now reduce/reshape states to feed on neural networks
                full_state_t = np.concatenate((sampled_joint_pos_state, sampled_joint_vel_state, sampled_joint_eff_state), axis=1)
                reduced_full_state_t = np.reshape(full_state_t,(-1,full_dim_robot))
                #-----------------------------------------------------------------------#
                reduced_sampled_obj_state = np.reshape(sampled_obj_state,(-1,3))
                sampled_joint_pos_state = np.reshape(sampled_joint_pos_state,(-1,pos_dim_robot))
                # give the actor info of efforts rather tan position
                sampled_joint_vel_state = np.reshape(sampled_joint_vel_state,(-1,vel_dim_robot))
                # state values (current)



                # For n-step returns # For n-step returns # For n-step returns 
                sampled_img_state_n = np.array([imgn[0] for imgn in states_tn])
                sampled_img_state_n = np.reshape(sampled_img_state_n,(-1,100,100,3))

                sampled_joint_pos_state_n = np.array([posn[1] for posn in states_tn])
                sampled_joint_vel_state_n = np.array([veln[2] for veln in states_tn])
                sampled_joint_eff_state_n = np.array([effn[3] for effn in states_tn])
                sampled_obj_state_n = np.array([objn[4] for objn in states_tn])
                #-------------------------------------------------------------------------
                full_state_t_n = np.concatenate((sampled_joint_pos_state_n, sampled_joint_vel_state_n, sampled_joint_eff_state_n), axis=1)
                reduced_full_state_t_n = np.reshape(full_state_t_n,(-1,full_dim_robot))
                #-----------------------------------------------------------------------#
                reduced_sampled_obj_state_n = np.reshape(sampled_obj_state_n,(-1,3))
                sampled_joint_pos_state_n = np.reshape(sampled_joint_pos_state_n,(-1,pos_dim_robot))
                # give the actor info of efforts rather tan position
                sampled_joint_vel_state_n = np.reshape(sampled_joint_vel_state_n,(-1,vel_dim_robot))
                # For n-step returns # For n-step returns # For n-step returns


                # For 1-step returns # For 1-step returns # For 1-step returns 
                sampled_img_state_1 = np.array([img1[0] for img1 in states_t1])
                sampled_img_state_1 = np.reshape(sampled_img_state_1,(-1,100,100,3))

                sampled_joint_pos_state_1 = np.array([pos1[1] for pos1 in states_t1])
                sampled_joint_vel_state_1 = np.array([vel1[2] for vel1 in states_t1])
                sampled_joint_eff_state_1 = np.array([eff1[3] for eff1 in states_t1])
                sampled_obj_state_1 = np.array([obj1[4] for obj1 in states_t1])
                #-------------------------------------------------------------------------
                full_state_t_1 = np.concatenate((sampled_joint_pos_state_1, sampled_joint_vel_state_1, sampled_joint_eff_state_1), axis=1)
                reduced_full_state_t_1 = np.reshape(full_state_t_1,(-1,full_dim_robot))
                #-----------------------------------------------------------------------#
                reduced_sampled_obj_state_1 = np.reshape(sampled_obj_state_1,(-1,3))
                sampled_joint_pos_state_1 = np.reshape(sampled_joint_pos_state_1,(-1,pos_dim_robot))
                # give the actor info of efforts rather tan position
                sampled_joint_vel_state_1 = np.reshape(sampled_joint_vel_state_1,(-1,vel_dim_robot))
                # For 1-step returns # For 1-step returns # For 1-step returns 

                # action noise for target Q network smoothing            
                # reference from OpenAI spinup
                # epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
                # epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
                # a2 = pi_targ + epsilon
                # a2 = tf.clip_by_value(a2, -act_limit, act_limit)



                # noise_stddev = EXPLORE_CONST*np.array([(0.05602+0.1648)/2, (0.04885+0.0619)/2, (0.166+0.1195)/2, (0.2008+0.02899)/2, (0.09876+0.07708)/2, (0.07953+0.117)/2, (0.0641+0.1224)/2])

                if not isParamNoise:
                    _action_noise[0] = np.clip(_action_noise[0], -0.1, 0.1)
                    _action_noise[1] = np.clip(_action_noise[1], -0.1, 0.1)
                    _action_noise[2] = np.clip(_action_noise[2], -0.1, 0.1)
                else:
                    _action_noise = np.zeros(3)
                # _action_noise[3] = np.clip(_action_noise[3], -0.1, 0.1)
                # _action_noise[4] = np.clip(_action_noise[4], -0.1, 0.1)
                # _action_noise[5] = np.clip(_action_noise[5], -0.1, 0.1)
                # _action_noise[6] = np.clip(_action_noise[6], -0.1, 0.1)


                # test with roll-out for n-value
                # full state = joint_pos_state +joint_vel_state
                # Implement Twin Critic here!

                # for Q1
                _1_target_q_values_n, _  = critic.target_model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])[0]+_action_noise])
                _1_target_q_value_1, _ = critic.target_model.predict([reduced_full_state_t_1, reduced_sampled_obj_state_1, actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1])[0]+_action_noise])

                # for Q2
                _2_target_q_values_n, _  = critic.target_model_2.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])[0]+_action_noise])
                _2_target_q_value_1, _ = critic.target_model_2.predict([reduced_full_state_t_1, reduced_sampled_obj_state_1, actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1])[0]+_action_noise])

                # min operation

                target_q_values_n = np.minimum(_1_target_q_values_n, _2_target_q_values_n)
                target_q_value_1 = np.minimum(_1_target_q_value_1, _2_target_q_value_1)
                
                # for n-step target
                for k in range(len(batch)):
                    # print 'q_values_n'
                    # print target_q_values_n[k]


                    if dones[k]:
                        y_t_n[k] = rewards_tn[k] # if terminal state exists
                    else:
                        y_t_n[k] = rewards_tn[k] + (GAMMA**N_STEP_RETURN)*target_q_values_n[k] ## target function update
                # for 1-step target
                for l in range(len(batch)):

                    # print 'q_values_1'
                    # print target_q_value_1[l]

                    if dones[l]:
                        y_t[l] = rewards[l] # if terminal state exists
                    else:
                        y_t[l] = rewards[l] + (GAMMA)*target_q_value_1[l] ## target function update


                if (train_indicator): #training mode


                    _isDemos = np.reshape(_isDemos,(BATCH_SIZE,1))
                    isDemos = np.ones((BATCH_SIZE,action_dim),dtype=bool)

                    for x in range(BATCH_SIZE):
                        for y in range(action_dim):
                            isDemos[x][y] *=_isDemos[x]  

                    # print 'How minibatch transitions are composed'
                    # print isDemos

                    isDemos = np.reshape(isDemos,(BATCH_SIZE,action_dim))
                    isDemos_float = isDemos.astype(float)

                    # weights_array_n = weights_array_n.reshape((BATCH_SIZE,1))

                    y_t_n = y_t_n.reshape((BATCH_SIZE,1))
                    y_t = y_t.reshape((BATCH_SIZE,1))

                    # actor param noise computation
                    with tf.name_scope('adapt_actor_param_noise'):
                        # input -> action_batch
                        # implement get_action_batch@PER

                        # feed dict for param_noise should be states of t_0
                        # action_batch = buff.getActionBatch(BATCH_SIZE)                    
                        # if not isPretrain:
                        if not isPretrain and isParamNoise and  updates%param_noise_interval == 0:
                            print ('adapting actor param_noise')
                            actor._adapt_param_noise(sampled_img_state, sampled_joint_pos_state)

                    # Train Critic # Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic

                    with tf.variable_scope('train_critic'):
                        # _mse_loss = critic.model.train_on_batch([reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions], {"one_step_Q": y_t, 
                        # "N_step_Q": y_t_n}) #Minibatch loss -> MSE, and current estimation is made
                        # Inputs: onestep_target, Nstep_target, state_full, state_obj, goal_state_critic
                
                        #  Return valeus = onestep_loss, Nstep_loss, onestep_td_err, Nstep_td_err, q_pred, l2_loss
                        q1_info, q2_info = critic.train_critic_with_flatgrad(weights_array, y_t, y_t_n, reduced_full_state_t, reduced_sampled_obj_state, actions)
                        
                        # log the critic losses
                        l2_reg_loss += q1_info[5]
                        one_step_loss += q1_info[0]
                        n_step_loss += q1_info[1]
                        avg_q += np.average(q1_info[4])

                        l2_reg_loss_2 += q2_info[5]
                        one_step_loss_2 += q2_info[0]
                        n_step_loss_2 += q2_info[1]
                        avg_q_2 += np.average(q2_info[4])

                        td_err_n = (q1_info[3] + q2_info[3])/2
                        td_err_1 = (q1_info[2] + q2_info[2])/2

                        
                    
                    with tf.variable_scope('train_actor'):
                        if updates % delayed_actor_update_interval ==0:
                            actor.train_actor_with_flatgrad(sampled_img_state, sampled_joint_pos_state, reduced_full_state_t,reduced_full_state_t, reduced_sampled_obj_state, actions, isDemos_float, float(not isPretrain))
            

                    # Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic

                    # We're about to use a method that directly acquires gradients of both actor and critic
                    # Tensors should flow between actor and critic

                    if not isPretrain:
                        # if self.prioritized_replay:
                        # new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        # self.replay_buffer.update_priorities(batch_idxes, new_priorities)
                        # td errors works as new priorities
                        buff.update_priorities(indices, td_err_n, td_err_1, _isDemos)

                    # fireloss_array
                    # _critic_weights = critic.model.get_weights()


                    # total_loss = Nstep_loss + onestep_loss + l2_loss
                    total_loss = n_step_loss + one_step_loss + l2_reg_loss


                    # actor.train_actor(sampled_img_state, sampled_joint_pos_state, goal_states_Actr, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, bc_loss)
                    # actor.train_actor(sampled_img_state, sampled_joint_pos_state, goal_states_Actr, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions, isDemos_float)


                    # env.act_for_ctrl_period(a_t[0]) # for 10ms ctrl
                    actor.target_train()
                    # env.act_for_ctrl_period(a_t[0]) # 15
                    critic.target_train()
            env.unset_learning_phase() # stops Sawyer action command for cont' control

            print 'Saves mean and std for this training session'
            summary_manager.s_t0_rms.save_mean_std(path+'mean_std0.bin')
            summary_manager.s_t1_rms.save_mean_std(path+'mean_std1.bin')
            summary_manager.s_t2_rms.save_mean_std(path+'mean_std2.bin')
            summary_manager.s_t3_rms.save_mean_std(path+'mean_std3.bin')
            summary_manager.s_t4_rms.save_mean_std(path+'mean_std4.bin')
            # summary_manager.goal_obs_rms.save_mean_std(path+'mean_std5.bin')
            # summary_manager.goal_state0_rms.save_mean_std(path+'mean_std6.bin')
            # summary_manager.goal_state1_rms.save_mean_std(path+'mean_std7.bin')
            # summary_manager.achvd_obs_rms.save_mean_std(path+'mean_std8.bin')
            # summary_manager.achvd_state0_rms.save_mean_std(path+'mean_std9.bin')
            # summary_manager.achvd_state1_rms.save_mean_std(path+'mean_std10.bin')


            elapsed_time = time.time() - start_time
            total_time +=elapsed_time

            print ("onestep_loss", one_step_loss /float(MAX_UPDATES), "Nstep_loss", n_step_loss /float(MAX_UPDATES), "L2_loss", l2_loss /float(MAX_UPDATES))
            print("Episode", i,"Total_reward", total_reward, "Loss", total_loss/float(MAX_UPDATES), "Per-episode-Time", total_time)
            total_time = 0
               
                # Check for termination of each episode! & If true, record to Summary Writers

            # record stats for each episode
            # action_noise.reset() # reset OU process

            # l2_reg_loss
            # one_step_loss
            # n_step_loss

            stats = [total_reward, step, avg_q / float(MAX_UPDATES),l2_reg_loss / float(MAX_UPDATES), one_step_loss / float(MAX_UPDATES),
             n_step_loss / float(MAX_UPDATES), avg_q_2 / float(MAX_UPDATES), l2_reg_loss_2 / float(MAX_UPDATES), one_step_loss_2 / float(MAX_UPDATES),
             n_step_loss_2 / float(MAX_UPDATES)]

            for f in range(len(stats)):
                sess.run(update_ops[f], feed_dict={summary_placeholders[f]: float(stats[f])})

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i + 1)
            summary_manager.update_stat_summary(step=i+1)

            actor.update_actor_summary(summary_writer = summary_writer,global_step = i + 1)
            critic.update_critic_summary(summary_writer = summary_writer,global_step = i + 1)
            # monitor critic weights
            # write_log(callback, critic_history_names, critic_log, i + 1)

            avg_q , l2_reg_loss, l2_loss, one_step_loss, n_step_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            avg_q_2 , l2_reg_loss_2, l2_loss_2, one_step_loss_2, n_step_loss_2, total_loss_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        

            # buffer_added = False

            if np.mod(i, 10) == 0:
                if (train_indicator):
                    print("Now we save model")
                    actor.model.save_weights(path+"weights_her_4d/actormodel_"+str(i)+".h5", overwrite=True)
                    critic.model.save_weights(path+"weights_her_4d/criticmodel_"+str(i)+".h5", overwrite=True)
                    critic.model_2.save_weights(path+"weights_her_4d/criticmodel2_"+str(i)+".h5", overwrite=True)

            print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
            print("Total Step: " + str(step))
            print("")
            if (train_indicator):
                resultArray[i][0]=total_reward
                resultArray[i][1]=dist
                np.savetxt(path+'results.txt', resultArray)
                np.save(path+'results.npy', resultArray)
                actor.model.save_weights(path+"weights_her_4d/actormodel_"+".h5", overwrite=True)
                critic.model.save_weights(path+"weights_her_4d/criticmodel_"+".h5", overwrite=True)
                critic.model_2.save_weights(path+"weights_her_4d/criticmodel2_"+".h5", overwrite=True)

        i +=1
   # End of episode loop ###############################################################
   #####################################################################################

    # env.done()
    print("Finish.")


if __name__ == "__main__":
    DDPGfD(0) # 1: train 0: test