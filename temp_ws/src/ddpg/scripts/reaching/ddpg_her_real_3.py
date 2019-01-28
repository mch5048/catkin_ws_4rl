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
from PriorReplayBuffer_her_dev import ReplayBuffer
from ActorNetwork_her_dev_temp_3 import ActorNetwork
from CriticNetwork_her_dev_3 import CriticNetwork
from new_robotGame_her import robotGame
import time
from tempfile import TemporaryFile
from OU import OU
from Gaussian import Gaussian
from running_mean_std import RunningMeanStd
from collections import deque
import rospy
import time
import pickle
import random
# for Tensorflow Debugger
from tensorflow.python import debug as tf_debug
import cv2
import random


LOSS_COEF_DICT = {'onestep':1.0, 'Nstep':0.0, 'L2coef':0.01}

ACTION_LOW_BOUND = -1.0
ACTION_HIGH_BOUND = 1.0
epsilon_train = 0.2

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

# def setup_stats(self):

#         ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
#         names += ['obs_rms_mean', 'obs_rms_std']

#         ops += [tf.reduce_mean(self.critic_tf)]
#         names += ['reference_Q_mean']
#         ops += [reduce_std(self.critic_tf)]
#         names += ['reference_Q_std']

#         ops += [tf.reduce_mean(self.critic_with_actor_tf)]
#         names += ['reference_actor_Q_mean']
#         ops += [reduce_std(self.critic_with_actor_tf)]
#         names += ['reference_actor_Q_std']

#         ops += [tf.reduce_mean(self.actor_tf)]
#         names += ['reference_action_mean']
#         ops += [reduce_std(self.actor_tf)]
#         names += ['reference_action_std']

#         if self.param_noise:
#             ops += [tf.reduce_mean(self.perturbed_actor_tf)]
#             names += ['reference_perturbed_action_mean']
#             ops += [reduce_std(self.perturbed_actor_tf)]
#             names += ['reference_perturbed_action_std']

#         self.stats_ops = ops
#         self.stats_names = names



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

def denormalize_action(norm_action_arr):
    lb_array = ACTION_LOW_BOUND*np.ones(action_arr.shape)
    hb_array = ACTION_HIGH_BOUND*np.ones(action_arr.shape)

    _denorm_action = 2*(norm_action_arr - lb_array) / (hb_array - lb_array) -1.0*np.ones(action_arr.shape)
    _denorm_action = np.clip(_denorm_action, lb_array, hb_array)
    _denorm_action = _denorm_action.reshape(norm_action_arr.shape)
    return _denorm_action

def setup_summary():
    # Will be used for enriching debugging of training session
    # log_obs = [] if dimO[0]>20 else [tf.histogram_summary("obs/"+str(i),obs[:,i]) for i in range(dimO[0])]
    # log_act = [] if dimA[0]>20 else [tf.histogram_summary("act/inf"+str(i),act_test[:,i]) for i in range(dimA[0])]
    # log_act2 = [] if dimA[0]>20 else [tf.histogram_summary("act/train"+str(i),act_train[:,i]) for i in range(dimA[0])]
    # log_misc = [sum_p, sum_qq, tf.histogram_summary("td_error", td_error)]
    # log_grad = [grad_histograms(grads_and_vars_p), grad_histograms(grads_and_vars_q)]
    # log_train = log_obs + log_act + log_act2 + log_misc + log_grad

    with tf.variable_scope('training_summary'):
        episode_total_reward = tf.Variable(0.,name='total_reward')
        episode_q = tf.Variable(0.,name='q_val')
        episode_step = tf.Variable(0.,name='step')
        episode_l2_loss = tf.Variable(0.,name='l2_crit_loss')
        episode_1step_loss = tf.Variable(0.,name='1step_loss')
        episode_nstep_loss = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')
        s0_meanstd = tf.Variable(0.,name='nstep_loss')

        gs = []
        gs.append(tf.summary.scalar('Total_Reward/Episode', episode_total_reward))
        gs.append(tf.summary.scalar('Avg_Q/Episode', episode_q))
        gs.append(tf.summary.scalar('Took_Steps/Episode', episode_step))
        gs.append(tf.summary.scalar('L2_Critic_Loss/Episode', episode_l2_loss))
        gs.append(tf.summary.scalar('1step_Loss/Episode', episode_1step_loss))
        gs.append(tf.summary.scalar('Nstep_Loss/Episode', episode_nstep_loss))

        # histogram summary for mean stddev monitoring
        # gs.append(tf.histogram_summary('Nstep_Loss/Episode', episode_nstep_loss))





        summary_vars = [episode_total_reward, episode_q,
                        episode_step, episode_l2_loss, episode_1step_loss, episode_nstep_loss]
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

#path ="/media/dalinel/Maxtor/ddpg/"
def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    SAMPLE_SIZE = 64
    GAMMA = 0.99
    TAU = 0.01     #Target Network HyperParameters
    LRA = 1e-4    #Learning rate for Actor
    LRC = 1e-3    #Lerning rate for Critic
    N_STEP_RETURN = 5

    # =========================Crucial hyperparameters=====================

    action_dim = 7  #num of joints being controlled
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
    episode_count = 4000 if (train_indicator) else 10
    resultArray=np.zeros((episode_count,2))
    max_steps = 1000
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
    joint_vel_scaling = [0.1, 0.05, 0.2, 0.3, 0.1, 0.15, 0.15]
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
    pretrain_steps = 10000
    cur_step_count = 0
    avg_q = 0.0

    l2_reg_loss = 0.0
    one_step_loss = 0.0
    n_step_loss = 0.0
    isStart = False
    action_noise_ratio = 1.0
    gauss_stddev = 0.3
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    total_loss = 0
    test_episode = 400
    MAX_UPDATES = 1
    isPretrain = True
    noise_mean = np.array([-0.0492, -0.0006, 0.005, 0.0343, -0.0021, -0.0073, -0.0089])
    noise_stddev = np.array([(0.05602+0.1648)/2, (0.04885+0.0619)/2, (0.166+0.1195)/2, (0.2008+0.02899)/2, (0.09876+0.07708)/2, (0.07953+0.117)/2, (0.0641+0.1224)/2])

    # Hyper-parameters # Hyper-parameters # Hyper-parameters # Hyper-parameters # Hyper-parameters
    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess,'cheolhui:7000')
    from keras import backend as K
    K.set_session(sess)
    # if(train_indicator == 1):
    env = robotGame()

    obs_shape_list = env.get_observation_shape()
    # [self.color_bos.shape, self.joint_pos.shape, self.joint_vels.shape, self.joint_effos.shape, self.destPos, gs_pos.shape, gs_vel.shape, goal_obs.shape, gs_pos.shape, gs_vel.shape, goal_obs.shape] # last three for substitute observations


    s_t0_rms = RunningMeanStd(shape=obs_shape_list[0])
    s_t1_rms = RunningMeanStd(shape=obs_shape_list[1])
    s_t2_rms = RunningMeanStd(shape=obs_shape_list[2])
    s_t3_rms = RunningMeanStd(shape=obs_shape_list[3])
    s_t4_rms = RunningMeanStd(shape=obs_shape_list[4])
    goal_state0_rms = RunningMeanStd(shape=obs_shape_list[5])
    goal_state1_rms = RunningMeanStd(shape=obs_shape_list[6])
    goal_obs_rms = RunningMeanStd(shape=obs_shape_list[7])

    # achieved goals have the same shape with that of desired goals
    achvd_obs_rms = RunningMeanStd(shape=obs_shape_list[8])
    achvd_state0_rms = RunningMeanStd(shape=obs_shape_list[9])
    achvd_state1_rms = RunningMeanStd(shape=obs_shape_list[10])
    # setup ops and their names for tensorboard debugging
    ops = []
    names = []

    ops += [np.mean(s_t0_rms.mean), np.mean(s_t0_rms.std)]
    names += ['s_t0_rms_mean', 's_t0_rms_std']

    ops += [np.mean(s_t1_rms.mean), np.mean(s_t1_rms.std)]
    names += ['s_t1_rms_mean', 's_t1_rms_std']

    ops += [np.mean(s_t2_rms.mean), np.mean(s_t2_rms.std)]
    names += ['s_t2_rms_mean', 's_t2_rms_std']

    ops += [np.mean(s_t3_rms.mean), np.mean(s_t3_rms.std)]
    names += ['s_t3_rms_mean', 's_t3_rms_std']

    ops += [np.mean(s_t4_rms.mean), np.mean(s_t4_rms.std)]
    names += ['s_t4_rms_mean', 's_t4_rms_std']

    ops += [np.mean(goal_obs_rms.mean), np.mean(goal_obs_rms.std)]
    names += ['goal_obs_rms_mean', 'goal_obs_rms_std']

    ops += [np.mean(goal_state0_rms.mean), np.mean(goal_state0_rms.std)]
    names += ['goal_state0_rms_mean', 'goal_state0_rms_std']

    ops += [np.mean(achvd_obs_rms.mean), np.mean(achvd_obs_rms.std)]
    names += ['goal_state1_rms_mean', 'goal_state1_rms_std']

    ops += [np.mean(achvd_state0_rms.mean), np.mean(achvd_state0_rms.std)]
    names += ['achvd_state0_rms_mean', 'achvd_state0_rms_std']

    ops += [np.mean(achvd_state1_rms.mean), np.mean(achvd_state1_rms.std)]
    names += ['achvd_state1_rms_mean', 'achvd_state1_rms_std']

    stats_ops = ops
    stats_names = names



    K.set_learning_phase(1)

    critic = CriticNetwork(sess, full_dim_robot, state_dim_object, action_dim, goal_critic_dim, BATCH_SIZE, TAU, LRC)

    # critic = CriticNetwork(sess, full_dim_robot, action_dim, BATCH_SIZE, TAU, LRC)

    actor = ActorNetwork(sess, state_dim_rgb, pos_dim_robot, action_dim, goal_dim_rgb, BATCH_SIZE, TAU, LRA, critic.state_full, critic.state_obj, critic.goal_state_critic, critic.model)
    # action_noise = OU(mu=np.zeros(action_dim), sigma=float(0.05) * np.ones(action_dim))
    # Setup critic's gradient op here!
    # critic.add_value_grad_op(actor) # now we're able to exploit get_value_grad_to_action()



    # Setup target network updates here!!
    sess.run(tf.global_variables_initializer())
    # critic.init_network()
    # actor.init_network()
    summary_writer = tf.summary.FileWriter(path + 'reaching/summary/her', sess.graph)
    # Summary for Critic learning
    # callback = TensorBoard(critic_log_path)
    # callback.set_model(critic.model)
    # critic_history_names = ['critic_1step_loss','critic_Nstep_loss','critic_L2_reg_loss']


    # action_noise = Gaussian(mu=noise_mean, sigma=float(gauss_stddev) * np.ones(action_dim))
    action_noise = OU(mu=noise_mean, sigma=noise_stddev)
    buff = ReplayBuffer(BUFFER_SIZE) #Create replay buffer
    # Generate a Torcs environment




    # print actor.grads_and_vars_actor
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # summary_placeholders, update_ops, summary_op = setup_summary(stats_ops, stats_names)
    summary_placeholders, update_ops, summary_op = setup_summary()

    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts/reaching')

    if os.path.exists(dagger_data) and train_indicator:
        print 'initialize the replay buffer with demo data'
        with open(dagger_data, 'rb') as f:
            dagger = pickle.load(f)
            for idx, item in enumerate(dagger):
                if idx % 200 == 0:
                    print item
                buff.add(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9])
            print (idx, 'data has retrieved')

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(path+"weights_her/actormodel_30.h5")
        critic.model.load_weights(path+"weights_her/criticmodel_30.h5")
        actor.target_model.load_weights(path+"weights_her/actormodel_30.h5")
        critic.target_model.load_weights(path+"weights_her/criticmodel_30.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    pretrain_count = 0 

    # Training loop for each episode starts here! # Training loop for each episode starts here! # Training loop for each episode starts here!

    for i in range(episode_count): # pseudo: for each episode

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
        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t, goal_state_pos, goal_state_vel, goal_obs = env.reset(isReal=True, isStart=isStart) #(self.color_obs, self.joint_values)
        obj_state_t = env.getObjPose(isReal=True)

        print '!!!!!!!!!!!!!!!!!!!!!!'


        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t), np.array(obj_state_t)]
        
        goal_obs_actor = np.array(goal_obs)
        goal_state_critic = [np.array(goal_state_pos), np.array(goal_state_vel)]


        if i == 0:
            s_t0_rms = RunningMeanStd(shape=(1,) + s_t[0].shape)
            s_t1_rms = RunningMeanStd(shape=(1,) + s_t[1].shape)
            s_t2_rms = RunningMeanStd(shape=(1,) + s_t[2].shape)
            s_t3_rms = RunningMeanStd(shape=(1,) + s_t[3].shape)
            s_t4_rms = RunningMeanStd(shape=(1,) + s_t[4].shape)
            goal_obs_rms = RunningMeanStd(shape=(1,) + goal_obs_actor.shape)
            goal_state0_rms = RunningMeanStd(shape=(1,) + goal_state_critic[0].shape)
            goal_state1_rms = RunningMeanStd(shape=(1,) + goal_state_critic[1].shape)

            # achieved goals have the same shape with that of desired goals
            achvd_obs_rms = RunningMeanStd(shape=(1,) + goal_obs_actor.shape)
            achvd_state0_rms = RunningMeanStd(shape=(1,) + goal_state_critic[0].shape)
            achvd_state1_rms = RunningMeanStd(shape=(1,) + goal_state_critic[1].shape)

            # setup ops and their names for tensorboard debugging
            # ops = []
            # names = []

            # ops += [np.mean(s_t0_rms.mean), np.mean(s_t0_rms.std)]
            # names += ['s_t0_rms_mean', 's_t0_rms_std']

            # ops += [np.mean(s_t1_rms.mean), np.mean(s_t1_rms.std)]
            # names += ['s_t1_rms_mean', 's_t1_rms_std']

            # ops += [np.mean(s_t2_rms.mean), np.mean(s_t2_rms.std)]
            # names += ['s_t2_rms_mean', 's_t2_rms_std']

            # ops += [np.mean(s_t3_rms.mean), np.mean(s_t3_rms.std)]
            # names += ['s_t3_rms_mean', 's_t3_rms_std']

            # ops += [np.mean(s_t4_rms.mean), np.mean(s_t4_rms.std)]
            # names += ['s_t4_rms_mean', 's_t4_rms_std']

            # ops += [np.mean(goal_obs_rms.mean), np.mean(goal_obs_rms.std)]
            # names += ['goal_obs_rms_mean', 'goal_obs_rms_std']

            # ops += [np.mean(goal_state0_rms.mean), np.mean(goal_state0_rms.std)]
            # names += ['goal_state0_rms_mean', 'goal_state0_rms_std']

            # ops += [np.mean(achvd_obs_rms.mean), np.mean(achvd_obs_rms.std)]
            # names += ['goal_state1_rms_mean', 'goal_state1_rms_std']

            # ops += [np.mean(achvd_state0_rms.mean), np.mean(achvd_state0_rms.std)]
            # names += ['achvd_state0_rms_mean', 'achvd_state0_rms_std']

            # ops += [np.mean(achvd_state1_rms.mean), np.mean(achvd_state1_rms.std)]
            # names += ['achvd_state1_rms_mean', 'achvd_state1_rms_std']

            # stats_ops = ops
            # stats_nams = names

            # values = sess.run(stats_ops, feed_dict={
            #     self.obs0: self.stats_sample['obs0'],
            #     self.actions: self.stats_sample['actions'],
            # })





            if not train_indicator:
                print 'Loads the mean and stddev for test time'
                s_t0_rms.load_mean_std(path+'mean_std0.bin')
                s_t1_rms.load_mean_std(path+'mean_std1.bin')
                s_t2_rms.load_mean_std(path+'mean_std2.bin')
                s_t3_rms.load_mean_std(path+'mean_std3.bin')
                s_t4_rms.load_mean_std(path+'mean_std4.bin')
                goal_obs_rms.load_mean_std(path+'mean_std5.bin')
                goal_state0_rms.load_mean_std(path+'mean_std6.bin')
                goal_state1_rms.load_mean_std(path+'mean_std7.bin')
                achvd_obs_rms.load_mean_std(path+'mean_std8.bin')
                achvd_state0_rms.load_mean_std(path+'mean_std9.bin')
                achvd_state1_rms.load_mean_std(path+'mean_std10.bin')


        # s_t[0]: image, s_t[1]: position, s_t[2]: velocity, s_t[3]: effort, s_t[4]: object pose
        # reshape before normalization
        s_t[0] = np.reshape(s_t[0],(-1,100,100,3))
        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])

        # for update
        _rms_s_t = s_t[:]



        goal_obs_actor = np.reshape(goal_obs_actor,(-1,100,100,3))
        goal_state_critic[0] = goal_state_critic[0].reshape(1,goal_state_critic[0].shape[0])
        goal_state_critic[1] = goal_state_critic[1].reshape(1,goal_state_critic[1].shape[0])





        # print '=========================================================================='
        # print s_t[0].shape
        # print s_t[1].shape
        # print s_t[2].shape
        # print s_t[3].shape
        # print s_t[4].shape
        # print '=========================================================================='



        s_t[0] = normalize(s_t[0], s_t0_rms)
        s_t[1] = normalize(s_t[1], s_t1_rms)
        s_t[2] = normalize(s_t[2], s_t2_rms)
        s_t[3] = normalize(s_t[3], s_t3_rms)
        s_t[4] = normalize(s_t[4], s_t4_rms)
        # goal_obs_actor = normalize(goal_obs_actor, goal_obs_rms)






        goal_obs_actor_norm = normalize(goal_obs_actor, goal_obs_rms)
        _norm_goal_state_crit0 = normalize(goal_state_critic[0], goal_state0_rms)
        _norm_goal_state_crit1 = normalize(goal_state_critic[1], goal_state1_rms)

        goal_state_critic_norm = [_norm_goal_state_crit0, _norm_goal_state_crit1]




        # goal state critic is 7 joint positions 
        # goal_state_critic[0] = goal_state_critic[0].reshape(1,goal_state_critic[0].shape[0])
        # goal_state_critic[1] = goal_state_critic[1].reshape(1,goal_state_critic[1].shape[0])

        # Preparation for HER 
        # state, goal = episode.reset()
        # transition_store = []
        # done = False
        total_reward = 0.
        step = 0
        transition_store = [] # necessary for second loop
        # 1st Loop of HER, collect transitions just by doing actions # 1st Loop of HER, collect transitions just by doing actions #

        # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER
        
        start_time = time.time()
        for j in range(max_steps):

            a_t = np.zeros([1,action_dim])
            # execute e-Greedy behaviour policy
            ## Pseudo2: Obtain action using behavioural policy
            # print '=========================================================='
            # print 'Actor observation'
            # print s_t[0]
            # print 'Actor goal observation'
            # print goal_obs_actor_norm

            # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            # print s_t[1]







            a_t_original = actor.model.predict([s_t[0], s_t[1], goal_obs_actor_norm]) # state; image & position




            # a_t_original[0][0] = joint_vel_scaling[0]*a_t_original[0][0]
            # a_t_original[0][1] = joint_vel_scaling[1]*a_t_original[0][1]
            # a_t_original[0][2] = joint_vel_scaling[2]*a_t_original[0][2]
            # a_t_original[0][3] = joint_vel_scaling[3]*a_t_original[0][3]
            # a_t_original[0][4] = joint_vel_scaling[4]*a_t_original[0][4]
            # a_t_original[0][5] = joint_vel_scaling[5]*a_t_original[0][5]
            # a_t_original[0][6] = joint_vel_scaling[6]*a_t_original[0][6]

            _action_noise = action_noise()



            # self.state_rep = _concat(next_state, desired_goal)

            # _action_noise[0] = np.clip(_action_noise[0], -action_noise_ratio*abs(a_t_original[0][0]), action_noise_ratio*abs(a_t_original[0][0]))
            # _action_noise[1] = np.clip(_action_noise[1], -action_noise_ratio*abs(a_t_original[0][1]), action_noise_ratio*abs(a_t_original[0][1]))
            # _action_noise[2] = np.clip(_action_noise[2], -action_noise_ratio*abs(a_t_original[0][2]), action_noise_ratio*abs(a_t_original[0][2]))
            # _action_noise[3] = np.clip(_action_noise[3], -action_noise_ratio*abs(a_t_original[0][3]), action_noise_ratio*abs(a_t_original[0][3]))
            # _action_noise[4] = np.clip(_action_noise[4], -action_noise_ratio*abs(a_t_original[0][4]), action_noise_ratio*abs(a_t_original[0][4]))
            # _action_noise[5] = np.clip(_action_noise[5], -action_noise_ratio*abs(a_t_original[0][5]), action_noise_ratio*abs(a_t_original[0][5]))
            # _action_noise[6] = np.clip(_action_noise[6], -action_noise_ratio*abs(a_t_original[0][6]), action_noise_ratio*abs(a_t_original[0][6]))
            

            a_t[0] = a_t_original[0] + _action_noise*train_indicator # apply noise only for training
            # a_t[0] = a_t_original[0] + _action_noise # apply noise only for training


            a_t[0][0] = joint_vel_scaling[0]*a_t[0][0]
            a_t[0][1] = joint_vel_scaling[1]*a_t[0][1]
            a_t[0][2] = joint_vel_scaling[2]*a_t[0][2]
            a_t[0][3] = joint_vel_scaling[3]*a_t[0][3]
            a_t[0][4] = joint_vel_scaling[4]*a_t[0][4]
            a_t[0][5] = joint_vel_scaling[5]*a_t[0][5]
            a_t[0][6] = joint_vel_scaling[6]*a_t[0][6]

            # action clipping

            a_t[0][0] = np.clip(a_t[0][0], -0.16, 0.16)
            a_t[0][1] = np.clip(a_t[0][1], -0.07, 0.07)
            a_t[0][2] = np.clip(a_t[0][2], -0.2, 0.2)
            a_t[0][3] = np.clip(a_t[0][3], -0., 1.0)
            a_t[0][4] = np.clip(a_t[0][4], -1.0, 1.0)
            a_t[0][5] = np.clip(a_t[0][5], -1.0, 1.0)
            a_t[0][6] = np.clip(a_t[0][6], -1.0, 1.0)

          
            print '===================Actions======================='
            print (a_t[0],'@step', step )
            print 
            print '================================================='
            ## Pseudo3: Execute action, receive reward and transition
            # state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo
            # print a_t[0]

            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, achvd_pos, achvd_vel, achvd_color_obs, r_t, done = env.step(a_t[0], step, isReal=True) # 1st
            obj_state_t_1 = env.getObjPose(isReal=True)


            total_reward += np.average(r_t) # adds up averaged reward for each episode
            # achieved goal is returned every
            achvd_obs_actor = np.array(achvd_color_obs)
            achvd_state_critic = [np.array(achvd_pos), np.array(achvd_vel)]
            # s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1)]
            
            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]  



 
            s_t_1[0] = np.reshape(s_t_1[0],(-1,100,100,3))
            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])

            goal_obs_actor = np.reshape(goal_obs_actor,(-1,100,100,3))


            # copy for 
            _rms_s_t_1 = s_t_1[:]



            # goal_state_critic[0] = np.reshape(1,goal_state_critic[0].shape[0])
            # goal_state_critic[1] = np.reshape(1,goal_state_critic[1].shape[0])






            achvd_obs_actor = np.reshape(achvd_obs_actor,(-1,100,100,3))




            achvd_state_critic[0] = achvd_state_critic[0].reshape(1,achvd_state_critic[0].shape[0])
            achvd_state_critic[1] = achvd_state_critic[1].reshape(1,achvd_state_critic[1].shape[0])







            s_t_1[0] = normalize(s_t_1[0], s_t0_rms)
            s_t_1[1] = normalize(s_t_1[1], s_t1_rms)
            s_t_1[2] = normalize(s_t_1[2], s_t2_rms)
            s_t_1[3] = normalize(s_t_1[3], s_t3_rms)
            s_t_1[4] = normalize(s_t_1[4], s_t4_rms)
            # goal_obs_actor = normalize(goal_obs_actor, goal_obs_rms)
            # goal_obs_actor_norm = normalize(goal_obs_actor, goal_obs_rms)
            goal_obs_actor_norm = normalize(goal_obs_actor, goal_obs_rms)
            _norm_goal_state_crit0 = normalize(goal_state_critic[0], goal_state0_rms)
            _norm_goal_state_crit1 = normalize(goal_state_critic[1], goal_state1_rms)
            
            goal_state_critic_norm = [_norm_goal_state_crit0, _norm_goal_state_crit1]




            achvd_obs_actor = normalize(achvd_obs_actor, s_t0_rms)
            achvd_state_critic[0] = normalize(achvd_state_critic[0], s_t1_rms)
            achvd_state_critic[1] = normalize(achvd_state_critic[1], s_t2_rms)
 






            # achvd_state_critic[0] = achvd_state_critic[0].reshape(1,achvd_state_critic[0].shape[0])
            # achvd_state_critic[1] = achvd_state_critic[1].reshape(1,achvd_state_critic[1].shape[0])
            
            _g = np.concatenate((goal_state_critic_norm[0],goal_state_critic_norm[1]))
            _goal_critic = np.reshape(_g,(-1,14))

            _a = np.concatenate((achvd_state_critic[0],achvd_state_critic[1]))
            _achvd_critic = np.reshape(_a,(-1,14))



            # a_t[0] = normalize_action(a_t[0]) # Action normalizations
            s_t0_rms.update(_rms_s_t_1[0])
            s_t1_rms.update(_rms_s_t_1[1])
            s_t2_rms.update(_rms_s_t_1[2])
            s_t3_rms.update(_rms_s_t_1[3])
            s_t4_rms.update(_rms_s_t_1[4])                
            goal_obs_rms.update(goal_obs_actor)
            goal_state0_rms.update(goal_state_critic[0])
            goal_state1_rms.update(goal_state_critic[1])

            achvd_obs_rms.update(_rms_s_t_1[0])
            achvd_state0_rms.update(_rms_s_t_1[1])
            achvd_state1_rms.update(_rms_s_t_1[2])

            memory.append((s_t, s_t_1, a_t[0], r_t, achvd_obs_actor, _achvd_critic, done))
                        #    1    2       3      4            5           6          
            if len(memory) > (N_STEP_RETURN) and not isPretrain: # if buffer has more than 4 memories, execute
            # if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                st, st1, at, discount_r, achvd_obs_t, achvd_state_t, dn = memory.popleft()
                rt = discount_r
                for idx, ( _, _, _, ri, _, _, _) in enumerate(memory): # index & contents
                    discount_r += ri * GAMMA ** (idx + 1)
                buff.add(st, st1, at, discount_r, rt, s_t_1, _goal_critic , goal_obs_actor_norm, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                transition_store.append((st, st1, at, discount_r, rt, s_t_1, achvd_state_t, achvd_obs_t, _achvd_critic , achvd_obs_actor, dn, isDem)) # HER


            s_t = s_t_1[:]  # careful for copying list!!
            step +=1
            pretrain_count +=1

            if done:
                break
        # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER # First loop of HER
        print 'Acquired transition data'
        # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER
        env.set_learning_phase() # stops Sawyer action command for cont' control

        if len(transition_store) > 0 and not isPretrain and train_indicator: # set the last state as achieved goal and substitute
            print 'substitute reward computation starts'
            sbsttd_state_t1, sbsttd_obs_t1, sbsttd_state_tN, sbsttd_obs_tN = env.get_substitute_goal(transition_store) # extract achieved goals
            # print sbsttd_obs_tN.shape
            # Randomly sample trajectories from replay buffer
            


            for state, next_state, action, n_reward, reward, nstep_state, achvd_state_t1, acvhd_obs_t1, achvd_state_tN, achvd_obs_tN, dn, isDem in transition_store:
                assert sbsttd_state_t1[0].shape == achvd_state_t1[0].shape # assure that goal to be substituted has the same shape with substitute
                # print achvd_state_t1.shape

                sbsttd_r_1 = env.compute_substitute_reward(achvd_state_t1[0][0:7], sbsttd_state_tN[0][0:7]) # compute 1 step substitute reward
                sbsttd_r_n = env.compute_substitute_reward(achvd_state_tN[0][0:7], sbsttd_state_tN[0][0:7], isNstep=True) # compute n step substitute reward
                # substitute_reward_n = self.env.compute_reward(achieved_goal, substitute_goal)

                buff.add(state, next_state, action, n_reward, reward, nstep_state, sbsttd_state_tN, sbsttd_obs_tN, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                # buff.add(st, st1, at, discount_r, rt, s_t_1, goal_state_critic, goal_obs_actor, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
            # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER # Second loop of HER

            # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER # Third loop of HER 
        print 'A-C update session starts'
        for updates in range(MAX_UPDATES):
            if train_indicator:
                # Buffer index reference
                # state, next_state, action, n_reward, reward, nstep_state, sbsttd_state_tN, sbsttd_obs_tN, dn, isDem # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                #    0         1        2        3        4         5           6                   7       8      9
                batch, indices, weights_array = buff.getBatch(BATCH_SIZE)
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
                # Goals for HER # Goals for HER # Goals for HER # Goals for HER

                goal_states_Crit = np.vstack((np.array(e[6]) for e in batch)) # [0]: pos [1]: vel
                goal_states_Crit = np.vstack((np.array(e[6]) for e in batch)) # [0]: pos [1]: vel
                # print '=========================================='
                # print goal_states_Crit
                # print goal_states_Crit.shape
                # print goal_states_Crit[0].shape
                # print '=========================================='

                # print goal_states_Crit.shape
                # print goal_states_Crit[0].shape
                goal_states_Crit = np.reshape(goal_states_Crit,(-1,14))


                # print goal_states_Crit.shape
                goal_states_Actr = np.asarray([e[7] for e in batch]) # only visual observation
                
                goal_states_Actr = np.reshape(goal_states_Actr,(-1,100,100,3))


                # Goals for HER # Goals for HER # Goals for HER # Goals for HER

                dones = np.asarray([e[8] for e in batch])
                _isDemos = np.asarray([e[9] for e in batch])  
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




                # test with roll-out for n-value
                # full state = joint_pos_state +joint_vel_state
                
                # for n-step learning (BatchGD) to compute n-step loss for Prioritized exp.replay
                target_q_values_n, _  = critic.target_model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n,goal_states_Crit, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n, goal_states_Actr])])
                # for n-step learning (BatchGD)

                # for 1-step learning (SGD) to compute n-step loss for Prioritized exp.replay
                target_q_value_1, _ = critic.target_model.predict([reduced_full_state_t_1, reduced_sampled_obj_state_1, goal_states_Crit,actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1,goal_states_Actr])])

           

                # for n-step target
                for k in range(len(batch)):
                    if dones[k]:
                        y_t_n[k] = rewards_tn[k] # if terminal state exists
                    else:
                        y_t_n[k] = rewards_tn[k] + (GAMMA**N_STEP_RETURN)*target_q_values_n[k] ## target function update
                # for 1-step target
                for l in range(len(batch)):
                    if dones[l]:
                        y_t[l] = rewards[l] # if terminal state exists
                    else:
                        y_t[l] = rewards[l] + (GAMMA)*target_q_value_1[l] ## target function update



                if (train_indicator): #training mode
                    # loss += critic.model.train_on_batch([sampled_img_state_n, sampled_joint_pos_state_n, actions], y_t) #Minibatch loss -> MSE, and current estimation is made


                    # Compute Behavioural Cloning loss # Compute Behavioural Cloning loss 
                    # Condition; should be demo data and meet (q_values > target_q_value_1)

                    # action_pred_from_obs = actor.model.predict([sampled_img_state, sampled_joint_pos_state,goal_states_Actr])
                    # action_pred_from_obs = np.reshape(action_pred_from_obs,(BATCH_SIZE,7))

                    # q_values, q_val_from_pred = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, action_pred_from_obs])
                    # q_val_from_demo, _ = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions])
                    # q_values = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, actions])

                    # prepare items for Summary Writer


                    _isDemos = np.reshape(_isDemos,(BATCH_SIZE,1))
                    isDemos = np.ones((BATCH_SIZE,7),dtype=bool)

                    for x in range(BATCH_SIZE):
                        for y in range(7):
                            isDemos[x][y] *=_isDemos[x]  

                    # print 'How minibatch transitions are composed'
                    # print isDemos

                    isDemos = np.reshape(isDemos,(BATCH_SIZE,7))
                    isDemos_float = isDemos.astype(float)

                    # weights_array_n = weights_array_n.reshape((BATCH_SIZE,1))

                    y_t_n = y_t_n.reshape((BATCH_SIZE,1))
                    y_t = y_t.reshape((BATCH_SIZE,1))

                    # _loss_n = (y_t_n - q_val_from_pred)**2
                    # _loss_1 = (y_t - q_val_from_pred)**2

                    # mse_loss_n = _loss_n.mean()
                    # mse_loss_1 = _loss_1.mean()

                    # n_step_loss +=mse_loss_n
                    # one_step_loss +=mse_loss_1
                 
                    # # compute TD error
                    # # loss_array_n = weights_array_n * _loss_n
                    # # loss_array_1 =  weights_array_1* _loss_1


                    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                    # print y_t.shape
                    # print y_t_n.shape




                    # Train Critic # Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic
                    with tf.variable_scope('train_critic'):
                        # _mse_loss = critic.model.train_on_batch([reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions], {"one_step_Q": y_t, 
                        # "N_step_Q": y_t_n}) #Minibatch loss -> MSE, and current estimation is made
                        # Inputs: onestep_target, Nstep_target, state_full, state_obj, goal_state_critic
                

                        #  Return valeus = onestep_loss, Nstep_loss, onestep_td_err, Nstep_td_err, q_pred, l2_loss

                        # print '~~~~~~~~/~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                        # print weights/_array
                        # print weights_array.shape
                        # onestep_loss, Nstep_loss, td_err_1, td_err_n, q_pred, l2_loss = critic.train_critic_with_gradient(weights_array, y_t, y_t_n, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions)

                        onestep_loss, Nstep_loss, td_err_1, td_err_n, q_pred, l2_loss = critic.train_critic_with_flatgrad(weights_array, y_t, y_t_n, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions)
                        
                        # Actor is not related to this!!!

                        l2_reg_loss += l2_loss
                        one_step_loss += onestep_loss
                        n_step_loss += Nstep_loss


                        avg_q += np.average(q_pred)


                    # Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic# Train Critic

                    # We're about to use a method that directly acquires gradients of both actor and critic
                    # Tensors should flow between actor and critic
                    with tf.variable_scope('train_actor'):

                        actor.train_actor_with_flatgrad(sampled_img_state, sampled_joint_pos_state, goal_states_Actr, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions, isDemos_float, float(not isPretrain))


                    if not isPretrain:
                        buff.update_priorities(indices, td_err_n, td_err_1, _isDemos)

                    # fireloss_array
                    # _critic_weights = critic.model.get_weights()


                    total_loss = Nstep_loss + onestep_loss + l2_loss


                    # actor.train_actor(sampled_img_state, sampled_joint_pos_state, goal_states_Actr, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, bc_loss)
                    # actor.train_actor(sampled_img_state, sampled_joint_pos_state, goal_states_Actr, reduced_full_state_t, reduced_sampled_obj_state, goal_states_Crit, actions, isDemos_float)


                    # env.act_for_ctrl_period(a_t[0]) # for 10ms ctrl
                    actor.target_train()
                    # env.act_for_ctrl_period(a_t[0]) # 15
                    critic.target_train()
        env.unset_learning_phase() # stops Sawyer action command for cont' control
        if train_indicator:
            print 'Saves mean and std for this training session'
            s_t0_rms.save_mean_std(path+'mean_std0.bin')
            s_t1_rms.save_mean_std(path+'mean_std1.bin')
            s_t2_rms.save_mean_std(path+'mean_std2.bin')
            s_t3_rms.save_mean_std(path+'mean_std3.bin')
            s_t4_rms.save_mean_std(path+'mean_std4.bin')
            goal_obs_rms.save_mean_std(path+'mean_std5.bin')
            goal_state0_rms.save_mean_std(path+'mean_std6.bin')
            goal_state1_rms.save_mean_std(path+'mean_std7.bin')
            achvd_obs_rms.save_mean_std(path+'mean_std8.bin')
            achvd_state0_rms.save_mean_std(path+'mean_std9.bin')
            achvd_state1_rms.save_mean_std(path+'mean_std10.bin')


            elapsed_time = time.time() - start_time
            total_time +=elapsed_time

            print ("onestep_loss", onestep_loss /float(MAX_UPDATES), "Nstep_loss", Nstep_loss /float(MAX_UPDATES), "L2_loss", l2_loss /float(MAX_UPDATES))
            print("Episode", i,"Total_reward", total_reward, "Loss", total_loss, "Per-episode-Time", total_time)
            total_time = 0
               
                # Check for termination of each episode! & If true, record to Summary Writers

            # record stats for each episode
            # action_noise.reset() # reset OU process

            # l2_reg_loss
            # one_step_loss
            # n_step_loss

            stats = [total_reward, avg_q / float(MAX_UPDATES), step, l2_loss / float(MAX_UPDATES), one_step_loss / float(MAX_UPDATES),
             n_step_loss / float(MAX_UPDATES)]
            for f in range(len(stats)):
                sess.run(update_ops[f], feed_dict={summary_placeholders[f]: float(stats[f])})

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i + 1)
            actor.update_actor_summary(summary_writer = summary_writer,global_step = i + 1)
            critic.update_critic_summary(summary_writer = summary_writer,global_step = i + 1)
            # monitor critic weights
            # write_log(callback, critic_history_names, critic_log, i + 1)

            avg_q , l2_reg_loss, l2_loss, one_step_loss, n_step_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        

            # buffer_added = False

            if np.mod(i, 10) == 0:
                if (train_indicator):
                    print("Now we save model")
                    actor.model.save_weights(path+"weights_her/actormodel_"+str(i)+".h5", overwrite=True)
                    critic.model.save_weights(path+"weights_her/criticmodel_"+str(i)+".h5", overwrite=True)


            print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
            print("Total Step: " + str(step))
            print("")
            if (train_indicator):
                resultArray[i][0]=total_reward
                resultArray[i][1]=dist
                np.savetxt(path+'results.txt', resultArray)
                np.save(path+'results.npy', resultArray)
                actor.model.save_weights(path+"weights_her/actormodel_"+".h5", overwrite=True)
                critic.model.save_weights(path+"weights_her/criticmodel_"+".h5", overwrite=True)
   
   # End of episode loop ###############################################################
   #####################################################################################

    env.done()
    print("Finish.")


if __name__ == "__main__":
    playGame(0) # 1: train 0: test