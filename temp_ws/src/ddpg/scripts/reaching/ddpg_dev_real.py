#!/usr/bin/env python
import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import losses
import tensorflow as tf
from keras.engine.training import *
import json
import os
from PriorReplayBuffer_dev import ReplayBuffer
from ActorNetwork_dev import ActorNetwork
from CriticNetwork_dev import CriticNetwork
from new_robotGame import robotGame
import time
from tempfile import TemporaryFile
from OU import OU
from Gaussian import Gaussian
from running_mean_std import RunningMeanStd
from collections import deque
import rospy
import time
import pickle


path="/home/irobot/catkin_ws/src/ddpg/scripts/"


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def setup_summary():
    episode_total_reward = tf.Variable(0.)
    episode_q = tf.Variable(0.)
    episode_step = tf.Variable(0.)
    episode_l2_loss = tf.Variable(0.)
    episode_1step_loss = tf.Variable(0.)
    episode_nstep_loss = tf.Variable(0.)

    tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
    tf.summary.scalar('Avg_Q/Episode', episode_q)
    tf.summary.scalar('Took_Steps/Episode', episode_step)
    tf.summary.scalar('L2_Critic_Loss/Episode', episode_l2_loss)
    tf.summary.scalar('1step_Loss/Episode', episode_1step_loss)
    tf.summary.scalar('Nstep_Loss/Episode', episode_nstep_loss)

    summary_vars = [episode_total_reward, episode_q,
                    episode_step, episode_l2_loss, episode_1step_loss, episode_nstep_loss]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in
                            range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                  range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
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
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    SAMPLE_SIZE = 64
    GAMMA = 0.995
    TAU = 0.01     #Target Network HyperParameters
    LRA = 1e-4    #Learning rate for Actor
    LRC = 1e-3     #Lerning rate for Critic
    N_STEP_RETURN = 10

    action_dim = 7  #num of joints being controlled
    state_dim_rgb = [100,100,3]  #num of features in state refer to 'new_robotGame/reset'
    pos_dim_robot = 7  # 7 joint positions + gripper pos (last)
    vel_dim_robot = 7  # 7 joint positions + gripper pos (last)
    eff_dim_robot = 7  # 7 joint positions + gripper pos (last)
    full_dim_robot = 21  # 7 joint pos + 7 joint vels + joint efforts
    state_dim_object = 3  #num of features in state refer to 'new_robotGame/eset'
    episode_count = 4000 if (train_indicator) else 10
    resultArray=np.zeros((episode_count,2))
    max_steps = 500
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
    joint_vel_scale_base = 1.0
    joint_vel_scale_shoulder_1 = 1.0
    joint_vel_scale_shoulder_2 = 1.0
    joint_vel_scale_shoulder_3  = 1.0
    joint_vel_scale_wrist_1 =1.0
    joint_vel_scale_wrist_2 =1.0
    joint_vel_scale_wrist_3 =1.0
    joint_vel_scaling = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
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
    pretrain_steps = 100
    cur_step_count = 0
    avg_q = 0.0
    l2_reg_loss = 0.0
    one_step_loss = 0.0
    n_step_loss = 0.0
    isStart = False
    action_noise_ratio = 0.3
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    total_loss = 0
    test_episode = 400

    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    from keras import backend as K
    K.set_session(sess)
    # if(train_indicator == 1):
    K.set_learning_phase(1)

    critic = CriticNetwork(sess, full_dim_robot, state_dim_object, action_dim, BATCH_SIZE, TAU, LRC)

    # critic = CriticNetwork(sess, full_dim_robot, action_dim, BATCH_SIZE, TAU, LRC)


    actor = ActorNetwork(sess, state_dim_rgb, pos_dim_robot, action_dim, BATCH_SIZE, TAU, LRA, critic.state_full, critic.state_obj, critic.model)

    # action_noise = OU(mu=np.zeros(action_dim), sigma=float(0.05) * np.ones(action_dim))
    action_noise = Gaussian(mu=np.zeros(action_dim), sigma=float(0.02) * np.ones(action_dim))



    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = robotGame()

    summary_placeholders, update_ops, summary_op = setup_summary()
    summary_writer = tf.summary.FileWriter(
            path+'reaching/summary/ddpg_reacher', sess.graph)

    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts/reaching')

    if os.path.exists(dagger_data) and train_indicator:
        print 'initialize the replay buffer with demo data'
        with open(dagger_data, 'rb') as f:
            dagger = pickle.load(f)
            for idx, item in enumerate(dagger):
                if idx == 10:
                    print item
                buff.add(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7])
            print (idx, 'data has retrieved')

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(path+"weights/actormodel_"+test_episode+".h5")
        critic.model.load_weights(path+"weights/criticmodel_"+test_episode+".h5")
        actor.target_model.load_weights(path+"weights/actormodel_"+test_episode+".h5")
        critic.target_model.load_weights(path+"weights/criticmodel_"+test_episode+".h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    pretrain_count = 0 
    for i in range(episode_count): # pseudo: for each episode
        if i ==0:
            isStart = True
        else:
            isStart = False

        rospy.loginfo("Now It's EPISODE{0}".format(i))

        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t = env.reset(isStart=isStart) #(self.color_obs, self.joint_values)

        obj_state_t = env.getObjPose()

        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t), np.array(obj_state_t)]


        if i == 0:
            s_t0_rms = RunningMeanStd(shape=s_t[0].shape)
            s_t1_rms = RunningMeanStd(shape=s_t[1].shape)
            s_t2_rms = RunningMeanStd(shape=s_t[2].shape)
            s_t3_rms = RunningMeanStd(shape=s_t[3].shape)
            s_t4_rms = RunningMeanStd(shape=s_t[4].shape)


        # s_t[0]: image, s_t[1]: position, s_t[2]: velocity, s_t[3]: effort, s_t[4]: object pose

        s_t[0] = normalize(s_t[0], s_t0_rms)
        s_t[1] = normalize(s_t[1], s_t1_rms)
        s_t[2] = normalize(s_t[2], s_t2_rms)
        s_t[3] = normalize(s_t[3], s_t3_rms)
        s_t[4] = normalize(s_t[4], s_t4_rms)


        # s_t[1] = (s_t[1] - s_t[1].mean())/s_t[1].std()
        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        # s_t[2] = (s_t[2] - s_t[2].mean())/s_t[2].std()
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        # s_t[3] = (s_t[3] - s_t[3].mean())/s_t[3].std()
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        # s_t[4] = (s_t[4] - s_t[4].mean())/s_t[4].std()
        s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])


        total_reward = 0.
        step = 0
        for j in range(max_steps): # pseudo: for each timestep
            start_time = time.time()
            # initialize variables  

            mse_loss = 0
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])


            # Assymetric network, actor requires RGB observation + 
            # 1D array input should be re-shaped
            # s_t.reshape(1, s_t.shape[0learning_rate])
            s_t[0] = np.reshape(s_t[0],(-1,100,100,3))

            '''
            default_joint_tolerance = 0.05  # rad
            default_max_linear_speed = 0.6  # m/s
            default_max_linear_accel = 0.6  # m/s/s
            default_max_rot_speed = 1.57  # rad/s
            default_max_rot_accel = 1.57  # rad/s/s
            '''

            a_t_original = actor.model.predict([s_t[0], s_t[1]]) # state; image & position
            if  np.mod(j, 10) == 0:
                print ('action @ step%d' %j)
                print a_t_original


            a_t_original[0][0] = joint_vel_scaling[0]*a_t_original[0][0]
            a_t_original[0][1] = joint_vel_scaling[1]*a_t_original[0][1]
            a_t_original[0][2] = joint_vel_scaling[2]*a_t_original[0][2]
            a_t_original[0][3] = joint_vel_scaling[3]*a_t_original[0][3]
            a_t_original[0][4] = joint_vel_scaling[4]*a_t_original[0][4]
            a_t_original[0][5] = joint_vel_scaling[5]*a_t_original[0][5]
            a_t_original[0][6] = joint_vel_scaling[6]*a_t_original[0][6]

            _action_noise = action_noise()


            _action_noise[0] = np.clip(_action_noise[0], -action_noise_ratio*abs(a_t_original[0][0]), action_noise_ratio*abs(a_t_original[0][0]))
            _action_noise[1] = np.clip(_action_noise[1], -action_noise_ratio*abs(a_t_original[0][1]), action_noise_ratio*abs(a_t_original[0][1]))
            _action_noise[2] = np.clip(_action_noise[2], -action_noise_ratio*abs(a_t_original[0][2]), action_noise_ratio*abs(a_t_original[0][2]))
            _action_noise[3] = np.clip(_action_noise[3], -action_noise_ratio*abs(a_t_original[0][3]), action_noise_ratio*abs(a_t_original[0][3]))
            _action_noise[4] = np.clip(_action_noise[4], -action_noise_ratio*abs(a_t_original[0][4]), action_noise_ratio*abs(a_t_original[0][4]))
            _action_noise[5] = np.clip(_action_noise[5], -action_noise_ratio*abs(a_t_original[0][5]), action_noise_ratio*abs(a_t_original[0][5]))
            _action_noise[6] = np.clip(_action_noise[6], -action_noise_ratio*abs(a_t_original[0][6]), action_noise_ratio*abs(a_t_original[0][6]))

            a_t[0] = a_t_original[0] + _action_noise*train_indicator # apply noise only for training



            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, r_t, done = env.step(a_t[0], step) # 1st
            obj_state_t_1 = env.getObjPose()
            # s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1)]
            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]


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
            # env.act_for_ctrl_period(a_t[0]) # 2nd


            # N-step return implementation!!
            # Keep the experience in memory until 'N_STEP_RETURN' steps has
            # passed to get the delayed return r_1 + ... + gamma^n r_n
            # for N-step return
            memory.append((s_t, s_t_1, a_t[0], r_t, done))

            if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                st, st1, at, discount_r, dn = memory.popleft()
                rt = discount_r
                for idx, (si, s1i, ai, ri, di) in enumerate(memory):
                    # env.act_for_ctrl_period(a_t[0]) # 3, 4, 5, 6, 7
                    discount_r += ri * GAMMA ** (idx + 1)

 
                buff.add(st, st1, at, discount_r, rt, s_t_1, dn, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                s_t0_rms.update(s_t[0])
                s_t1_rms.update(s_t[1])
                s_t2_rms.update(s_t[2])
                s_t3_rms.update(s_t[3])
                s_t4_rms.update(s_t[4])





                buffer_added = True

         
            if buffer_added and train_indicator:
            # if buffer_added and train_indicator:

                # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                # print buffer_added   
                batch, indices, weights_array_1 = buff.getBatch(BATCH_SIZE)
                weights_array_n = weights_array_1
                # find loss

                states_t = ([e[0] for e in batch])

                states_t1 = ([e[1] for e in batch])
                states_tn = ([e[5] for e in batch])
                actions = np.asarray([e[2] for e in batch])
                rewards = np.asarray([e[4] for e in batch])
                rewards_tn = np.asarray([e[3] for e in batch])             
                dones = np.asarray([e[6] for e in batch])
                isDemos = np.asarray([e[7] for e in batch])  
                # Check if shape matches ?!
                y_t_n = np.asarray([e[2][0] for e in batch])
                y_t = np.asarray([e[2][0] for e in batch])



                '''         
                    How state list comnsists of
            
                [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), 
                np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]


                '''
               
                # state values (current)
                sampled_img_state = np.array([img[0] for img in states_t])
                sampled_img_state = np.reshape(sampled_img_state,(-1,100,100,3))

                sampled_joint_pos_state = np.array([pos[1] for pos in states_t])
                sampled_joint_vel_state = np.array([vel[2] for vel in states_t])

                sampled_joint_eff_state = np.array([eff[3] for eff in states_t])
                sampled_obj_state = np.array([obj[4] for obj in states_t])
                # now reduce/reshape states to feed on neural networks
                full_state_t = np.concatenate((sampled_joint_pos_state, sampled_joint_vel_state, sampled_joint_eff_state))
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
                full_state_t_n = np.concatenate((sampled_joint_pos_state_n, sampled_joint_vel_state_n, sampled_joint_eff_state_n))
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
                full_state_t_1 = np.concatenate((sampled_joint_pos_state_1, sampled_joint_vel_state_1, sampled_joint_eff_state_1))
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
                target_q_values_n, _  = critic.target_model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])])
                # for n-step learning (BatchGD)

                # for 1-step learning (SGD) to compute n-step loss for Prioritized exp.replay
                target_q_value_1, _ = critic.target_model.predict([reduced_full_state_t_1, reduced_sampled_obj_state_1, actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1])])


                q_values, _ = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, actions])
                # q_values = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, actions])

                # prepare items for Summary Writer
                avg_q += np.average(q_values)
            

                # Numpy array type
                # print rewards.shape
                # print '==============================================='
                # print target_q_values[0]
                # print '==============================================='
                # print target_q_values[1]
                # print '==============================================='

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



                # print '========================================='

                # print 'Q estimation'
                # print q_values
                # print '========================================='

                mean_reward = r_t
                show_reward = True

                if (train_indicator): #training mode
                    # loss += critic.model.train_on_batch([sampled_img_state_n, sampled_joint_pos_state_n, actions], y_t) #Minibatch loss -> MSE, and current estimation is made

                    # batch, indices, weights_array
                    weights_array_n = weights_array_n.reshape((BATCH_SIZE,1))
                    weights_array_1 = weights_array_1.reshape((BATCH_SIZE,1))


                    y_t_n = y_t_n.reshape((BATCH_SIZE,1))
                    y_t = y_t.reshape((BATCH_SIZE,1))

                    _loss_n = (y_t_n - q_values)**2
                    _loss_1 = (y_t - q_values)**2



                    mse_loss_n = _loss_n.mean()
                    mse_loss_1 = _loss_1.mean()

                    n_step_loss +=mse_loss_n
                    one_step_loss +=mse_loss_1
                 
                    # compute TD error
                    loss_array_n = weights_array_n * _loss_n
                    loss_array_1 =  weights_array_1* _loss_1  
                    loss_array = loss_array_1 + loss_array_n + 1e-6*np.ones((SAMPLE_SIZE,1))

                    '''
                        Q_n = K.placeholder(shape=(None,), dtype='float32')        
                        y_target_n = K.placeholder(shape=(None,), dtype='float32')
                        Q_1 = K.placeholder(shape=(None,), dtype='float32')
                        fy_target_1 = K.placeholder(shape=(None,), dtype='float32')

                    '''
                    # [n_step_states, actions, ]
                    # loss = critic.optimizer([q_values_n, target_q_values_n, q_values, target_q_values])
                    buff.update_priorities(indices, loss_array_n, loss_array_1)
                    # env.act_for_ctrl_period(a_t[0]) # 12
                    # MSE loss
                    # Re-implement the loss!

                    '''
                        Function
                            keras.backend.function(inputs, outputs, updates=None)
                            Instantiates a Keras function.

                            Arguments

                            inputs: List of placeholder tensors.
                            outputs: List of output tensors.`
                            updates: List of update ops.
                            **kwargs: Passed to tf.Session.run.

                        Returns

                            Output values as Numpy arrays.

                   y: Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. 
                   If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.

                    '''


                    _mse_loss = critic.model.train_on_batch([reduced_full_state_t, reduced_sampled_obj_state, actions], {"one_step_Q": y_t, 
                   "N_step_Q": y_t_n}) #Minibatch loss -> MSE, and current estimation is made

                    l2_loss =_mse_loss[0]

                    total_loss = sum(_mse_loss)

                    l2_reg_loss +=l2_loss

                    # Shouldn't it be scalar loss?
                    # from states, and acitons ( sampled batch states and actions)
                    # Keras, Sequential, train_on_batch(self, x, y, sample_weight=None, class_weight=None)

                    # a_for_grad = actor.model.predict([sampled_img_state, sampled_joint_pos_state])   # action gradient

                    # states_rgb, states_rob, critic_full_input, critic_obj_input
                    # actor.train_actor(sampled_img_state, sampled_joint_pos_state, reduced_full_state_t, reduced_sampled_obj_state, a_for_grad)
                    actor.train_actor(sampled_img_state, sampled_joint_pos_state, reduced_full_state_t, reduced_sampled_obj_state)

                    # env.act_for_ctrl_period(a_t[0]) # 14
                    # q_for_actor, _ = critic.model.predict([reduced_full_state_t,  a_for_grad])



                    # val = actor.actor_train_fn([sampled_img_state, sampled_joint_pos_state, reduced_full_state_t,  a_for_grad])
                    # val = actor.actor_train_fn([sampled_img_state, sampled_joint_pos_state, reduced_full_state_t, reduced_sampled_obj_state,  a_for_grad])


                    # env.act_for_ctrl_period(a_t[0]) # for 10ms ctrl
                    actor.target_train()
                    # env.act_for_ctrl_period(a_t[0]) # 15
                    critic.target_train()
                    # env.act_for_ctrl_period(a_t[0]) # 16
            elapsed_time = time.time() - start_time
            total_time +=elapsed_time

            # buffer_added = False



            if show_reward:
                total_reward += mean_reward
            s_t = s_t_1
            if done:
                break

            if np.mod(j, 10) == 0:
                print("Episode", i, "Step", step, "Action", a_t, "Total_reward", total_reward, "Loss", total_loss, "10step-Time", total_time)
                total_time = 0
           
            # Check for termination of each episode! & If true, record to Summary Writers
            step += 1
            pretrain_count += 1

        # record stats for each episode
        # action_noise.reset() # reset OU process
        stats = [total_reward, avg_q / float(step), step, l2_reg_loss / float(step), one_step_loss / float(step),
         n_step_loss / float(step)]
        for f in range(len(stats)):
            sess.run(update_ops[f], feed_dict={summary_placeholders[f]: float(stats[f])})
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, i + 1)
        avg_q , l2_reg_loss, l2_loss, one_step_loss, n_step_loss, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # buffer_added = False

        if np.mod(i, 10) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights(path+"weights/actormodel_"+str(i)+".h5", overwrite=True)
                critic.model.save_weights(path+"weights/criticmodel_"+str(i)+".h5", overwrite=True)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        if (train_indicator):
            resultArray[i][0]=total_reward
            resultArray[i][1]=dist
            np.savetxt(path+'results.txt', resultArray)
            np.save(path+'results.npy', resultArray)
            actor.model.save_weights(path+"weights/actormodel_"+".h5", overwrite=True)
            critic.model.save_weights(path+"weights/criticmodel_"+".h5", overwrite=True)
       

    env.done()
    print("Finish.")

    def _get_transition(self, idx):
        transition = [None] * (self.history + self.n)
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
          if transition[t + 1].timestep == 0:
            transition[t] = blank_trans  # If future frame has timestep 0
          else:
            transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
          if transition[t - 1].nonterminal:
            transition[t] = self.transitions.get(idx - self.history + 1 + t)
          else:
            transition[t] = blank_trans  # If prev (next) frame is terminal
        return transition



    # def discount_correct_rewards(r, gamma=GAMMA):
    #   """ take 1D float array of rewards and compute discounted reward """
    #   discounted_r = np.zeros_like(r)
    #   running_add = 0
    #   for t in reversed(range(0, r.size)):
    #     #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    #     running_add = running_add * gamma + r[t]
    #     discounted_r[t] = running_add

    #   discounted_r -= discounted_r.mean()
    #   discounted_r /- discounted_r.std()
    #   return discounted_r


if __name__ == "__main__":
    playGame(1) # 1: train 0: test