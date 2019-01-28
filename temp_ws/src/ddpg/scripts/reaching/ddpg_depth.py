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
from collections import deque
import rospy
import time
import pickle

OU = OU()


path="/home/irobot/catkin_ws/src/ddpg/scripts/"


def setup_summary():
    episode_total_reward = tf.Variable(0.)
    episode_q = tf.Variable(0.)
    episode_step = tf.Variable(0.)
    episode_avg_loss = tf.Variable(0.)

    tf.summary.scalar('Total Reward/Episode', episode_total_reward)
    tf.summary.scalar('Avg Q/Episode', episode_q)
    tf.summary.scalar('Took Steps/Episode', episode_step)
    tf.summary.scalar('Avg Loss/Episode', episode_avg_loss)

    summary_vars = [episode_total_reward, episode_q,
                    episode_step, episode_avg_loss]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in
                            range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                  range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op





#path ="/media/dalinel/Maxtor/ddpg/"
def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    SAMPLE_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 1e-4    #Learning rate for Actor
    LRC = 1e-3     #Lerning rate for Critic
    N_STEP_RETURN = 5

    action_dim = 7  #num of joints being controlled
    state_dim_rgb = [100,100,3]  #num of features in state refer to 'new_robotGame/reset'
    pos_dim_robot = 7  # 7 joint positions + gripper pos (last)
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
    joint_vel_scale_base = 5.
    joint_vel_scale_shoulder_1 = 5.
    joint_vel_scale_shoulder_2 = 5.
    joint_vel_scale_shoulder_3 = 5.
    joint_vel_scale_wrist_1 =5.
    joint_vel_scale_wrist_2 =5.
    joint_vel_scale_wrist_3 =5.
    joint_torque_scale_base = 20
    joint_torque_scale_shoulder_1 = 10
    joint_torque_scale_shoulder_2 = 5
    joint_torque_scale_shoulder_3 = 5.0
    joint_torque_scale_wrist_1 =5
    joint_torque_scale_wrist_2 =5
    joint_torque_scale_wrist_3 =5
    memory = deque()
    show_reward = False
    total_time = 0
    dagger_data = 'traj_dagger.bin'
    isDem = False
    pretrain_steps = 3000
    avg_q = 0.0
    avg_loss = 0.0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)
    from keras import backend as K
    K.set_session(sess)
    # if(train_indicator == 1):
    K.set_learning_phase(1)

    # critic = CriticNetwork(sess, full_dim_robot, state_dim_object, action_dim, BATCH_SIZE, TAU, LRC)
    critic = CriticNetwork(sess, full_dim_robot, action_dim, BATCH_SIZE, TAU, LRC)

    actor = ActorNetwork(sess, state_dim_rgb, pos_dim_robot, action_dim, BATCH_SIZE, TAU, LRA, critic.model.output)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = robotGame()

    summary_placeholders, update_ops, summary_op = setup_summary()
    summary_writer = tf.summary.FileWriter(
            path+'reaching/summary/ddpg_reacher', sess.graph)

    os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts/reaching')
    if os.path.exists(dagger_data) and train_indicator:
        # pass
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
        actor.model.load_weights(path+"weights/actormodel_.h5")
        critic.model.load_weights(path+"weights/criticmodel_.h5")
        actor.target_model.load_weights(path+"weights/actormodel_.h5")
        critic.target_model.load_weights(path+"weights/criticmodel_.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    pretrain_count = 0 
    for i in range(episode_count): # pseudo: for each episode

        rospy.loginfo("Now It's EPISODE{0}".format(i))
        color_obs_t, joint_pos_t, joint_vels_t, joint_efforts_t = env.reset() #(self.color_obs, self.joint_values)
        # color_obs_t = np.expand_dims(color_obs_t, axis=0)
        obj_state_t = env.getObjPose()

        # s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t), np.array(obj_state_t)]
        s_t = [np.array(color_obs_t), np.array(joint_pos_t), np.array(joint_vels_t), np.array(joint_efforts_t)]


        # print s_t[0]
        # normalized obs
        s_t[1] = (s_t[1] - s_t[1].mean())/s_t[1].std()
        s_t[1] = s_t[1].reshape(1,s_t[1].shape[0])
        s_t[2] = (s_t[2] - s_t[2].mean())/s_t[2].std()
        s_t[2] = s_t[2].reshape(1,s_t[2].shape[0])
        s_t[3] = (s_t[3] - s_t[3].mean())/s_t[3].std()
        s_t[3] = s_t[3].reshape(1,s_t[3].shape[0])
        # s_t[4] = (s_t[4] - s_t[4].mean())/s_t[4].std()
        # s_t[4] = s_t[4].reshape(1,s_t[4].shape[0])


        total_reward = 0.
        step = 0
        # is this part necessary????
        for j in range(max_steps): # pseudo: for each timestep
            pretrain_count +=1
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

            a_t_original = actor.model.predict([s_t[0], s_t[1]]) #r prediction (action) from CNN

            if  np.mod(j, 10) == 0:
                print ('action @ step%d' %j)
                print a_t_original
            # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 1.00, 0.05) # mu, theta, sigma
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.0 , 1.00, 0.20)
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.0 , 1.00, 0.20)
            # noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3],  0.0 , 1.00, 0.20)
            # noise_t[0][4] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][4],  0.1 , 0.6, 0.30)
            # noise_t[0][5] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][5],  0.1 , 0.6, 0.30)
            # noise_t[0][6] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][6],  0.1 , 0.6, 0.30)
            # noise_t[0][7] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.0 , 1.00, 0.15) # for gripper

                # a_type = "Explore"
                # a_t = np.random.uniform(-1,1, size=(1,action_dim))



            a_t[0][0] = joint_vel_scale_base * OU.function(a_t_original[0][0], 0.0 , .15, 0.3) #x_prev,  mu, theta, sigma
            a_t[0][1] = joint_vel_scale_shoulder_1 * OU.function(a_t_original[0][1], 0.0 , 0.15, 0.30)
            a_t[0][2] = joint_vel_scale_shoulder_2 * OU.function(a_t_original[0][2], 0.0 , 0.15, 0.30)
            a_t[0][3] = joint_vel_scale_shoulder_3 * OU.function(a_t_original[0][3], 0.0 , 0.15, 0.30)
            a_t[0][4] = joint_vel_scale_wrist_1 * OU.function(a_t_original[0][4], 0.0 , .15, 0.10)
            a_t[0][5] = joint_vel_scale_wrist_2 * OU.function(a_t_original[0][5], 0.0 , .15, 0.20)
            a_t[0][6] = joint_vel_scale_wrist_3 * OU.function(a_t_original[0][6], 0.0 , .15, 0.10)

            # AVERAGE DEMO JOINT TORQUE SCALE
            # -0.4297748957   -19.2295545455  -1.8767223073   -9.8073163636   2.1041193939    -0.1431800796   -0.0262954771
            # np.clip(action, self.action_range[0], self.action_range[1])
            # a_t[0][0] = joint_torque_scale_base * OU.function(a_t_original[0][0], 0.0 , 0.15, 0.2) #x_prev,  mu, theta, sigma
            # a_t[0][1] = joint_torque_scale_shoulder_1 * OU.function(a_t_original[0][1], 0.0 , 0.15, 0.20)
            # a_t[0][2] = joint_torque_scale_shoulder_2 * OU.function(a_t_original[0][2], 0.0 , 0.15, 0.20)
            # a_t[0][3] = joint_torque_scale_shoulder_3 * OU.function(a_t_original[0][3], 0.0 , 0.15, 0.20)
            # a_t[0][4] = joint_torque_scale_wrist_1 * OU.function(a_t_original[0][4], 0.0 , 0.15, 0.20)
            # a_t[0][5] = joint_torque_scale_wrist_2 * OU.function(a_t_original[0][5], 0.0 , 0.15, 0.20)
            # a_t[0][6] = joint_torque_scale_wrist_3 * OU.function(a_t_original[0][6], 0.0 , 0.15, 0.20)
            
            # a_t[0][0] = np.clip(a_t[0][0], -1, 1)
            # a_t[0][1] = np.clip(a_t[0][1], -20, 20.0)
            # a_t[0][2] = np.clip(a_t[0][2], -2.0, 2.0)
            # a_t[0][3] = np.clip(a_t[0][3], -10, 10)
            # a_t[0][4] = np.clip(a_t[0][4], -2.5, 2.5)
            # a_t[0][5] = np.clip(a_t[0][5], -0.15, 0.15)
            # a_t[0][6] = np.clip(a_t[0][6], -0.05, 0.05)


            
            # a_t[0][7] = a_t_original[0][7] + noise_t[0][7]

            # distance,color_obs, tjv, reward, done
            # pseudo: Execute action a_t and observe reward r_t and observe new state s_t+1


            dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, r_t, done = env.step(a_t[0], step)
            obj_state_t_1 = env.getObjPose()
            s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1)]
            # s_t_1 = [np.array(color_obs_t_1), np.array(joint_pos_t_1), np.array(joint_vels_t_1), np.array(joint_efforts_t_1),  np.array(obj_state_t_1)]

            # normalized obs
            s_t_1[1] = (s_t_1[1] - s_t_1[1].mean())/s_t_1[1].std()
            s_t_1[1] = s_t_1[1].reshape(1,s_t_1[1].shape[0])
            s_t_1[2] = (s_t_1[2] - s_t_1[2].mean())/s_t_1[2].std()
            s_t_1[2] = s_t_1[2].reshape(1,s_t_1[2].shape[0])
            s_t_1[3] = (s_t_1[3] - s_t_1[3].mean())/s_t_1[3].std()
            s_t_1[3] = s_t_1[3].reshape(1,s_t_1[3].shape[0])
            # s_t_1[4] = (s_t_1[4] - s_t_1[4].mean())/s_t_1[4].std()
            # s_t_1[4] = s_t_1[4].reshape(1,s_t_1[4].shape[0])


            s_t_1[0] = np.reshape(s_t_1[0],(-1,100,100,3))



            # N-step return implementation!!
            # Keep the experience in memory until 'N_STEP_RETURN' steps has
            # passed to get the delayed return r_1 + ... + gamma^n r_n
            # for N-step return
            memory.append((s_t, s_t_1, a_t[0], r_t))

            if len(memory) > (N_STEP_RETURN): # if buffer has more than 4 memories, execute
                s_t, s_t1, a_t, discount_r = memory.popleft()
                r_t = discount_r
                for idx, (si, s1i, ai, ri) in enumerate(memory):

                    discount_r += ri * GAMMA ** (idx + 1)

                # self.buffer.add((s_mem, a_mem, discount_r, s_, 1 if not done else 0))
                # if pretrain_count>=pretrain_steps: # for some timesteps, just train with demo data
                buff.add(s_t, s_t1, a_t, discount_r, r_t, s_t_1, done, isDem) # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                buffer_added = True

            # print buff.count()
            # pseudo: store transition (s_t, a_t, r_t s_t+1 ) in Replay buffer R

            # pseudo: sample a random minibatch of N trainsitions
            # return samples, indices, np.array(weights, dtype=np.float32) # returns a list()



            if not train_indicator:
                rospy.sleep(0.05)
            if buffer_added and train_indicator:

                # (state, next_state, action, nstep_reward, reward, nstep_state, isDone, isDemo)
                # print buffer_added  
                batch, indices, weights_array = buff.getBatch(BATCH_SIZE)
                index = indices
                weight_array = weights_array

                states_t = ([e[0] for e in batch])

                states_t1 = ([e[1] for e in batch])
                states_tn = ([e[5] for e in batch])
                actions = np.asarray([e[2] for e in batch])
                rewards = np.asarray([e[4] for e in batch])
                rewards_tn = np.asarray([e[3] for e in batch])             
                dones = np.asarray([e[6] for e in batch])
                isDemos = np.asarray([e[7] for e in batch])  


                y_t_n = np.asarray([e[2][0] for e in batch])
                y_t= np.asarray([e[2][0] for e in batch])



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
                # sampled_obj_state = np.array([obj[4] for obj in states_t])
                # now reduce/reshape states to feed on neural networks
                full_state_t = np.concatenate((sampled_joint_pos_state, sampled_joint_vel_state, sampled_joint_eff_state))
                reduced_full_state_t = np.reshape(full_state_t,(-1,21))
                #-----------------------------------------------------------------------#
                # reduced_sampled_obj_state = np.reshape(sampled_obj_state,(-1,3))
                sampled_joint_pos_state = np.reshape(sampled_joint_pos_state,(-1,7))
                # give the actor info of efforts rather tan position
                sampled_joint_eff_state = np.reshape(sampled_joint_eff_state,(-1,7))
                # state values (current)



                # For n-step returns # For n-step returns # For n-step returns 
                sampled_img_state_n = np.array([imgn[0] for imgn in states_tn])
                sampled_img_state_n = np.reshape(sampled_img_state_n,(-1,100,100,3))

                sampled_joint_pos_state_n = np.array([posn[1] for posn in states_tn])
                sampled_joint_vel_state_n = np.array([veln[2] for veln in states_tn])
                sampled_joint_eff_state_n = np.array([effn[3] for effn in states_tn])
                # sampled_obj_state_n = np.array([objn[4] for objn in states_tn])
                #-------------------------------------------------------------------------
                full_state_t_n = np.concatenate((sampled_joint_pos_state_n, sampled_joint_vel_state_n, sampled_joint_eff_state_n))
                reduced_full_state_t_n = np.reshape(full_state_t_n,(-1,21))
                #-----------------------------------------------------------------------#
                # reduced_sampled_obj_state_n = np.reshape(sampled_obj_state_n,(-1,3))
                sampled_joint_pos_state_n = np.reshape(sampled_joint_pos_state_n,(-1,7))
                # give the actor info of efforts rather tan position
                sampled_joint_eff_state_n = np.reshape(sampled_joint_eff_state_n,(-1,7))
                # For n-step returns # For n-step returns # For n-step returns


                # For 1-step returns # For 1-step returns # For 1-step returns 
                sampled_img_state_1 = np.array([img1[0] for img1 in states_t1])
                sampled_img_state_1 = np.reshape(sampled_img_state_1,(-1,100,100,3))

                sampled_joint_pos_state_1 = np.array([pos1[1] for pos1 in states_t1])
                sampled_joint_vel_state_1 = np.array([vel1[2] for vel1 in states_t1])
                sampled_joint_eff_state_1 = np.array([eff1[3] for eff1 in states_t1])
                # sampled_obj_state_1 = np.array([obj1[4] for obj1 in states_t1])
                #-------------------------------------------------------------------------
                full_state_t_1 = np.concatenate((sampled_joint_pos_state_1, sampled_joint_vel_state_1, sampled_joint_eff_state_1))
                reduced_full_state_t_1 = np.reshape(full_state_t_1,(-1,21))
                #-----------------------------------------------------------------------#
                # reduced_sampled_obj_state_1 = np.reshape(sampled_obj_state_1,(-1,3))
                sampled_joint_pos_state_1 = np.reshape(sampled_joint_pos_state_1,(-1,7))
                # give the actor info of efforts rather tan position
                sampled_joint_eff_state_1 = np.reshape(sampled_joint_eff_state_1,(-1,7))
                # For 1-step returns # For 1-step returns # For 1-step returns 


                '''
                Arguments

                x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
                batch_size: Integer. If unspecified, it will default to 32.
                verbose: Verbosity mode, 0 or 1.
                steps: Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.
                Returns

                Numpy array(s) of prediction
                '''
                # test with roll-out for n-value
                # full state = joint_pos_state +joint_vel_state
                # buff.add(s_t, a_t, discount_r, r_t, s_t_1, done, isDemo)      #A s_t , s_t+N, a_t+N, disc_r
                # target_q_values_n = critic.target_model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])])
                
                # for n-step learning (BatchGD) to compute n-step loss for Prioritized exp.replay
                # target_q_values_n, _  = critic.target_model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])])
                target_q_values_n, _  = critic.target_model.predict([reduced_full_state_t_n, actor.target_model.predict([sampled_img_state_n, sampled_joint_pos_state_n])])
                # target_q_values, _ = critic.target_model.predict([reduced_full_state_t, reduced_sampled_obj_state, actor.target_model.predict([sampled_img_state, sampled_joint_pos_state])])
                # q_values_n = critic.model.predict([reduced_full_state_t_n, reduced_sampled_obj_state_n, actions])
                # for n-step learning (BatchGD)

                # for 1-step learning (SGD) to compute n-step loss for Prioritized exp.replay
                # target_q_value_1, _ = critic.target_model.predict([reduced_full_state_t_1, reduced_sampled_obj_state_1, actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1])])
                target_q_value_1, _ = critic.target_model.predict([reduced_full_state_t_1, actor.target_model.predict([sampled_img_state_1, sampled_joint_pos_state_1])])
                # target_q_value = critic.target_model.predict([reduced_full_state_t_ro, reduced_sampled_obj_state_ro, actor.target_model.predict([sampled_img_state_ro, sampled_joint_pos_state_ro])])
                # q_values, _ = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, actions])
                q_values, _ = critic.model.predict([reduced_full_state_t,  actions])





                # prepare items for Summary Writer
                avg_q += np.amax(q_values)
                # avg_q += np.amax(q_value)
                # avg_q = avg_q / 2.0
                



                # Numpy array type
                # print rewards.shape
                # print '==============================================='
                # print target_q_values[0]
                # print '==============================================='
                # print target_q_values[1]
                # print '==============================================='


                for k in range(len(batch)):
                    if dones[k]:
                        y_t_n[k] = rewards_tn[k] # if terminal state exists
                    else:
                        y_t_n[k] = rewards_tn[k] + (GAMMA**N_STEP_RETURN)*target_q_values_n[k] ## target function update

                # sample, index, weight_array = buff.getBatch(1)
                # for l in range(len(sample)):
                #     if done_[l]:
                #         y_t[l] = reward[l] # if terminal state exists
                #     else:
                #         y_t[l] = reward[l] + (GAMMA)*target_q_value_1[l] ## target function update

                for l in range(len(batch)):
                    if dones[l]:
                        y_t[l] = rewards[l] # if terminal state exists
                    else:
                        y_t[l] = rewards[l] + (GAMMA)*target_q_value_1[l] ## target function update



                # print 'Target ftns'
                # print '1step'
                # print y_t
                # print 'nstep'
                # print y_t_n
                # print '========================================='

                # print 'Q estimation'
                # print q_values
                # print '========================================='




                mean_reward = rewards.mean()
                show_reward = True

                if (train_indicator): #training mode
                    # loss += critic.model.train_on_batch([sampled_img_state_n, sampled_joint_pos_state_n, actions], y_t) #Minibatch loss -> MSE, and current estimation is made

                    # batch, indices, weights_array
                    weights_array = weights_array.reshape((BATCH_SIZE,1))
                    weight_array = weight_array.reshape((SAMPLE_SIZE,1))


                    y_t_n = y_t_n.reshape((BATCH_SIZE,1))
                    y_t = y_t.reshape((SAMPLE_SIZE,1))

                 
                    # compute TD error
                    loss_array_n = weights_array * (y_t_n - q_values) ** 2 
                    loss_array_1 =  weight_array* (y_t - q_values) ** 2  
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

                    # 



                    # _mse_loss = critic.model.train_on_batch([reduced_full_state_t, reduced_sampled_obj_state, actions], {"one_step_Q": y_t, 
                    _mse_loss = critic.model.train_on_batch([reduced_full_state_t,  actions], {"one_step_Q": y_t, 
                   "N_step_Q": y_t_n}) #Minibatch loss -> MSE, and current estimation is made
                    

                    print 'Loss monitor'
                    print _mse_loss
                    # print _mse_loss
                    mse_loss = sum(_mse_loss)

                    avg_loss +=mse_loss
                    # Shouldn't it be scalar loss?
                    # from states, and acitons ( sampled batch states and actions)
                    # Keras, Sequential, train_on_batch(self, x, y, sample_weight=None, class_weight=None)
                    sampled_img_state = np.reshape(sampled_img_state,(-1,100,100,3))
                    a_for_grad = actor.model.predict([sampled_img_state, sampled_joint_eff_state])   # action gradient
                    
                    # print a_for_grad
                    # print '!!!!!!'

                    # full, aux, a_grad
                    # q_values_2, _ = critic.model.predict([reduced_full_state_t, reduced_sampled_obj_state, a_for_grad]) 
                    
                    # target_zero = np.zeros(q_values_2.shape)
                    # q_values
                    # _ = actor.model.train_on_batch(q_values_2,target_zero)

                    # grads = critic.gradients(reduced_full_state_t, reduced_sampled_obj_state, a_for_grad) # action is not sampled from mini_batch, q_value gradpeint
                    grads = critic.gradients(reduced_full_state_t, a_for_grad) # action is not sampled from mini_batch, q_value gradpeint

                    actor.train(sampled_img_state, sampled_joint_eff_state, grads)
                    actor.target_train()
                    critic.target_train()
            elapsed_time = time.time() - start_time
            total_time +=elapsed_time

                # buffer_added = False



            if show_reward:
                total_reward += mean_reward
            s_t = s_t_1
            if done:
                break

            if np.mod(j, 10) == 0:
                print("Episode", i, "Step", step, "Action", a_t, "Total_reward", total_reward, "Loss", mse_loss, "10step-Time", total_time)
                total_time = 0
           
            # Check for termination of each episode! & If true, record to Summary Writers
            avg_q , avg_loss = 0.0, 0.0
            step += 1
        stats = [total_reward, avg_q / float(step), step, avg_loss / float(step)]
        for f in range(len(stats)):
            sess.run(update_ops[f], feed_dict={summary_placeholders[f]: float(stats[f])})
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, f + 1)
        
        # buffer_added = False

        if np.mod(i, 50) == 0:
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
    playGame(1)