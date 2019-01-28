#!/usr/bin/env python
import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.models import model_from_json
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, concatenate, Lambda, Activation, Conv2D, Add, Concatenate, LeakyReLU
from keras.initializers import VarianceScaling, RandomUniform
from keras.layers.normalization import BatchNormalization
from keras import regularizers, losses

from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import tensorflow.contrib as tc
import tf_util as U
# from mpi_adam import MpiAdam
from mpi4py import MPI
import random

from mpi_tf import MpiAdamOptimizer
from keras_layer_normalization import LayerNormalization



# for Target policy smoothing
# from Gaussian import Gaussian 

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
STATE_LENGTH = 4
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_FULL_STATE = 21
CRITIC_GOAL_STATE = 14
OBJ_STATE = 3
CHANNELS =3
L2_COEF = 0.01
BATCH_SIZE = 64
# LOSS_COEF_DICT = {'onestep':1.0, 'Nstep':0.0, 'L2coef':0.01}
LOSS_COEF_DICT = {'onestep':0.3, 'Nstep':0.7, 'L2coef':0.01}


# An MPI Adam version of critic learning...

class MpiAdam(object):
    def __init__(self, var_list, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)





class CriticNetwork(object):
    def __init__(self, sess, state_robot_full, state_obj, action_size, goal_critic_dim, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # self.action_noise = Gaussian(mu=noise_mean, sigma=float(gauss_stddev) * np.ones(action_dim))


        self.action_size = action_size
        # self.model,self.action, self.state_full, self.state_obj = self.create_critic_network(state_robot_full, action_size)
        with tf.variable_scope('Critic'):
            self.model, self.action, self.state_full,self.state_obj, self.goal_state_critic = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='eval_net', trainable=True)
            self.model_2, self.action_2, self.state_full_2,self.state_obj_2, self.goal_state_critic_2 = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='eval_net_2', trainable=True)

        # self.target_model, self.target_action, self.target_state_full, self.target_state_obj  = self.create_critic_network(state_robot_full,  action_size)
            self.target_model, self.target_action, self.target_state_full, self.target_state_obj, self.target_goal_state_critic = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='target_net', trainable=False)
            self.target_model_2, self.target_action_2, self.target_state_full_2,self.target_state_obj_2, self.target_goal_state_critic_2 = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='target_net_2', trainable=False)

        # self.optimizer = self.optimizer()
        K.set_session(sess)
        #Now create the model
        self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/target_net')

        self.weights = self.model.trainable_weights
        self.weights_2 = self.model_2.trainable_weights
        self.total_critic_weights = self.weights + self.weights_2
        random.seed(219)
        np.random.seed(219)



        # pi_loss = -tf.reduce_mean(q1_pi)
        # q1_loss = tf.reduce_mean((q1-backup)**2)
        # q2_loss = tf.reduce_mean((q2-backup)**2)
        # q_loss = q1_loss + q2_loss

        # define custom critic loss
        with tf.variable_scope('critic_loss'):  
            self.priority_weight = tf.placeholder(tf.float32,[None,1],name='priority_weight_for_loss')            
            self.onestep_target = tf.placeholder(tf.float32,[None,1], name='onestep_target') # it's derived from target network , so does not affect learning            
            self.Nstep_target = tf.placeholder(tf.float32,[None,1], name='Nstep_target') # it's derived from target network , so does not affect learning
             
            with tf.variable_scope('q1_loss'):
                 # self.q_pred = self.model([self.state_full, self.state_obj, self.goal_state_critic, self.action])   
                self.q_pred_1= self.model([self.state_full, self.state_obj, self.goal_state_critic, self.action])   
                self.onestep_td_err_1 =  tf.square(self.onestep_target-self.q_pred_1[0])                   # Should check for the shapes
                self.Nstep_td_err_1 =  tf.square(self.Nstep_target-self.q_pred_1[0])
                # self.avg_q_pred_1 = 

                self.onestep_loss_1 = tf.reduce_mean(self.onestep_td_err_1*self.priority_weight, name='onestep_loss')
                self.Nstep_loss_1 = tf.reduce_mean(self.Nstep_td_err_1*self.priority_weight, name='Nstep_loss')
                # self.l2_regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in self.weights if 'kernel' in v.name and 'output' not in var.name], name='l2_reg_loss')
                with tf.variable_scope('critic_l2_reg'):  
                    self.critic_reg_vars_1 = [var for var in self.weights if 'kernel' in var.name and 'q_value' not in var.name]            
                    self.critic_reg_loss_1 = tc.layers.apply_regularization(tc.layers.l2_regularizer(LOSS_COEF_DICT['L2coef']),weights_list=self.critic_reg_vars_1)                       
            with tf.variable_scope('q2_loss'):
                # print 'Q_pred shape is'        
                # print self.q_pred[0].shap
                self.q_pred_2= self.model_2([self.state_full_2, self.state_obj_2, self.goal_state_critic_2, self.action_2])       
                self.onestep_td_err_2 =  tf.square(self.onestep_target-self.q_pred_2[0])                   # Should check for the shapes
                self.Nstep_td_err_2 =  tf.square(self.Nstep_target-self.q_pred_2[0])
                
                self.onestep_loss_2 = tf.reduce_mean(self.onestep_td_err_2*self.priority_weight, name='onestep_loss')
                self.Nstep_loss_2 = tf.reduce_mean(self.Nstep_td_err_2*self.priority_weight, name='Nstep_loss')
                # self.l2_regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in self.weights if 'kernel' in v.name and 'output' not in var.name], name='l2_reg_loss')
                with tf.variable_scope('critic_l2_reg'):  
                    self.critic_reg_vars_2 = [var for var in self.weights_2 if 'kernel' in var.name and 'q_value' not in var.name]            
                    self.critic_reg_loss_2 = tc.layers.apply_regularization(tc.layers.l2_regularizer(LOSS_COEF_DICT['L2coef']),weights_list=self.critic_reg_vars_2)                       

            # q_loss = q1_loss + q2_loss
            self.critic_total_loss = LOSS_COEF_DICT['onestep']*self.onestep_loss_1 + LOSS_COEF_DICT['Nstep']*self.Nstep_loss_1 + self.critic_reg_loss_1 + LOSS_COEF_DICT['onestep']*self.onestep_loss_2 + LOSS_COEF_DICT['Nstep']*self.Nstep_loss_2 + self.critic_reg_loss_2
        # now define optimizations
        with tf.variable_scope('train_critic'):
            with tf.variable_scope('critic_MPIadam'):
                # self.critic_optimizer = MpiAdam(var_list=self.weights,  beta1=0.9, beta2=0.999, epsilon=1e-08)
                self.critic_optimizer = MpiAdamOptimizer(LEARNING_RATE)
            with tf.variable_scope('critic_flatgrad'):
                # self.critic_flatgrad, self.grads = U.flatgrad(self.critic_total_loss, self.weights, clip_norm=200.0)
                self.critic_flatgrad_vars, self.grads_and_vars = self.critic_optimizer.compute_gradients(self.critic_total_loss, var_list = self.total_critic_weights, clip_norm=200.0)
            with tf.variable_scope('critic_apply_gradients'):
                self.train_critic_op = self.critic_optimizer.apply_gradients(self.critic_flatgrad_vars)


            # self.optimize_critic = tf.train.AdamOptimizer(LEARNING_RATE)
            # self.grads_and_vars_q = self.optimize_critic.compute_gradients(self.critic_total_loss, var_list=self.weights)
        with tf.variable_scope('critic_l2_norm_summary'):
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
       

        s = []
        # for grad, var in zip(self.grads, self.weights):
        for grad, var in self.grads_and_vars:
            print grad
            print var            
            s.append(tf.summary.histogram(var.op.name + '', var, family='critic_summary'))
            if grad is not None:
                s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='critic_summary'))
                s.append(tf.summary.histogram(var.op.name + '/gradients/norm', l2_norm(grad), family='critic_summary'))
        # s.append(tf.summary.histogram(self.critic_flatgrad.op.name + '', self.critic_flatgrad, family='critic_summary'))
        self.critic_summary_op = tf.summary.merge(s)


        # with tf.variable_scope("loss", reuse=False):
        #     self.critic_tf = denormalize(
        #         tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]),
        #         self.ret_rms)

        #     self.critic_with_actor_tf = denormalize(
        #         tf.clip_by_value(self.normalized_critic_with_actor_tf,
        #                          self.return_range[0], self.return_range[1]),
        #         self.ret_rms)

        #     q_obs1 = denormalize(critic_target, self.ret_rms)
        #     self.target_q = self.rewards + (1. - self.terminals1) * self.gamma * q_obs1

        #     tf.summary.scalar('critic_target', tf.reduce_mean(self.critic_target))
        #     tf.summary.histogram('critic_target', self.critic_target)

        #     # Set up parts.
        #     if self.normalize_returns and self.enable_popart:
        #         self._setup_popart()
        #     self._setup_stats()
        #     self._setup_target_network_updates()

        # self.train_critic_op = self.optimize_critic.apply_gradients(self.grads_and_vars_q)



        '''
            Critic Loss reference
                self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
                if self.critic_l2_reg > 0.:
                    critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
                    for var in critic_reg_vars:
                        logger.info('  regularizing: {}'.format(var.name))
                    logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
                    critic_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.critic_l2_reg),
                        weights_list=critic_reg_vars
                    )
                    self.critic_loss += critic_reg

        '''


    # def train_critic_with_gradient(self, priority_weight, onestep_target, Nstep_target, state_full, state_obj, goal_state_critic, action):
    #     with tf.variable_scope('update_critic_param'):
    #         train_op = [self.train_critic_op, self.critic_summary_op, self.oneste-p_loss, self.Nstep_loss, self.onestep_td_err, self.Nstep_td_err, self.q_pred, self.l2_regularizer]
    #                         #1                             #2               #3              #4                      #5                  #6              #7             #8
    #         self.train_out = self.sess.run(train_op, feed_dict={
    #             self.priority_weight:priority_weight, 
    #             self.onestep_target: onestep_target,
    #             self.Nstep_target: Nstep_target,
    #             self.state_full: state_full,  
    #             self.state_obj: state_obj,  # feed_dict for critic
    #             self.goal_state_critic: goal_state_critic,  # feed_dict for critic
    #             self.action: action  # feed_dict for critic
    #         })
    #     return  LOSS_COEF_DICT['onestep']*self.train_out[2], LOSS_COEF_DICT['Nstep']*self.train_out[3], self.train_out[4], self.train_out[5] , self.train_out[6], LOSS_COEF_DICT['L2coef']*self.train_out[7] # for priority update     


    # just needed to add 
    def train_critic_with_flatgrad(self, priority_weight, onestep_target, Nstep_target, state_full, state_obj, goal_state_critic, action):
        with tf.variable_scope('update_critic_param'):
            # train_op = [self.critic_flatgrad, self.critic_summary_op, self.onestep_loss, self.Nstep_loss, self.onestep_td_err, self.Nstep_td_err, self.q_pred, self.critic_reg_loss]
            train_op = [self.train_critic_op, self.critic_summary_op, self.onestep_loss_1, self.Nstep_loss_1, self.onestep_td_err_1, self.Nstep_td_err_1, self.q_pred_1, self.critic_reg_loss_1, self.onestep_loss_2, self.Nstep_loss_2, self.onestep_td_err_2, self.Nstep_td_err_2, self.q_pred_2, self.critic_reg_loss_2]
                            # 0                             1               2                       3                   4                   5                   6               7                   8                   9               
            self.train_out = self.sess.run(train_op, feed_dict={
                self.priority_weight:priority_weight, 
                self.onestep_target: onestep_target,
                self.Nstep_target: Nstep_target,   # shared feed variables

                self.state_full: state_full,  
                self.state_obj: state_obj,  # feed_dict for critic
                self.goal_state_critic: goal_state_critic,  # feed_dict for critic
                self.action: action,  # feed_dict for critic

                self.state_full_2: state_full,  
                self.state_obj_2: state_obj,  # feed_dict for critic
                self.goal_state_critic_2: goal_state_critic,  # feed_dict for critic
                self.action_2: action,  # feed_dict for critic
            })

                                # 1step loss                            nstep Loss                             onestep tderr      nstep tderr             q_pred              l2 reg loss                                                  
        q1_info = (LOSS_COEF_DICT['onestep']*self.train_out[2], LOSS_COEF_DICT['Nstep']*self.train_out[3], self.train_out[4], self.train_out[5] , self.train_out[6], self.train_out[7])
        q2_info = (LOSS_COEF_DICT['onestep']*self.train_out[8], LOSS_COEF_DICT['Nstep']*self.train_out[9], self.train_out[10], self.train_out[11] , self.train_out[12], self.train_out[13])

        # return  LOSS_COEF_DICT['onestep']*self.train_out[2], LOSS_COEF_DICT['Nstep']*self.train_out[3], self.train_out[4], self.train_out[5] , self.train_out[6] # for priority update     
        return  q1_info, q2_info     


    def get_vars(scope):
        return [x for x in tf.global_variables() if scope in x.name]


    def target_train(self, critic_actor_weights_arr):
        # for Cic 1
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
        # for Critic 2
        critic_weights_2 = self.model_2.get_weights()
        critic_target_weights_2 = self.target_model_2.get_weights()
        for i in xrange(len(critic_weights_2)):
            critic_target_weights_2[i] = self.TAU * critic_weights_2[i] + (1 - self.TAU)* critic_target_weights_2[i]
        self.target_model_2.set_weights(critic_target_weights_2)

        # critic_actor_vars = self.get_vars('Critic_1')
        # return [x for x in tf.global_variables() if scope in x.name]
        # train Critic_1


    def init_network(self):
        print 'Executes update for both eval and target net'
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] =  critic_weights[i]
        self.target_model.set_weights(critic_target_weights)

        critic_weights_2 = self.model_2.get_weights()
        critic_target_weights_2 = self.target_model_2.get_weights()
        for i in xrange(len(critic_weights_2)):
            critic_target_weights_2[i] = critic_weights_2[i]
        self.target_model_2.set_weights(critic_target_weights_2)

    def update_critic_summary(self, summary_writer = None, global_step=None):
        # critic_weights_str = self.sess.run(self.critic_summary_op)
        # self.summar/y_writer = summary_writer
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, i + 1)
        if self.train_out[1]:
            summary_str = self.train_out[1]
            summary_writer.add_summary(summary_str, global_step)

    # def target_train(self):
    #     print 'Executes target network update'
    #     self.eval_params
    #     self.target_params



        # soft_updates = []
        # assert len(self.eval_params) == len(self.target_params)
        # for var, target_var in zip(self.eval_params, self.target_params):
        #     print target_var
        #     soft_updates.append(tf.assign(target_var, (1. - self.TAU) * target_var + self.TAU * var))
        # assert len(soft_updates) == len(self.eval_params)
        # soft_update_op = tf.group(*soft_updates)
        # self.sess.run(soft_update_op)
    def Layer_Normalization(self, x):
        return tc.layers.layer_norm(x, center=True, scale=True)

    def create_critic_network(self, state_robot_full, state_obj, action_dim, goal_critic_dim, scope, trainable):
        ## Assymetric Critic ##
        with tf.variable_scope(scope):
            print("Now we build the model")

            # backup 181106
            # with tf.name_scope('states'):
            #     S_full = Input(shape=[ROBOT_FULL_STATE], name = 'critic_full_obs')
            #     S_obj = Input(shape=[OBJ_STATE], name = 'critic_obj_input') 
            #     G_critic = Input(shape=[CRITIC_GOAL_STATE], name = 'critic_goal_input')
            #     Concatd = Concatenate(axis=-1)([S_full, S_obj,G_critic])
            # with tf.name_scope('actions'):
            #     A = Input(shape=[action_dim],name='action_for_crit')
            # FC_A = Dense(200, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fca',  trainable=trainable)(A)
            # LNA = Lambda(self.Layer_Normalization, name='layer_norm_a')(FC_A) # Q is scalar value!! dimenstion should be 1
            # A_A = LeakyReLU()(LNA)            

            with tf.name_scope('states'):
                S_full = Input(shape=[ROBOT_FULL_STATE], name = 'critic_full_obs')
                S_obj = Input(shape=[OBJ_STATE], name = 'critic_obj_input') 
                G_critic = Input(shape=[CRITIC_GOAL_STATE], name = 'critic_goal_input')
            Concatd = Concatenate(axis=-1)([S_full, S_obj,G_critic])

            with tf.name_scope('actions'):
                A = Input(shape=[action_dim],name='action_for_crit')

            # FC_A = Dense(200, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fca',  trainable=trainable)(A)
            # LNA = Lambda(self.Layer_Normalization, name='layer_norm_a')(FC_A) # Q is scalar value!! dimenstion should be 1
            # A_A = LeakyReLU()(LNA)  
            # A_A = Activation('relu')(LNA)
            # FC_S = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fcs',  trainable=trainable)(Concatd)
            # LNS = Lambda(self.Layer_Normalization, name='layer_norm_sc')(FC_S) # Q is scalar value!! dimenstion should be 1
            # A_S = Activation('relu')(LNS)
            # Concatd = Concatenate(axis=-1)([A_A , A_S])

            # reference
            # norm_layer = LayerNormalization()(input_layer)
            # references


            FC1 = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc1',  trainable=trainable)(Concatd)
            # LN1 = Lambda(self.Layer_Normalization, name='layer_norm1')(FC1) # Q is scalar value!! dimenstion should be 1
            LN1 = LayerNormalization()(FC1)# replace with normal layer norm module!
            A1 = Activation('relu')(LN1)

            Concatd_2 = Concatenate(axis=-1)([A1, A])  

            FC2 = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc2',  trainable=trainable)(Concatd_2)
            LN2 = LayerNormalization()(FC2) # Q is scalar value!! dimenstion should be 1
            A2 = Activation('relu')(LN2)     

            FC3 = Dense(200, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc3',  trainable=trainable)(A2)
            LN3 = LayerNormalization()(FC3) # Q is scalar value!! dimenstion should be 1
            A3 = Activation('relu')(LN3)   

            # FC3 = Dense(256, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc3',  trainable=trainable)(A2)
            # LN3 = Lambda(self.Layer_Normalization, name='layer_norm3sub')(FC3) # Q is scalar value!! dimenstion should be 1
            # A3 = Activation('relu')(LN3)      

            # AFC2 = Leakyselu()(FC2)            
            # FC3 = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc3', trainable=trainable)(A2)
            # LN3 = Lambda(self.Layer_Normalization, name='layer_norm3')(FC3) # Q is scalar value!! dimenstion should be 1
            # A3 = Activation('relu')(LN3)

            # FC4 = Dense(512, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc4', trainable=trainable)(A3)
            # LN4 = Lambda(self.Layer_Normalization, name='layer_norm4')(FC4) # Q is scalar value!! dimenstion should be 1
            # A4 = Activation('selu')(LN4)

            # AFC4 = Leakyselu()(FC4)
            # FC3 = Dense(300, activation='selu', kernel_regularizer=None)(BFC2)
            # BFC3 = BatchNormalization()(FC3)
            # FC4 = Dense(300, activation='selu', kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), kernel_regularizer=None)(BFC3)
            # BFC4 = BatchNormalization()(FC4)        
            # Q = Dense(1, kernel_initializer=lambda shape:VarianceScaling(scale=3e-3)(shape),activation='linear')(BFC4) # Q is scalar value!! dimenstion should be 1
            Q = Dense(1, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='linear', name='q_value',trainable=trainable)(A3) # Q is scalar value!! dimenstion should be 1
            Q_1 = Lambda(lambda x:x, name = "one_step_Q")(Q) # Q is scalar value!! dimenstion should be 1
            Q_2 = Lambda(lambda x:x, name = "N_step_Q")(Q) # Q is scalar value!! dimenstion should be 1
            # model = Model(input=[S_full, A],output=[Q_1, Q_2]) # Q_1 to n_step, Q_n to 1_stpe
            model = Model(input=[S_full, S_obj, G_critic, A],output=[Q_1, Q_2]) # Q_1 to n_step, Q_n to 1_stpe
            # model = Model(input=[S_full, S_obj, A],output=Q)
            # adam = Adam(lr=self.LEARNING_RATE)

            # losses = { "one_step_Q": "mse", 
            #            "N_step_Q": "mse"} 

            # lossWeights = {"one_step_Q": 0.7, "N_step_Q": 0.3}
            # model.compile(loss='mse', optimizer=adam)  # Critic's Loss ftn is defined from MSE!!
            # model.compile( loss=losses, loss_weights=lossWeights, optimizer=adam)
            # model.compile( loss=losses, loss_weights=lossWeights, optimizer=adam)
            model.summary()

            # model   
            # adam = Adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999 , epsilon=1e-8  )

            # return model, A, S_full, S_obj
            return model, A, S_full, S_obj, G_critic


