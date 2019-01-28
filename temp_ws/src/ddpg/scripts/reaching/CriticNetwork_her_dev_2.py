#!/usr/bin/env python
import numpy as np
import math
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
LOSS_COEF_DICT = {'onestep':0.9, 'Nstep':0.1, 'L2coef':10.0}



class CriticNetwork(object):
    def __init__(self, sess, state_robot_full, state_obj, action_size, goal_critic_dim, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.action_size = action_size
        # self.model,self.action, self.state_full, self.state_obj = self.create_critic_network(state_robot_full, action_size)
        with tf.variable_scope('Critic'):
            self.model, self.action, self.state_full,self.state_obj, self.goal_state_critic = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='eval_net', trainable=True)
        # self.target_model, self.target_action, self.target_state_full, self.target_state_obj  = self.create_critic_network(state_robot_full,  action_size)
            self.target_model, self.target_action, self.target_state_full, self.target_state_obj, self.target_goal_state_critic = self.create_critic_network(state_robot_full, state_obj, action_size, goal_critic_dim, scope='target_net', trainable=False)
        # self.optimizer = self.optimizer()
        K.set_session(sess)
        #Now create the model
        self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/target_net')

        self.weights = self.model.trainable_weights

        # define custom critic loss
        with tf.variable_scope('critic_loss'):  
            self.priority_weight = tf.placeholder(tf.float32,[None,1],name='priority_weight_for_loss')            
            self.onestep_target = tf.placeholder(tf.float32,[None,1], name='onestep_target') # it's derived from target network , so does not affect learning            
            self.Nstep_target = tf.placeholder(tf.float32,[None,1], name='Nstep_target') # it's derived from target network , so does not affect learning
            self.q_pred = self.model([self.state_full, self.state_obj, self.goal_state_critic, self.action])   
            # print 'Q_pred shape is'        
            # print self.q_pred[0].shape        
            self.onestep_td_err =  tf.square(self.onestep_target-self.q_pred[0])                   # Should check for the shapes
            # print 'onestep td err shape is'        
            # print self.onestep_td_err.shape 
            self.Nstep_td_err =  tf.square(self.Nstep_target-self.q_pred[0])
            self.onestep_loss = tf.reduce_mean(self.onestep_td_err*self.priority_weight, name='onestep_loss')
            self.Nstep_loss = tf.reduce_mean(self.Nstep_td_err*self.priority_weight, name='Nstep_loss')
            self.l2_regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in self.weights if 'kernel' in v.name ], name='l2_reg_loss')
            self.critic_total_loss = LOSS_COEF_DICT['onestep']*self.onestep_loss + LOSS_COEF_DICT['Nstep']*self.Nstep_loss +  LOSS_COEF_DICT['L2coef']*self.l2_regularizer
        # now define optimizations
        with tf.variable_scope('optimize_critic'):
            self.optimize_critic = tf.train.AdamOptimizer(LEARNING_RATE)
            self.grads_and_vars_q = self.optimize_critic.compute_gradients(self.critic_total_loss, var_list=self.weights)

        s = []
        for grad, var in self.grads_and_vars_q:
            s.append(tf.summary.histogram(var.op.name + '', var, family='critic_summary'))
            s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='critic_summary'))
        self.critic_summary_op = tf.summary.merge(s)
        self.train_critic_op = self.optimize_critic.apply_gradients(self.grads_and_vars_q)

    def train_critic_with_gradient(self, priority_weight, onestep_target, Nstep_target, state_full, state_obj, goal_state_critic, action):
        with tf.variable_scope('update_critic_param'):
            train_op = [self.train_critic_op, self.critic_summary_op, self.onestep_loss, self.Nstep_loss, self.onestep_td_err, self.Nstep_td_err, self.q_pred, self.l2_regularizer]
                            #1                             #2               #3              #4                      #5                  #6              #7             #8
            self.train_out = self.sess.run(train_op, feed_dict={
                self.priority_weight:priority_weight, 
                self.onestep_target: onestep_target,
                self.Nstep_target: Nstep_target,
                self.state_full: state_full,  
                self.state_obj: state_obj,  # feed_dict for critic
                self.goal_state_critic: goal_state_critic,  # feed_dict for critic
                self.action: action  # feed_dict for critic
            })
        return  LOSS_COEF_DICT['onestep']*self.train_out[2], LOSS_COEF_DICT['Nstep']*self.train_out[3], self.train_out[4], self.train_out[5] , self.train_out[6], LOSS_COEF_DICT['L2coef']*self.train_out[7] # for priority update     


    # def gradients(self, states_full, actions):
    # def get_value_grad_to_action(self, states_full, state_obj, goal_state_critic, actor_rgb, actor_rob, actor_goal_rgb):
    #     with tf.variable_scope('Q_val_grad_to_action'):
    #         return self.sess.run(self.value_grads, feed_dict={
    #             self.state_full: states_full,
    #             self.goal_state_critic: goal_state_critic,
    #             self.state_obj: state_obj,
    #             self.actor_rgb: actor_rgb,  # feed_dict for critic
    #             self.actor_rob: actor_rob,  # feed_dict for critic
    #             self.actor_goal_rgb: actor_goal_rgb,  # feed_dict for critic
    #         })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def init_network(self):
        print 'Executes update for both eval and target net'
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] =  critic_weights[i]
        self.target_model.set_weights(critic_target_weights)

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
            with tf.name_scope('states'):
                S_full = Input(shape=[ROBOT_FULL_STATE], name = 'critic_full_obs')
                S_obj = Input(shape=[OBJ_STATE], name = 'critic_obj_input') 
                G_critic = Input(shape=[CRITIC_GOAL_STATE], name = 'critic_goal_input') 
                A = Input(shape=[action_dim],name='action_for_crit')
            # FC_a = Dense(90, activation='relu',  kernel_regularizer=regularizers.l2(0.01))(A)
            # FC1 = Dense(300, activation='relu',  kernel_regularizer=regularizers.l2(0.01))(Concatd)
            # CH1= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(S)
            # CH2= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(CH1)
            # CH3= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(CH2)
            # F = Flatten()(CH3)
            # BF = BatchNormalization()(F)
            # BA = BatchNormalization()(A)
            Concatd = Concatenate(axis=-1)([S_full, S_obj, A, G_critic])
            # Concatd = Concatenate()([FC_s, A])
            FC1 = Dense(512, kernel_initializer='he_uniform', name='critic_fc1',  trainable=trainable)(Concatd)
            LN1 = Lambda(self.Layer_Normalization, name='layer_norm1')(FC1) # Q is scalar value!! dimenstion should be 1
            A1 = Activation('relu')(LN1)

            FC2 = Dense(512, kernel_initializer='he_uniform', name='critic_fc2',  trainable=trainable)(A1)
            LN2 = Lambda(self.Layer_Normalization, name='layer_norm2')(FC2) # Q is scalar value!! dimenstion should be 1
            A2 = Activation('relu')(LN2)           

            # AFC2 = LeakyReLU()(FC2)            
            FC3 = Dense(512, kernel_initializer='he_uniform', name='critic_fc3', trainable=trainable)(A2)
            LN3 = Lambda(self.Layer_Normalization, name='layer_norm3')(FC3) # Q is scalar value!! dimenstion should be 1
            A3 = Activation('relu')(LN3)

            # FC4 = Dense(512, kernel_initializer='he_uniform', name='critic_fc4', trainable=trainable)(A3)
            # LN4 = Lambda(self.Layer_Normalization, name='layer_norm4')(FC4) # Q is scalar value!! dimenstion should be 1
            # A4 = Activation('relu')(LN4)

            # AFC4 = LeakyReLU()(FC4)
            # FC3 = Dense(300, activation='relu', kernel_regularizer=None)(BFC2)
            # BFC3 = BatchNormalization()(FC3)
            # FC4 = Dense(300, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=None)(BFC3)
            # BFC4 = BatchNormalization()(FC4)        
            # Q = Dense(1, kernel_initializer=lambda shape:VarianceScaling(scale=3e-3)(shape),activation='linear')(BFC4) # Q is scalar value!! dimenstion should be 1
            Q = Dense(1, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='linear', name='q_value',trainable=trainable)(A3) # Q is scalar value!! dimenstion should be 1
            Q_1 = Lambda(lambda x:x, name = "one_step_Q")(Q) # Q is scalar value!! dimenstion should be 1
            Q_2 = Lambda(lambda x:x, name = "N_step_Q")(Q) # Q is scalar value!! dimenstion should be 1
            # model = Model(input=[S_full, A],output=[Q_1, Q_2]) # Q_1 to n_step, Q_n to 1_stpe
            model = Model(input=[S_full, S_obj, G_critic, A],output=[Q_1, Q_2]) # Q_1 to n_step, Q_n to 1_stpe
            # model = Model(input=[S_full, S_obj, A],output=Q)
            adam = Adam(lr=self.LEARNING_RATE)

            losses = { "one_step_Q": "mse", 
                       "N_step_Q": "mse"} 

            lossWeights = {"one_step_Q": 0.7, "N_step_Q": 0.3}
            # model.compile(loss='mse', optimizer=adam)  # Critic's Loss ftn is defined from MSE!!
            model.compile( loss=losses, loss_weights=lossWeights, optimizer=adam)
            # model.compile( loss=losses, loss_weights=lossWeights, optimizer=adam)
            model.summary()

            model   
            # adam = Adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999 , epsilon=1e-8  )

            # return model, A, S_full, S_obj
            return model, A, S_full, S_obj, G_critic

