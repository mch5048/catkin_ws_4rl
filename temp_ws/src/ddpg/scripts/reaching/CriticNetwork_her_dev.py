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
L2_COEF = 5e-3

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

        # define Critic loss here

        # with tf.variable_scope('critic_loss'):            
        #     self.critic_loss = tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.model([self.state_rgb, self.state_rob, self.goal_rgb])])[0], name='actor_main_loss')
        #     # Behaviour cloning
        #     # tf.where(condition,true state,false state)
        #     with tf.variable_scope('1_step_loss'):
        #         self.one_stp_loss = tf.reduce_mean()
        #         self.sampled_action = tf.placeholder(tf.float32,[None,7], name='sampled_action_from_batch')        
        #         self.isDemo = tf.placeholder(tf.float32,[None,7], name='isDemo')
        #         self.bc_loss = tf.reduce_sum(tf.square(self.model([self.state_rgb, self.state_rob, self.goal_rgb])-self.sampled_action), name='bc_loss_sum')*self.isDemo #*tf.cast(self.isDemo,tf.int32)*tf.cast(tf.greater(self.critic_sample_pred, self.critic_target_pred), tf.int32) # x>y # network prediction, sample action
            
        #     self.actor_total_loss = tf.add(tf.negative(self.actor_loss), self.bc_loss, name='actor_total_loss')



        # # How to implement L2 Critic loss??
        # loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=out_layer, labels=tf_train_labels)) +
        #     0.01*tf.nn.l2_loss(hidden_weights) +
        #     0.01*tf.nn.l2_loss(hidden_biases) +
        #     0.01*tf.nn.l2_loss(out_weights) +
        #     0.01*tf.nn.l2_loss(out_biases))


        
        # self.critic_optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        # self.grads_and_vars_critic = self.critic_optimizer.compute_gradients(self.actor_total_loss, var_list=self.weights)
        # self.optimize_critic = self.critic_optimizer.apply_gradients(self.grads_and_vars_actor, name='optimize_actor')

        # setup critic training sumamry
        self.critic_weights = self.model.trainable_weights
        # print self.model.get_weights()
        # print self.critic_weights
        s = []
        for var in self.critic_weights:
            s.append(tf.summary.histogram(var.op.name + '', var, family='critic_summary'))
            # s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='actor_summary'))
        # s.append(tf.summary.image('actor_conv_result_achvd',self.conv_achv_out, family='actor_summary'))    
        # s.append(tf.summary.image('actor_conv_result_achvdresult_goal',self.conv_goal_out, family='actor_summary'))    
        self.critic_summary_op = tf.summary.merge(s)
        # self.value_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update



    # check if it's necessary
    # def gradients(self, states_full, states_aux, actions):

    #     return self.sess.run(self.value_grads, feed_dict={
    #         self.state_full: states_full,
    #         self.state_aux: states_aux,
    #         self.action: actions
    #     })[0]

    # def gradients(self, states_full, actions):
    
    # def gradients(self, states_full, state_obj, actions, goal_state_critic):

    #     return self.sess.run(self.value_grads, feed_dict={
    #         self.state_full: states_full,
    #         self.goal_state_critic: goal_state_critic,
    #         self.state_obj: state_obj,
    #         self.action: actions
    #     })[0]

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
        critic_weights_str = self.sess.run(self.critic_summary_op)
        # self.summar/y_writer = summary_writer
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, i + 1)
        if critic_weights_str:
            summary_str = critic_weights_str
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
            FC1 = Dense(512, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(L2_COEF),name='critic_fc1', trainable=trainable)(Concatd)
            AFC1 = LeakyReLU()(FC1)
            BFC1 = BatchNormalization()(AFC1)
            FC2 = Dense(512, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(L2_COEF),name='critic_fc2',trainable=trainable)(BFC1)
            AFC2 = LeakyReLU()(FC2)            
            BFC2 = BatchNormalization()(AFC2)
            FC3 = Dense(512, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(L2_COEF),name='critic_fc3',trainable=trainable)(BFC2)
            AFC3 = LeakyReLU()(FC3)
            BFC3 = BatchNormalization()(AFC3)
            # FC3 = Dense(300, activation='relu', kernel_regularizer=None)(BFC2)
            # BFC3 = BatchNormalization()(FC3)
            # FC4 = Dense(300, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=None)(BFC3)
            # BFC4 = BatchNormalization()(FC4)        
            # Q = Dense(1, kernel_initializer=lambda shape:VarianceScaling(scale=3e-3)(shape),activation='linear')(BFC4) # Q is scalar value!! dimenstion should be 1
            Q = Dense(1, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='linear', name='q_value',trainable=trainable, kernel_regularizer=regularizers.l2(L2_COEF))(BFC3) # Q is scalar value!! dimenstion should be 1
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
            # adam = Adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999 , epsilon=1e-8  )

            # return model, A, S_full, S_obj
            return model, A, S_full, S_obj, G_critic

