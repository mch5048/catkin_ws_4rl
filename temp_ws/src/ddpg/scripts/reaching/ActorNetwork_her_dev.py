#!/usr/bin/env python
import numpy as np
import math
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling, RandomUniform
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import  Dense, Flatten, Input, Concatenate,Conv2D, Activation
from keras.layers.normalization import BatchNormalization
from keras import losses

from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

STATE_LENGTH = 3 # 3 consecutive frames will be fed into both Actor and Critic Networks
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_POSE_STATE =7
CHANNELS = 3 # RGB Kinect-v1
LAMBDA_BC = 0.01

class ActorNetwork(object):
    def __init__(self, sess, state_rgb, state_robot, action_size, goal_rgb, BATCH_SIZE, TAU, LEARNING_RATE, critic_full_input, critic_obj_input, critic_goal_state, critic_model):
        self.critic_model = critic_model
        self.critic_full_input = critic_full_input
        self.critic_obj_input = critic_obj_input
        self.critic_goal_state = critic_goal_state
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        K.set_session(sess)
        #Now create the model
        self.model , self.weights, self.state_rgb, self.state_rob, self.goal_rgb = self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb)
        self.target_model, self.target_weights, self.target_state_rgb, self.target_state_rob, self.target_goal_rgb = self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient) #ys,  xs,  grad_ys=None negative gradient!!
        grads = zip(self.params_grad, self.weights)
        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())
        # actor_updates = Adam().get_updates(params=self.weights, loss=-K.mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0]))
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob],[self.model.output], updates=actor_updates)
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob, self.critic_full_input, self.critic_obj_input, self.model.output],[self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0]], updates=actor_updates)
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob, self.critic_full_input, self.model.output],[self.model.output], updates=actor_updates)
        # keras.backend.function(inputs, outputs, updates=None)
        # self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output]))
        self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.model([self.state_rgb, self.state_rob, self.goal_rgb])])[0])
        # Behaviour cloning
        # self.bc_loss = tf.reduce_sum(tf.square(self.output-))
        # self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0])
        # self.actor_grads = self.flatgrad(self.actor_loss, self.weights)
        self.optimize_actor = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.actor_loss, var_list=self.weights)


        # self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars,
        #     beta1=0.9, beta2=0.999, epsilon=1e-08)

        # def flatgrad(self, loss, weights):
        #     grads = tf.gradients(loss, weights)
            
        #     return tf.concat(axis=0, values=[
        #         tf.reshape(grad if grad is not None else tf.zeros_like(v), [np.size(v)])
        #         for (v, grad) in zip(weights, grads)
        #     ])

    # def train_actor(self, states_rgb, states_rob, critic_full_input, critic_obj_input, action_for_grad):
    def train_actor(self, states_rgb, states_rob, goal_rgb, critic_full_input, critic_obj_input, critic_goal_state):

        grad = self.sess.run(self.optimize_actor, feed_dict={
            self.state_rgb: states_rgb,
            self.state_rob: states_rob,
            self.goal_rgb: goal_rgb,
            self.critic_full_input: critic_full_input,
            self.critic_obj_input: critic_obj_input,
            self.critic_goal_state: critic_goal_state
            })

    # def loss(y_true,y_pred):
    #     return K.mean(K.square(y_pred-y_true) - K.square(y_true-noisy_img))

    # def train_actor(self, states_rgb, states_rob, critic_full_input, critic_obj_input, action_for_grad):

    #     ops = [self.actor_loss]
    #     actor_grads, actor_loss = self.sess.run(ops, feed_dict={
    #         self.state_rgb: states_rgb,
    #         self.state_rob: states_rob,
    #         self.critic_full_input: critic_full_input,
    #         self.critic_obj_input: critic_obj_input,
    #         self.model.output: action_for_grad})

    #     grad = self.sess.run(self.optimize_actor, feed_dict={
    #         self.self.actor_loss: actor_loss})



    # imported from OpenAI baselines # imported from OpenAI baselines # imported from OpenAI baselines

    # def train(self, states_rgb, states_rob, action_grads):
    #     grad = self.sess.run(self.optimize, feed_dict={
    #         self.state_rgb: states_rgb,
    #         self.state_rob: states_rob,
    #         self.action_gradient: action_grads})

    # def train(self):
    #     updates = actor_optimizer.get_updates(
    #     params=self.actor.trainable_weights, loss=-K.mean(combined_output))


    # def actor_loss(self, y_true, y_pred):
    #     # custom loss for actor
    #     loss = -tf.reduce_mean(y_true - y_pred)

    #     return loss

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # asym Actor Critic reference        
    # for i in range(0, 4):
    #     obs = tf.layers.conv2d(
    #         inputs=obs,
    #         filters=64,
    #         kernel_size=2,
    #     )

    # for i in range(0, 4):
    #     goalobs = tf.layers.conv2d(
    #         inputs=goalobs,
    #         filters=64,
    #         kernel_size=2,
    #     )  

    # x = tf.concat([obs, goalobs], axis=-1)
    # x = tf.layers.flatten(x)

    def create_actor_network(self, state_rgb, state_robot, action_dim, goal_rgb):
        ## Assymetric Actor ##

        print("Creates Actor Network")
        # Default data_format -> channels last : shape = (samples, rows, cols, channels)
        S_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_rgb_input') 
        S_robot = Input(shape=[ROBOT_POSE_STATE], name='actor_rob_pose_input') 
        G_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_goal_rgb_input') 
        AH1= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(S_rgb)
                # BFC1 = BatchNormalization()(  F)
        AH2= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu')(AH1)
        AH3= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu')(AH2)
        AH4= Conv2D(32, (4, 4),  strides=(2,2),  padding='valid', activation='relu')(AH3)

        AGH1= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(G_rgb)
        AGH2= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu')(AGH1)
        AGH3= Conv2D(32, (4, 4), strides=(2,2),  padding='valid', activation='relu')(AGH2)
        AGH4= Conv2D(32, (4, 4),  strides=(2,2),  padding='valid', activation='relu')(AGH3)

        Concatd = Concatenate(axis=-1)([AH4, AGH4])
        F = Flatten()(Concatd)
        # FC1 = Dense(300, kernel_initializer='he_uniform',activation='relu', kernel_regularizer=regularizers.l2(0.01))(F)
        FC1 = Dense(512, activation='relu')(F)

        Concatd2 = Concatenate()([FC1, S_robot])
        BFC1 = BatchNormalization()(Concatd2)
        AFC1 = Activation('relu')(BFC1)
        # FC1= Dense(256, kernel_initializer='he_uniform',activation='relu')(BF)
        # BFC1 = BatchNormalization()(Concatd)
        # FC2= Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC1)
        FC2= Dense(512,kernel_initializer='glorot_uniform')(AFC1)
        BFC2 = BatchNormalization()(FC2)
        AFC2 = Activation('relu')(BFC2)
        FC3= Dense(512, kernel_initializer='glorot_uniform')(AFC2)
        BFC3 = BatchNormalization()(FC3)
        AFC3 = Activation('relu')(BFC3)
        # FC3= Dense(300, kernel_initializer='he_uniform', activation='relu')(BFC2)
        # BFC3 = BatchNormalization()(FC3)
       
        Action= Dense(action_dim, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='tanh')(AFC3)
        # Action= Dense(action_dim, kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape),activation='tanh')(BFC4)
        adam = Adam(lr=self.LEARNING_RATE, beta_1=0.9, beta_2=0.999 , epsilon=1e-8)
       
        model = Model(input= [S_rgb,S_robot, G_rgb],output=Action)
        # model.compile(loss='mse', optimizer=adam)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, model.trainable_weights, S_rgb, S_robot, G_rgb
