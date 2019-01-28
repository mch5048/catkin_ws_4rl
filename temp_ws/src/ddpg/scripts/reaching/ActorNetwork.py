#!/usr/bin/env python
import numpy as np
import math
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling, RandomUniform
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import  Dense, Flatten, Input, Concatenate,Conv2D
from keras.layers.normalization import BatchNormalization


from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

STATE_LENGTH = 3 # 3 consecutive frames will be fed into both Actor and Critic Networks
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_POSE_STATE = 7
CHANNELS = 3 # RGB Kinect-v1

class ActorNetwork(object):
    def __init__(self, sess, state_rgb, state_robot, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        K.set_session(sess)
        #Now create the model
        self.model , self.weights, self.state_rgb, self.state_rob = self.create_actor_network(state_rgb, state_robot, action_size)
        self.target_model, self.target_weights, self.target_state_rgb, self.target_state_rob = self.create_actor_network(state_rgb, state_robot, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states_rgb, states_rob, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_rgb: states_rgb,
            self.state_rob: states_rob,
            self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_rgb,state_robot,action_dim):
        ## Assymetric Actor ##

        print("Creates Actor Network")
        # Default data_format -> channels last : shape = (samples, rows, cols, channels)
        S_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS)) 
        S_robot = Input(shape=[ROBOT_POSE_STATE]) 
        AH1= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(S_rgb)
        AH2= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(AH1)
        AH3= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(AH2)
        F = Flatten()(AH3)
        Concatd = Concatenate()([F, S_robot])
        # BF = BatchNormalization()(F)
        # FC1= Dense(256, kernel_initializer='he_uniform',activation='relu')(BF)
        FC1= Dense(256, kernel_initializer='he_uniform',activation='relu')(Concatd)
        BFC1 = BatchNormalization()(FC1)
        # FC2= Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC1)
        FC2= Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC1)
        BFC2 = BatchNormalization()(FC2)
        FC3= Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC2)
        BFC3 = BatchNormalization()(FC3)
        FC4= Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC3)
        BFC4 = BatchNormalization()(FC4)
        # FC3= Dense(action_dim, kernel_initializer=lambda shape:VarianceScaling(scale=3e-2)(shape),activation='tanh')(BFC2)
        
        Action= Dense(action_dim, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='tanh')(BFC4)

        
        # S = Input(shape=[state_size])
        # h0 = Dense(400, kernel_initializer='he_uniform',activation='relu')(S)
        # h1 = Dense(400, kernel_initializer='he_uniform',activation='relu')(h0)
        # V = Dense(action_dim, kernel_initializer=lambda shape:VarianceScaling(scale=3e-3)(shape),activation='tanh')(h1)
        #V = Dense(action_dim, init=lambda shape, name: uniform(shape, scale=3e-3, name=name),activation='tanh')(h1)
        model = Model(input= [S_rgb,S_robot],output=Action)
        return model, model.trainable_weights, S_rgb, S_robot
