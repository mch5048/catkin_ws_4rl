#!/usr/bin/env python
import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, concatenate, Lambda, Activation, Conv2D, Add, Concatenate
from keras.initializers import VarianceScaling, RandomUniform
from keras.layers.normalization import BatchNormalization
# fromkera import regularizers

from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
STATE_LENGTH = 4
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_FULL_STATE = 14
OBJ_STATE = 3
CHANNELS =3 

class CriticNetwork(object):
    def __init__(self, sess, state_robot_full, state_obj, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state_full,self.state_aux  = self.create_critic_network(state_robot_full, state_obj, action_size)
        self.target_model, self.target_action, self.target_state_full, self.target_state_aux  = self.create_critic_network(state_robot_full, state_obj, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states_full, states_aux, actions):
        # grads = critic.gradients(states, a_for_grad) 

        # print self.model.output
        return self.sess.run(self.action_grads, feed_dict={
            self.state_full: states_full,
            self.state_aux: states_aux,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_robot_full, state_obj, action_dim):
        ## Assymetric Critic ##


        # add l1 && l2 regularization

        print("Now we build the model")
        
        S_full = Input(shape=[ROBOT_FULL_STATE], name = 'critic_full_obs')
        S_obj = Input(shape=[OBJ_STATE], name = 'critic_obj_input') 
        A = Input(shape=[action_dim],name='action2')


        # CH1= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(S)
        # CH2= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(CH1)
        # CH3= Conv2D(32, (4, 4), kernel_initializer='he_uniform', strides=(2,2), padding='same', activation='relu')(CH2)
        # F = Flatten()(CH3)
        # BF = BatchNormalization()(F)
        # BA = BatchNormalization()(A)
        # WTF!!!
        Concatd = Concatenate()([S_full, S_obj, A])
        FC1 = Dense(256, kernel_initializer='he_uniform',activation='relu')(Concatd)
        BFC1 = BatchNormalization()(FC1)
        FC2 = Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC1)
        BFC2 = BatchNormalization()(FC2)
        FC3 = Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC2)
        BFC3 = BatchNormalization()(FC3)
        FC4 = Dense(256, kernel_initializer='he_uniform',activation='relu')(BFC3)
        BFC4 = BatchNormalization()(FC4)        
        # Q = Dense(1, kernel_initializer=lambda shape:VarianceScaling(scale=3e-3)(shape),activation='linear')(BFC4) # Q is scalar value!! dimenstion should be 1
        Q = Dense(1, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='linear')(BFC4) # Q is scalar value!! dimenstion should be 1
        model = Model(input=[S_full, S_obj, A],output=Q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)  # Critic's Loss ftn is defined from MSE!!

        return model, A, S_full, S_obj

