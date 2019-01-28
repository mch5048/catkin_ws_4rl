#!/usr/bin/env python
import numpy as np
import math
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling, RandomUniform
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import  Dense, Flatten, Input, Concatenate,Conv2D, Activation, LeakyReLU, Lambda
from keras.layers.normalization import BatchNormalization
from keras import losses

from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
# for layer norm
import tensorflow.contrib as tc

STATE_LENGTH = 3 # 3 consecutive frames will be fed into both Actor and Critic Networks
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_POSE_STATE =7
CHANNELS = 3 # RGB Kinect-v1
LAMBDA_BC = 10.0
L2_COEF = 0.01

class ActorNetwork(object):
    def __init__(self, sess, state_rgb, state_robot, action_size, goal_rgb, BATCH_SIZE, TAU, LEARNING_RATE, critic_full_input, critic_obj_input, critic_goal_state, critic_model):
        # self.summary_writer =summary_writer
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
        with tf.variable_scope('Actor'):
            self.model , self.weights, self.state_rgb, self.state_rob, self.goal_rgb= self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb, scope='eval_net', trainable=True)
            self.target_model, self.target_weights, self.target_state_rgb, self.target_state_rob, self.target_goal_rgb= self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb, scope='target_net', trainable=False)
        # self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        # self.sampled_action = tf.placeholder(tf.float32,[None,action_size])
        # self.isDemo = tf.placeholder(tf.bool,[None,1], name='isDemo')
        # self.action_pred = tf.placeholder(tf.float32,[None,7], name='action_err_for_bc_loss')        
                        

        # We cannot use target_weights!! cuz its trainable_weights=False

        # self.critic_target_pred = tf.placeholder(tf.float32,[None,1], name='critic_target_pred')        
        # self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient) #ys,  xs,  grad_ys=None negative gradient!!
        # grads = zip(self.params_grad, self.weights)
        # self.sess.run(tf.initialize_all_variables())
        # self.sess.run(tf.global_variables_initializer())
        # actor_updates = Adam().get_updates(params=self.weights, loss=-K.mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0]))
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob],[self.model.output], updates=actor_updates)
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob, self.critic_full_input, self.critic_obj_input, self.model.output],[self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0]], updates=actor_updates)
        # self.actor_train_fn = K.function([self.state_rgb, self.state_rob, self.critic_full_input, self.model.output],[self.model.output], updates=actor_updates)
        # keras.backend.function(inputs, outputs, updates=None)
        # self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output]))
        with tf.variable_scope('actor_loss'):            
            self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.model([self.state_rgb, self.state_rob, self.goal_rgb])])[0], name='actor_main_loss')
            # Behaviour cloning
            # tf.where(condition,true state,false state)

            with tf.variable_scope('bc_loss'):
                self.sampled_action = tf.placeholder(tf.float32,[None,7], name='sampled_action_from_batch')        
                self.isDemo = tf.placeholder(tf.float32,[None,7], name='isDemo')
                self.bc_loss = LAMBDA_BC*tf.reduce_sum(tf.square(self.model([self.state_rgb, self.state_rob, self.goal_rgb])-self.sampled_action)*self.isDemo, name='bc_loss_sum') #*tf.cast(self.isDemo,tf.int32)*tf.cast(tf.greater(self.critic_sample_pred, self.critic_target_pred), tf.int32) # x>y # network prediction, sample action
            
            self.actor_total_loss = tf.add(tf.negative(self.actor_loss), self.bc_loss, name='actor_total_loss')
        # self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0])
        # self.actor_grads = self.flatgrad(self.actor_loss, self.weights)
        # print len(self.weights)
        # print '=================================================================================================================='

        self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval_net')
        # print len(self.eval_params)
        # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        # a = set(self.eval_params).difference(self.weights)
        # print a 
        self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/target_net') # used for training
        # initialize each nework!

        # How we get trainable vars 
        # def trainable_vars(self):
        #     return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # clip_norm=self.clip_norm
        #  Can we apply clip norm for actor gradients ???
        
        # self.actor_optimizer = tf.train.AdamOptimizer(-LEARNING_RATE, name='actor_optimzer') # negated learning rate is important!!!!
        self.actor_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, name='actor_optimzer') # negated learning rate is important!!!!

        # self.grads_and_vars_actor = self.actor_optimizer.compute_gradients(self.actor_total_loss, var_list=self.weights)
        # self.optimize_actor = self.actor_optimizer.apply_gradients(self.grads_and_vars_actor, name='optimize_actor')

        # setup actor training sumamry


        # reference
        # self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        # grads = zip(self.params_grad, self.weights)
        # self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        # self.sess.run(tf.initialize_all_variables()

        # with tf.variable_scope('compute_gradients_for_actor'):
        #     self.action_pred = self.model([self.state_rgb, self.state_rob, self.goal_rgb]) 
        #     self.value_grads = tf.gradients(self.critic_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.action_pred])[0], self.action_pred, name='value_grad')[0]  #dQ/da
        #     self.action_grad_to_actor = tf.gradients(ys=self.action_pred, xs=self.weights, grad_ys=self.value_grads, name='action_grad') # negative grad to dQ/da



        with tf.variable_scope('compute_gradients_for_actor'):
            self.action_pred = self.model([self.state_rgb, self.state_rob, self.goal_rgb]) 
            self.value_grads = tf.gradients(self.critic_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.action_pred])[0], self.action_pred, name='value_grad')[0]  #dQ/da
            self.params_grad = tf.gradients(self.action_pred, self.weights, -self.value_grads) # negative is the CORE!!
            grads = zip(self.params_grad, self.weights)
            self.train_actor_op = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

            self.action_grad_to_actor = tf.gradients(ys=self.action_pred, xs=self.weights, grad_ys=self.value_grads, name='train_actor') # negative grad to dQ/da


        # tf.summary.image('image', sliced_image, max_outputs=3)
        s = []
        s.append(tf.summary.histogram('value_grads', self.value_grads, family='actor_summary'))

        # for grad, var in zip(self.action_grad_to_actor, self.weights):
        for grad, var in grads:
            s.append(tf.summary.histogram(var.op.name + '', var, family='actor_summary'))
            s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='actor_summary'))
        self.actor_summary_op = tf.summary.merge(s)
        # self.train_actor_op = self.actor_optimizer.apply_gradients(zip(self.action_grad_to_actor, self.weights), name='train_actor')


    def train_actor_with_gradient(self, critic_full_input, critic_obj_input, critic_goal_state, state_rgb, state_rob, goal_rgb):

        with tf.variable_scope('update_actor_param'):
            train_op = [self.train_actor_op, self.actor_summary_op]
            self.train_out = self.sess.run(train_op, feed_dict={   
                self.critic_full_input: critic_full_input,
                self.critic_goal_state: critic_goal_state,
                self.critic_obj_input: critic_obj_input,  
                self.state_rgb: state_rgb,  # feed_dict for critic
                self.state_rob: state_rob,  # feed_dict for critic
                self.goal_rgb: goal_rgb  # feed_dict for critic
            })     



    def update_actor_summary(self, summary_writer = None, global_step=None):
        # self.summar/y_writer = summary_writer
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, i + 1)
        if self.train_out[1]:
            summary_str = self.train_out[1]
            summary_writer.add_summary(summary_str, global_step)


    def init_network(self):
        print 'Executes update for both eval and target net'
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] =  actor_weights[i]
        self.target_model.set_weights(actor_target_weights)

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

    # def target_train(self): # It became problematic!!
    #     actor_weights = self.eval_params
    #     actor_target_weights = self.target_params
    #     for i in xrange(len(actor_weights)):
    #         actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
    #     self.target_model.set_weights(actor_target_weights)


    # def target_train(self):
    #     print 'Executes target network update'
    #     self.eval_params
    #     self.target_params
    #     soft_updates = []
    #     assert len(self.eval_params) == len(self.target_params)
    #     for var, target_var in zip(self.eval_params, self.target_params):
    #         soft_updates.append(tf.assign(target_var, (1. - self.TAU) * target_var + self.TAU * var))
    #     assert len(soft_updates) == len(self.eval_params)
    #     soft_update_op = tf.group(*soft_updates)
    #     self.sess.run(soft_update_op)

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

    def Layer_Normalization(self, x):
        return tc.layers.layer_norm(x, center=True, scale=True)

# model.add(Lambda(frac_max_pool))


    def create_actor_network(self, state_rgb, state_robot, action_dim, goal_rgb, scope, trainable):
        ## Assymetric Actor ##
        with tf.variable_scope(scope):
            print("Creates Actor Network")
            # Default data_format -> channels last : shape = (samples, rows, cols, channels)
            with tf.name_scope('observations'):
                S_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_rgb_input') 
                S_robot = Input(shape=[ROBOT_POSE_STATE], name='actor_rob_pose_input') 
                G_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_goal_rgb_input') 
            AH1= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS),name='actor_conv1' ,trainable=trainable)(S_rgb)
            LN1 = Lambda(self.Layer_Normalization, name='layer_norm1')(AH1) # Q is scalar value!! dimenstion should be 1
            A1 = Activation('tanh')(LN1)

            AH2= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_conv2',trainable=trainable)(A1)
            LN2 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AH2) # Q is scalar value!! dimenstion should be 1
            A2 = Activation('tanh')(LN2)

            AH3= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_conv3',trainable=trainable)(A2)
            # LN3 = Lambda(self.Layer_Normalization, name = "layer_norm3")(AH3) # Q is scalar value!! dimenstion should be 1
            # A3 = Activation('tanh')(LN3)

            # AH4= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_conv4',trainable=trainable)(A3)
            # LN4 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AH4) # Q is scalar value!! dimenstion should be 1
            #################################################################################################################################################
            AGH1= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_goal_conv1',trainable=trainable, input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(G_rgb)
            LNG1 = Lambda(self.Layer_Normalization, name = "layer_norm4")(AGH1) # Q is scalar value!! dimenstion should be 1
            A4 = Activation('tanh')(LNG1)

            AGH2= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_goal_conv2',trainable=trainable)(A4)
            LNG2 = Lambda(self.Layer_Normalization, name = "layer_norm5")(AGH2) # Q is scalar value!! dimenstion should be 1
            A5 = Activation('tanh')(LNG2)

            AGH3= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_goal_conv3',trainable=trainable)(A5)
            # LNG2 = Lambda(self.Layer_Normalization, name = "layer_norm6")(AGH3) # Q is scalar value!! dimenstion should be 1
            # A6 = Activation('tanh')(LNG2)

            # AGH4= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer='he_uniform', padding='valid', name='actor_goal_conv4',trainable=trainable)(A6)
            # LNG3 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AGH4) # Q is scalar value!! dimenstion should be 1

            #################################################################################################################################################
            # Concatd = Concatenate(axis=-1)([AH24, AGH4])
            Concatd = Concatenate(axis=-1)([AH3, AGH3])
            F = Flatten()(Concatd)
            # LNFC1 = Lambda(self.Layer_Normalization, name = "layer_norm7")(F) # Q is scalar value!! dimenstion should be 1
            # A7 = Activation('tanh')(LNFC1)
            Concatd2 = Concatenate()([F, S_robot])

            FC1 = Dense(512, kernel_initializer='he_uniform', name='actor_fc1',trainable=trainable)(Concatd2)
            LNFC2 = Lambda(self.Layer_Normalization, name = "layer_norm8")(FC1) # Q is scalar value!! dimenstion should be 1
            A8 = Activation('tanh')(LNFC2)

            # FC1= Dense(256, kernel_initializer='he_uniform',activation='tanh')(BF)
            # BFC1 = BatchNormalization()(Concatd)
            # FC2= Dense(256, kernel_initializer='he_uniform',activation='tanh')(BFC1)
            FC2= Dense(512,kernel_initializer='he_uniform', name='actor_fc2',trainable=trainable)(A8)
            LNFC3 = Lambda(self.Layer_Normalization, name = "layer_norm9")(FC2) # Q is scalar value!! dimenstion should be 1
            A9 = Activation('tanh')(LNFC3)         


           # AFC2 = LeakyReLU()(FC2)            
            # FC3= Dense(512, kernel_initializer='he_uniform', name='actor_fc3',trainable=trainable)(BFC2)
            # AFC3 = LeakyReLU()(FC3)            
            # BFC3 = BatchNormalization()(AFC3)
            # FC3= Dense(300, kernel_initializer='he_uniform', activation='relu')(BFC2)
            # BFC3 = BatchNormalization()(FC3)
           
            Action= Dense(action_dim, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='tanh',name='action_out',trainable=trainable)(A9)
            # Action= Dense(action_dim, kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape),activation='tanh')(BFC4)
            adam = Adam(lr=self.LEARNING_RATE)
           
            model = Model(input= [S_rgb,S_robot, G_rgb],output=Action)
            # model.compile(loss='mse', optimizer=adam)
            # model.compile(loss='mse', optimizer=adam)
            if scope=='eval_net':
                model.summary()
            return model, model.trainable_weights, S_rgb, S_robot, G_rgb