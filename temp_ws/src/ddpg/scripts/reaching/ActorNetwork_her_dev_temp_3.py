
#!/usr/bin/env python
import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling, RandomUniform, Orthogonal
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import  Dense, Flatten, Input, Concatenate,Conv2D, Activation, Lambda, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import losses

from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
# for layer norm
import tensorflow.contrib as tc

import tf_util as U
# from mpi_adam import MpiAdam
from mpi4py import MPI
from mpi_tf import MpiAdamOptimizer
import random


STATE_LENGTH = 3 # 3 consecutive frames will be fed into both Actor and Critic Networks
FRAME_WIDTH = 100
FRAME_HEIGHT = 100
ROBOT_POSE_STATE =7
CHANNELS = 3 # RGB Kinect-v1
LAMBDA_BC = 1.0
L2_COEF = 0.01
BHV_CLNG_COEF = [0.1, 0.05, 0.2, 0.3, 0.1, 0.15, 0.15]

ROBOT_FULL_STATE = 21
CRITIC_GOAL_STATE = 14
OBJ_STATE = 3


# TrainActor ver3. Mostly adapted from OpenAI Baseline's implementation




class AdaptiveParamNoise(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance): # naive float value??
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)



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

INIT_NOISE_STDDEV = 0.25
DES_NOISE_STDDEV = 0.15


class ActorNetwork(object):
    # def __init__(self, sess, state_rgb, state_robot, action_size, goal_rgb, BATCH_SIZE, TAU, LEARNING_RATE, critic_full_input, critic_obj_input, critic_goal_state, critic_model):
    def __init__(self, sess, state_rgb, state_robot, action_size, goal_rgb, BATCH_SIZE, TAU, LEARNING_RATE, critic_full_input, critic_obj_input, critic_goal_state, full_dim_robot, state_dim_object, action_dim, goal_critic_dim):
        # self.summary_writer =summary_writer
        # self.critic_model = critic_model
        self.critic_full_input = critic_full_input
        self.critic_obj_input = critic_obj_input
        self.critic_goal_state = critic_goal_state
        self.action_size = action_size

        # for param noise
        # self.noise_mean = float(noise_mean)
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.param_noise = AdaptiveParamNoise(initial_stddev=INIT_NOISE_STDDEV, desired_action_stddev=DES_NOISE_STDDEV)
        self.mean_distance = 0.0
        K.set_session(sess)

        with tf.variable_scope('Critic', reuse=True):
            self.crit_model, self.action, self.state_full,self.state_obj, self.goal_state_critic = self.create_critic_network(full_dim_robot, state_dim_object, action_dim, goal_critic_dim, scope='eval_net', trainable=True)
            self.critic_actor_weights = self.crit_model.get_weights()
                # full_dim_robot, state_dim_object, action_dim, goal_critic_dim
        #Now create the model
        with tf.variable_scope('Actor'):
            self.model , self.weights, self.state_rgb, self.state_rob, self.goal_rgb= self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb, scope='eval_net', trainable=True)
            self.target_model, self.target_weights, self.target_state_rgb, self.target_state_rob, self.target_goal_rgb= self.create_actor_network(state_rgb, state_robot, action_size, goal_rgb, scope='target_net', trainable=False)



        with tf.variable_scope('actor_loss'):     
            self.joint_vel_scale = tf.constant(BHV_CLNG_COEF)  
            self.isPretrain = tf.placeholder(tf.float32,  shape=(), name='is_pretrain')
   
            self.actor_loss = tf.multiply(self.isPretrain, -tf.reduce_mean(self.crit_model([self.critic_full_input, self.critic_obj_input, self.critic_goal_state, self.joint_vel_scale* self.model([self.state_rgb, self.state_rob, self.goal_rgb])])[0]), name='actor_main_loss')
            # Behaviour cloning
            # tf.where(condition,true state,false state)

            with tf.variable_scope('bc_loss'):
                self.sampled_action = tf.placeholder(tf.float32,[None,7], name='sampled_action_from_batch')        
                self.isDemo = tf.placeholder(tf.float32,[None,7], name='isDemo')
                self.bc_loss = LAMBDA_BC*tf.reduce_sum(tf.square(self.joint_vel_scale*self.model([self.state_rgb, self.state_rob, self.goal_rgb])-self.sampled_action)*self.isDemo, name='bc_loss_sum') #*tf.cast(self.isDemo,tf.int32)*tf.cast(tf.greater(self.critic_sample_pred, self.critic_target_pred), tf.int32) # x>y # network prediction, sample action


            self.actor_total_loss = tf.add( self.actor_loss, self.bc_loss, name='actor_total_2loss')
        # self.actor_loss = -tf.reduce_mean(self.critic_model([self.critic_full_input, self.critic_obj_input, self.model.output])[0])
        # print len(self.weights)
        # print '=================================================================================================================='

        self.eval_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/target_net') # used for training
        # initialize each nework!

        # How we get trainable vars 
        # def trainable_vars(self):
        #     return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # clip_norm=self.clip_norm
        #  Can we apply clip norm for actor gradients ???        
        # self.actor_optimizer = tf.train.AdamOptimizer(-LEARNING_RATE, name='actor_optimzer') # negated learning rate is important!!!!
        


        with tf.variable_scope('train_actor'):
            # self.actor_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, name='actor_optimzer') # negated learning rate is important!!!

            # Check feed dict for actor grads 
            # with tf.variable_scope('actor_flatgrad_update'):
                # self.actor_flatgrad = self.flatgrad(self.actor_total_loss, self.weights) # grads are acquired from actor_total_loss

            #Adopt MPI Adam optimizers # OpenAI Baselines # OpenAI Baselines # OpenAI Baselines
            # self.actor_flatgrad, self.grads = U.flatgrad(self.actor_total_loss, self.weights, clip_norm=5.0)                                                
            # self.actor_optimizer = MpiAdam(var_list=self.weights, beta1=0.9, beta2=0.999, epsilon=1e-08)
            with tf .variable_scope('actor_MPIadam'):
                self.actor_optimizer = MpiAdamOptimizer(LEARNING_RATE)
            with tf .variable_scope('actor_flatgrad'):
                # self.actor_flatgrad, self.grads = U.flatgrad(self.actor_total_loss, self.weights, clip_norm=5.0)                                                
                self.actor_flatgrad_vars, self.grads_and_vars = self.actor_optimizer.compute_gradients(self.actor_total_loss, self.weights, clip_norm=5.0)
            with tf.variable_scope('actor_apply_gradients'):
                self.train_actor_op = self.actor_optimizer.apply_gradients(self.actor_flatgrad_vars)
            # OpenAI Baselines # OpenAI Baselines # OpenAI Baselines # OpenAI Baselines # OpenAI Baselines



        if self.param_noise is not None:
            # Configure perturbed actor.
            with tf.name_scope('actor_param_noise'):
                
                #placeholders
                self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
                self.param_noise_distance = tf.placeholder(tf.float32, shape=(), name='param_noise_distance')

        with tf.variable_scope('actor_l2_norm_summary'):
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        s = []
        s.append(tf.summary.scalar('Actor_loss', self.actor_loss,family='actor_summary'))
        s.append(tf.summary.scalar('BehaviourCloning_loss', self.bc_loss,family='actor_summary'))
        # for grad, var in zip(self.grads, self.weights):
        for grad, var in self.grads_and_vars:
            s.append(tf.summary.histogram(var.op.name + '', var, family='actor_summary'))
            if grad is not None:
                s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='actor_summary'))
                s.append(tf.summary.histogram(var.op.name + '/gradients/norm', l2_norm(grad), family='actor_summary'))
        # s.append(tf.summary.histogram(self.actor_flatgrad.op.name + '', self.actor_flatgrad, family='actor_summary'))
        s.append(tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev),family='optional'))    
        s.append(tf.summary.scalar('param_noise_distance', tf.reduce_mean(self.param_noise_distance),family='optional'))    


        # for grad, var in self.grads_and_vars_actor:
        #     s.append(tf.summary.histogram(var.op.name + '', var, family='actor_summary'))
        #     s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='actor_summary'))
        # s.append(tf.summary.image('actor_conv_result_achvd',self.conv_achv_out, family='actor_summary'))    
        # s.append(tf.summary.image('actor_conv_result_achvdresult_goal',self.conv_goal_out, family='actor_summary'))    
        self.actor_summary_op = tf.summary.merge(s)



# Parameter Space Noise # Parameter Space Noise # Parameter Space Noise # Parameter Space Noise # Parameter Space Noise

    def _setup_param_noise(self):
        """
        set the parameter noise operations

        :param normalized_obs0: (TensorFlow Tensor) the normalized observation
        """
        print 'Now setup actors for param noise application'

        with tf.variable_scope("Actor/noise", reuse=False): # configure perturbed actor
            self.param_noise_actor , _, _, _, _= self.create_actor_network(self.state_rgb, self.state_rob, self.action_size, self.goal_rgb, scope='policy', trainable=True)
            # self.perturbed_actor_tf = self.param_noise_actor.make_actor(normalized_obs0)

        with tf.variable_scope("Actor/adapt_noise", reuse=False): # configure separate copy for stddev adoptation 
            self.adaptive_param_noise_actor , _, _, _, _= self.create_actor_network(self.state_rgb, self.state_rob, self.action_size, self.goal_rgb, scope='policy', trainable=True)
            # adaptive_actor_tf = self.adaptive_param_noise_actor.make_actor(normalized_obs0)

        with tf.variable_scope("noise_update_func", reuse=False):
            print ('Setting up parameter noises')
            self.perturb_policy_ops = self.get_perturbed_actor_updates('Actor/eval_net/', 'Actor/noise/policy', self.param_noise_stddev,
                                                                  verbose=0)

            self.perturb_adaptive_policy_ops = self.get_perturbed_actor_updates('Actor/eval_net/', 'Actor/adapt_noise/policy',
                                                                           self.param_noise_stddev,
                                                                           verbose=0)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.model([self.state_rgb, self.state_rob, self.goal_rgb]) 
                - self.adaptive_param_noise_actor([self.state_rgb, self.state_rob, self.goal_rgb]))))


    def _adapt_param_noise(self, state_rgb, state_rob, goal_rgb):
        """
        calculate the adaptation for the parameter noise

        :return: (float) the mean distance for the parameter noise
        """
        # if self.param_noise is None:
        #     return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.

        # normal action and perturbed action shares the same feed_dict

        with tf.name_scope('adapt_param_noise'):
            self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

            distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
                self.state_rgb: state_rgb,
                self.state_rob: state_rob,
                self.goal_rgb: goal_rgb,
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

            self.mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
            self.param_noise.adapt(self.mean_distance)
        # return mean_distance


                                # 'Actor/eval_net'   #'noise/pi'  # tensor
    def get_perturbed_actor_updates(self, actor, perturbed_actor, param_noise_stddev, verbose=0):
        """
        get the actor update, with noise.

        :param actor: (str) the actor
        :param perturbed_actor: (str) the pertubed actor
        :param param_noise_stddev: (float) the std of the parameter noise
        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        :return: (TensorFlow Operation) the update function
        """
        # TODO: simplify this to this:
        # assert len(actor.vars) == len(perturbed_actor.vars)
        # assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)
        with tf.name_scope('get_perturbed_actor_updates'):
            assert len(U.get_globals_vars(actor)) == len(U.get_globals_vars(perturbed_actor))
            assert len([var for var in U.get_trainable_vars(actor) if 'LayerNorm' not in var.name]) == \
                len([var for var in U.get_trainable_vars(perturbed_actor) if 'LayerNorm' not in var.name])

            updates = []

            # check for each weights s






            for var, perturbed_var in zip(U.get_globals_vars(actor), U.get_globals_vars(perturbed_actor)):



                if var in [v for v in U.get_trainable_vars(actor) if 'LayerNorm' not in var.name]:
                    # print 'Debug starts here!'
                    # print '==================================='
                    # print param_noise_stddev
                    # print '==================================='
                    # print var
                    # print '--------------'
                    # print perturbed_var
                    # print '==================================='
                    # print tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)
                    # print '======================================================================'
                    updates.append(tf.assign(perturbed_var,
                                             var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
                else:
                    # print '==================================='
                    # print param_noise_stddev
                    # print '==================================='
                    # print var
                    # print '--------------'
                    # print perturbed_var
                    # print '==================================='
                    # print tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)
                    # print '======================================================================'
                    updates.append(tf.assign(perturbed_var, var))
            assert len(updates) == len(U.get_globals_vars(actor))
            return tf.group(*updates)

# Parameter Space Noise # Parameter Space Noise # Parameter Space Noise # Parameter Space Noise # Parameter Space Noise



    def train_actor_with_flatgrad(self, states_rgb, states_rob, goal_rgb, critic_full_input, critic_obj_input, critic_goal_state, sampled_action, isDemo, isPretrain):
        
        # ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]

        # train_op = [self.actor_flatgrad, self.actor_summary_op]
        # train_op = [self.actor_flatgrad, self.actor_summary_op]
        train_op = [self.train_actor_op, self.actor_summary_op]
        actor_flatgrad, self.train_summary = self.sess.run(train_op, feed_dict={   
            self.state_rgb: states_rgb,
            self.state_rob: states_rob,
            self.goal_rgb: goal_rgb,
            self.critic_full_input: critic_full_input,
            self.critic_obj_input: critic_obj_input,
            self.critic_goal_state: critic_goal_state,
            self.sampled_action : sampled_action,
            self.isDemo : isDemo,
            self.isPretrain : isPretrain,
            self.param_noise_stddev : 0 if self.param_noise is None else self.param_noise.current_stddev,
            self.param_noise_distance : 0 if self.param_noise is None else self.mean_distance,
            })

        # update with flatgrads
        # self.actor_optimizer.update(actor_flatgrad, stepsize=self.LEARNING_RATE)
        # self.actor_optimizer.update(actor_flatgrad, stepsize=self.LEARNING_RATE)
        # self.actor_optimizer.apply_gradients(self.train_out[0])


    

        # print result
    # reference from MpiAdam, @ OpenAI

        # reference OpenAI 
        # self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)

    def var_shape(self, tensor):
        """
        get TensorFlow Tensor shape

        :param tensor: (TensorFlow Tensor) the input tensor
        :return: ([int]) the shape
        """
        out = tensor.get_shape().as_list()
        assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
        return out



    # def flatgrad(self, loss, var_list, clip_norm=None):
    #     """
    #     calculates the gradient and flattens it

    #     :param loss: (float) the loss value
    #     :param var_list: ([TensorFlow Tensor]) the variables
    #     :param clip_norm: (float) clip the gradients (disabled if None)
    #     :return: ([TensorFlow Tensor]) flattend gradient
    #     """
    #     grads = tf.gradients(loss, var_list)        
    #     if clip_norm is not None:
    #         grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    #     return tf.concat(axis=0, values=[
    #         tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
    #         for (v, grad) in zip(var_list, grads)]), grads



    # def train_actor_with_gradient(self, critic_full_input, critic_obj_input, critic_goal_state, state_rgb, state_rob, goal_rgb):

    #     with tf.variable_scope('update_actor_param'):
    #         # train_op = [self.train_actor_op, self.actor_summary_op]
    #         train_op = [self.actor_grads, self.actor_summary_op]
    #         self.train_out = self.sess.run(train_op, feed_dict={   
    #             self.critic_full_input: critic_full_input,
    #             self.critic_goal_state: critic_goal_state,
    #             self.critic_obj_input: critic_obj_input,  
    #             self.state_rgb: state_rgb,  # feed_dict for critic
    #             self.state_rob: state_rob,  # feed_dict for critic
    #             self.goal_rgb: goal_rgb  # feed_dict for critic
    #         })     



    def update_actor_summary(self, summary_writer = None, global_step=None):
        # self.summar/y_writer = summary_writer
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, i + 1)
        if self.train_summary:
            summary_str = self.train_summary
            summary_writer.add_summary(summary_str, global_step)


    def init_network(self):
        print 'Executes update for both eval and target net'
        actor_weightdds = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] =  actor_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # def loss(y_true,y_pred):
    #     return K.mean(K.square(y_pred-y_true) - K.square(y_true-noisy_img))
    # def train_actor(self, states_rgb, states_rob, goal_rgb, critic_full_input, critic_obj_input, critic_goal_state, sampled_action, isDemo):
    #     train_op = [self.optimize_actor, self.actor_summary_op]
    #     self.train_out = self.sess.run(train_op, feed_dict={   
    #         self.state_rgb: states_rgb,
    #         self.state_rob: states_rob,
    #         self.goal_rgb: goal_rgb,
    #         self.critic_full_input: critic_full_input,
    #         self.critic_obj_input: critic_obj_input,
    #         self.critic_goal_state: critic_goal_state,
    #         self.sampled_action : sampled_action,
    #         self.isDemo : isDemo,
    #         })

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

    # tensor: Optional existing tensor to wrap into the Input layer. If set, the layer will not create a placeholder tensor.

    def create_actor_network(self, state_rgb, state_robot, action_dim, goal_rgb, scope, trainable):
        ## Assymetric Actor ##
        with tf.variable_scope(scope):
            print("Creates Actor Network")
            # Default data_format -> channels last : shape = (samples, rows, cols, channels)
            with tf.name_scope('observations'):
                S_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_rgb_input') 
                S_robot = Input(shape=[ROBOT_POSE_STATE], name='actor_rob_pose_input') 
                G_rgb = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS), name='actor_goal_rgb_input') 
            AH1= Conv2D(32, (8, 8), strides=(4,4), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS),name='actor_conv1' ,trainable=trainable)(S_rgb)
            LN1 = Lambda(self.Layer_Normalization, name='layer_norm1')(AH1) # Q is scalar value!! dimenstion should be 1
            A1 = Activation('relu')(LN1)
            # A1 = Activation('relu')(AH1)

            AH2= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_conv2',trainable=trainable)(A1)
            LN2 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AH2) # Q is scalar value!! dimenstion should be 1
            A2 = Activation('tanh')(LN2)
            # A2 = Activation('relu')(AH2)

            AH3= Conv2D(32, (3, 3), strides=(1,1), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_conv3',trainable=trainable)(A2)
            LN3 = Lambda(self.Layer_Normalization, name = "layer_norm3")(AH3) # Q is scalar value!! dimenstion should be 1
            A3 = Activation('relu')(LN3)
            # A3 = Activation('relu')(AH3)

            # AH4= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_conv4',trainable=trainable)(A3)
            # LN4 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AH4) # Q is scalar value!! dimenstion should be 1
            #################################################################################################################################################
            AGH1= Conv2D(32, (8, 8), strides=(4,4), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_goal_conv1',trainable=trainable, input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(G_rgb)
            LNG1 = Lambda(self.Layer_Normalization, name = "layer_norm4")(AGH1) # Q is scalar value!! dimenstion should be 1
            A4 = Activation('relu')(LNG1)
            # A4 = Activation('tanh')(AGH1)

            AGH2= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_goal_conv2',trainable=trainable)(A4)
            LNG2 = Lambda(self.Layer_Normalization, name = "layer_norm5")(AGH2) # Q is scalar value!! dimenstion should be 1
            A5 = Activation('relu')(LNG2)
            # A5 = Activation('tanh')(AGH2)

            AGH3= Conv2D(32, (3, 3), strides=(1,1), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_goal_conv3',trainable=trainable)(A5)
            LNG3 = Lambda(self.Layer_Normalization, name = "layer_norm6")(AGH3) # Q is scalar value!! dimenstion should be 1
            A6 = Activation('relu')(LNG3)
            # A6 = Activation('relu')(AGH3)

            # AGH4= Conv2D(32, (4, 4), strides=(2,2), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), padding='valid', name='actor_goal_conv4',trainable=trainable)(A6)
            # LNG3 = Lambda(self.Layer_Normalization, name = "layer_norm2")(AGH4) # Q is scalar value!! dimenstion should be 1

            #################################################################################################################################################
            # Concatd = Concatenate(axis=-1)([AH24, AGH4])
            # Concatd = Concatenate(axis=-1)([AH3, AGH3])
            Concatd = Concatenate(axis=-1)([A3, A6])
            F = Flatten()(Concatd)
            # LNFC1 = Lambda(self.Layer_Normalization, name = "layer_norm7")(F) # Q is scalar value!! dimenstion should be 1
            # A7 = Activation('relu')(LNFC1)
            Concatd2 = Concatenate()([F, S_robot])

            FC1 = Dense(512, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='actor_fc1',trainable=trainable)(Concatd2)
            LNFC2 = Lambda(self.Layer_Normalization, name = "layer_norm8")(FC1) # Q is scalar value!! dimenstion should be 1
            AFC1 = Activation('relu')(LNFC2)
            # AFC1 = LeakyReLU()(LNFC2)            

            # FC1= Dense(256, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)),activation='relu')(BF)
            # BFC1 = BatchNormalization()(Concatd)
            # FC2= Dense(256, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)),activation='relu')(BFC1)
            FC2= Dense(256,kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='actor_fc2',trainable=trainable)(AFC1)
            LNFC3 = Lambda(self.Layer_Normalization, name = "layer_norm9")(FC2) # Q is scalar value!! dimenstion should be 1
            AFC2 = Activation('relu')(LNFC3)         
            # AFC2 = LeakyReLU()(LNFC3)            

            # AFC2 = Leakyrelu()(FC2)            

            # AFC2 = Leakyrelu()(FC2)            
            # FC3= Dense(512, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='actor_fc3',trainable=trainable)(BFC2)
            # AFC3 = Leakyselu()(FC3)            
            # BFC3 = BatchNormalization()(AFC3)
            # FC3= Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), activation='selu')(BFC2)
            # BFC3 = BatchNormalization()(FC3)
           
            Action= Dense(action_dim, kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None),activation='tanh',name='action_out',trainable=trainable)(AFC2)
            # Action= Dense(action_dim, kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape),activation='tanh')(BFC4)
            # adam = Adam(lr=self.LEARNING_RATE)
           
            model = Model(input= [S_rgb,S_robot, G_rgb],output=Action)
            # model.compile(loss='mse', optimizer=adam)
            # model.compile(loss='mse', optimizer=adam)
            if scope=='eval_net':
                model.summary()
            return model, model.trainable_weights, S_rgb, S_robot, G_rgb

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

            # CH1= Conv2D(32, (4, 4), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), strides=(2,2), padding='same', activation='selu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, CHANNELS))(S)
            # CH2= Conv2D(32, (4, 4), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), strides=(2,2), padding='same', activation='selu')(CH1)
            # CH3= Conv2D(32, (4, 4), kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), strides=(2,2), padding='same', activation='selu')(CH2)
            # F = Flatten()(CH3)
            # BF = BatchNormalization()(F)
            # BA = BatchNormalization()(A)
            # Concatd = Concatenate()([FC_s, A])
            FC1 = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc1',  trainable=trainable)(Concatd)
            LN1 = Lambda(self.Layer_Normalization, name='layer_norm1')(FC1) # Q is scalar value!! dimenstion should be 1
            A1 = Activation('relu')(LN1)

            Concatd_2 = Concatenate(axis=-1)([A1, A])  

            FC2 = Dense(300, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc2',  trainable=trainable)(Concatd_2)
            LN2 = Lambda(self.Layer_Normalization, name='layer_norm2')(FC2) # Q is scalar value!! dimenstion should be 1
            A2 = Activation('relu')(LN2)     

            FC3 = Dense(200, kernel_initializer=keras.initializers.Orthogonal(gain=np.sqrt(2)), name='critic_fc3',  trainable=trainable)(A2)
            LN3 = Lambda(self.Layer_Normalization, name='layer_norm3')(FC3) # Q is scalar value!! dimenstion should be 1
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
