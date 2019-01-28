import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import tensorflow as tf
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
from bc_logger import DataLog
from tqdm import tqdm # progress monitor module!
import os
import rospy

import core
from core import get_vars
from utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from core import get_vars #for retrieving weight vectors
from utils.logx import EpochLogger

##
import pickle
import json
import joblib
import shutil
import os.path as osp, time, atexit, os
import rospy
# from utils.mpi_tools import proc_id, mpi_statistics_scalar
# from utils.serialization_utils import convert_json

# def get_vars(scope=''):
#     return [x for x in tf.trainable_variables() if scope in x.name]

from std_srvs.srv import Empty, EmptyRequest


## Hyper params for ppo policy network!
# ORIGINAL_POLICY_LR = 1E-3

PPO_POLICY_LR=3e-4

DEMO_DATA = 'traj_dagger.bin'

BC_LOGDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/bc_pretrain/'
BC_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/bc_pretrain/weights/behav_cln'
BC_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/bc_pretrain/weights/behav_cln'


## Hyper params for ppo policy network!


# note that it's ON POLICY!!!


class BC:
    def __init__(self, sess=None, epochs = 1, batch_size = 128, lr = 1e-3,
                 optimizer = None, seed=0, exp_name='', output_dir='', env=None, pi_optimizer=None, summary_writer=None):

        # define placeholders_ for actor...
        self.env = env
        self.sess = sess
        self.pi_optimizer = pi_optimizer
        self.summary_writer = summary_writer


        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None,100,100,3]) # ph for BC policy
        self.act_ph = tf.placeholder(dtype=tf.float32, shape=[None,3]) # ph for BC policy
        self.t_act_ph = tf.placeholder(dtype=tf.float32, shape=[None,3]) # ph for BC policy
        


        with tf.variable_scope('pi', reuse=True):
            self.policy = core.cnn_gaussian_policy # check the graph on Tensorboard
            self.pi, _, _ = self.policy(self.obs_ph, self.act_ph)
            self.bc_lr = lr

            self.bc_mse_loss = tf.reduce_mean(tf.square(self.t_act_ph - self.pi))
     


            # self.bc_optimizer = MpiAdamOptimizer(learning_rate = self.bc_lr)
            self.grads_and_vars_bc = self.pi_optimizer.compute_gradients(self.bc_mse_loss, var_list=get_vars('pi'))

            self.optimize_bc = self.pi_optimizer.apply_gradients(self.grads_and_vars_bc)

        s = []
        s.append(tf.summary.scalar('BC_loss', self.bc_mse_loss,family='BC_summary'))

        with tf.variable_scope('BC_summary'):
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for grad, var in self.grads_and_vars_bc:
                s.append(tf.summary.histogram(var.op.name + '', var, family='BC_summary'))
                if grad is not None:
                    s.append(tf.summary.histogram(var.op.name + '/gradients', grad, family='BC_summary'))
                    s.append(tf.summary.histogram(var.op.name + '/gradients/norm', l2_norm(grad), family='BC_summary'))

        self.BC_sumamry_op = tf.summary.merge(s)




        rospy.loginfo("Setup logger for behaviour cloning")
        self.BC_logger = EpochLogger(output_dir=BC_LOGDIR,exp_name=exp_name, seed=seed)
        self.BC_logger.setup_tf_saver(sess, inputs={'obs': self.obs_ph}, outputs={'pi': self.pi})

        # policy to be learned, and its weights will be copied to that of actor

        # return values of cnn gaussian policy pi, logp, logp_pi
        # logp and logp_pi won't be need since we're doing supervised learning here

        # self.action_pred, _, _ = core.cnn_gaussian_policy(self.obs_ph, self.act_ph)
        # self.action_pred, _, _ = policy(self.obs_ph, self.act_ph)
  
        # self.expert_paths = expert_pathfs
        self.epochs = epochs
        self.save_freq = 2
        self.mb_size = batch_size
        self.logger = DataLog()

        # self.expert_paths = expert_paths


        self._observations = list() # empty array with no shape
        self._actions = list() # empty array with no shape

        # Extract and save demo data on initializations
        os.chdir('/home/irobot/catkin_ws/src/ddpg/scripts/ppo')

        # how demo transition consists of
        # obs_t, obs_t+1, act_t, disc_rew_sum, reward_t, obs_t+N, done, lisDemo
        #   O       X      O            X         X         X       X      X

        if os.path.exists(DEMO_DATA):
            rospy.loginfo("retrieving demonstration data")
            with open(DEMO_DATA, 'rb') as f:
                _demo = pickle.load(f)
                for idx, item in enumerate(_demo):
                    # add observations
                    self._observations.append(item[0][0]) # obs_t
                    # add actions
                    self._actions.append(item[2]) # act_t                  
                print (idx, 'data has retrieved')

        # on-policy demo data



        # We just require color observation (index-0)
        self.observations = np.array(self._observations)
        self.observations = np.reshape(self.observations,(-1,100,100,3))

        self.actions = np.array(self._actions)
        self.actions = np.reshape(self.actions,(-1,3))



            # apply param clipping on PPO

        # self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr) if optimizer is None else optimizer

        # loss criterion is MSE for maximum likelihood estimation
        # self.loss_function = torch.nn.MSELoss()


        # ACTION DONT HAVE TO BE FED FOR YIELDING 'PI'
        '''
            log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp = gaussian_likelihood(act, mu, log_std)
            logp_pi = gaussian_likelihood(pi, mu, log_std)
            return pi, logp, logp_pi

        '''
        # ACTION DONT HAVE TO BE FED FOR YIELDING 'PI'
    def update_summary(self, summary_writer = None, bc_summray_str=None, global_step=None):
        # self.summar/y_writer = summary_writer
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, i + 1)
        # self.bc_summray_str = bc_summray_str
        summary_writer.add_summary(bc_summray_str, global_step)


    def train_bc(self): # use session initiated in PPOs
        observations =self.observations
        actions = self.actions
        # params_before_opt = self.policy.get_param_values()

        ts = timer.time()
        step = 0
        num_samples = self.observations.shape[0]
        print num_samples
        for ep in tqdm(range(self.epochs)):
            mb = 0
            while not rospy.is_shutdown() and mb <=int(num_samples / self.mb_size):
                # for mb in range(int(num_samples / self.mb_size)): # sample_size/batch_size
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                # run session here!

                train_op = [self.BC_sumamry_op, self.optimize_bc] # acquire loss from train_Op
                train_fd = {self.t_act_ph: act,
                            self.obs_ph: obs,}                
                result = self.sess.run(train_op ,feed_dict=train_fd)
                # KEY POINT is 
                # How to retrieve meaniningful info from this dictionary
                self.logger.log_kv('epoch', ep)
                # loss_val = self.loss(observations, actions).data.numpy().ravel()[0]
                # self.logger.log_kv('loss', result[0])
                self.logger.log_kv('accum_time', (timer.time()-ts))
                
                if mb%10:
                    self.update_summary(self.summary_writer, result[0], step+1)
                step +=1
                mb +=1
                print ('============ EPOCH: '+str(ep)+' ============  TIME: ' +str(timer.time()-ts))


            if (ep % self.save_freq == 0) or (ep == self.epochs-1):
                self.BC_logger.save_weight(BC_MODEL_SAVEDIR, sess=self.sess, 
                    var_list=get_vars('pi'), step=step)
                # print ('LOSS: '+str(result[0])+' ============ EPOCH: '+str(ep)+' ============  TIME: ' +str(timer.time()-ts))
                    # self.BC_logger.save_state({'env': self.env}, None)


                # self.optimizer.zero_grad()
                # loss = self.loss(obs, act)
                # loss.backward()
                # self.optimizer.step()
        # do we have to apply std clipping here?
        # params_after_opt = self.policy.get_param_values()
        # self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        self.logger.log_kv('epoch', self.epochs)
        self.logger.log_kv('time', (timer.time()-ts))        
        # save network weights here!




# inputs = {k:v for k,v in zip(all_phs, buf.get())} # code style!
# pi_l_old, v_l_old, ent, pi_summary_str, v_summary_str = sess.run([pi_loss, v_loss, 
#                                     approx_ent, pi_summary_op, v_summary_op], feed_dict=inputs)

### BC policy should have following network architecture!!!!
### BC policy should have following network architecture!!!!

