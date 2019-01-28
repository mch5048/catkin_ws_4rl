#!/usr/bin/env python


import numpy as np
import tensorflow as tf
# import gym
import core
from core import get_vars
from utils.logx import EpochLogger
from utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# import sawyerEnv
from new_robotEnv_PPO_4D import robotEnv # REFACTOR TO REMOVE GOALS
import rospy
from behavior_cloning import BC    
import time as timer
import time
from std_srvs.srv import Empty, EmptyRequest
from running_mean_std import RunningMeanStd
from SetupSummary import SummaryManager_PPO as SummaryManager
import rospy


path = '/home/irobot/catkin_ws/src/ddpg/scripts/'
log_path = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp'
PPO_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/trained/ppo_trained'
PPO_VALUE_MODEL_SAVEDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/value_trained/ppo_trained'
PPO_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/trained/'
PPO_VALUE_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/value_trained/'
BC_MODEL_LOADDIR = '/home/irobot/catkin_ws/src/ddpg/scripts/ppo_exp/bc_pretrain/weights/'

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Let's define the state/observation for asymmetric A-C PPO


    Actor : 100,100,3 RGB observation from Sawyer reacher env. 



    """
        # obs_dim = [100, 100, 3] # for actor
        # stt_dim = 21 # for critic
        # act_dim = 3
        # obs_dim, stt_dim, act_dim
        # size >> buffer_size
    # buf = PPOBuffer(obs_dim, stt_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def __init__(self, obs_dim, stt_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(shape=(size,)+tuple(obs_dim), dtype=np.float32) # for Actor , [-1,100,100,3]
        self.stt_buf = np.zeros(shape=(size, stt_dim), dtype=np.float32) # for Critic [-1,21], consists for pos, vel, eff
        self.act_buf = np.zeros(shape=(size, act_dim), dtype=np.float32) # 3 dim for action
        
        self.adv_buf = np.zeros(size, dtype=np.float32) # advtg estimates from critic network
        self.rew_buf = np.zeros(size, dtype=np.float32) # sparse reward 
        
        self.ret_buf = np.zeros(size, dtype=np.float32) # return buffer
        self.val_buf = np.zeros(size, dtype=np.float32) # Critic's value estimate for baseline
        self.logp_buf = np.zeros(size, dtype=np.float32) # Log prob buffer
        self.gamma, self.lam = gamma, lam # ???
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, stt, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer. should be called in rollout loop
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.stt_buf[self.ptr] = stt
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # create slicing syntax
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] # TD resiual from start to END-1
        # TD-residual = Estimate[r_t + gamma*Value(s_t+1)-Value(s_t)]        

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """


        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        ###########################
        # Advantage normalization
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.stt_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

log_p indicates the stochastic policy

"""
def update_summary(summary_writer = None, pi_summray_str=None, v_summray_str=None, global_step=None):
    # self.summar/y_writer = summary_writer
    # summary_str = sess.run(summary_op)
    # summary_writer.add_summary(summary_str, i + 1)
        pi_summary_str = pi_summray_str
        v_summary_str = v_summray_str
        summary_writer.add_summary(pi_summary_str, global_step)
        summary_writer.add_summary(v_summary_str, global_step)

def randomize_world():

    # We first wait for the service for RandomEnvironment change to be ready
    # rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/dynamic_world_service')
    # rospy.loginfo("Service /dynamic_world_service READY")
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()

    dynamic_world_service_call(change_env_request)

# implement normalizer here!


# implement normalizer here!

def normalize(ndarray, stats):
    if stats is None:
        return ndarray
    return (ndarray - stats.mean) / stats.std


def denormalize(ndarray, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return ndarray
    denorm_ndarray = ndarray * stats.std + stats.mean

    return denorm_ndarray




def ppo(train_indicator=0, isReal=False, logger_kwargs=dict()):
    """
        PEP8-style
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # learning hyper-parameters
    seed=0 
    steps_per_epoch=2000
    epochs=100
    gamma=0.97
    clip_ratio=0.05
    # pi_lr=3e-4
    pi_lr=1e-4
    vf_lr=1e-3
    train_pi_iters=80
    train_v_iters=80
    lam=0.97
    max_ep_len=500
    target_kl=0.01

    save_freq=5


    # for test purpose!
    isReal =isReal

    BC_PRETRAIN = True
    TRAIN_BC_ONLY = False # this determines whether to train PPO agent or not 

    exp_name = 'ppo_first'

    ##
    logger = EpochLogger(output_dir=log_path,exp_name=exp_name, seed=seed)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    ##

    env = robotEnv(train_indicator=train_indicator)


    rospy.loginfo("Evironment has been created")


    actor_critic = core.ppo_actor_critic


    obs_dim = [100, 100, 3] # for actora
    stt_dim = 24 # for critic
    act_dim = 3
    
    '''
            def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_obs()['observation'].shape,
            dtype=np.float32)

    '''
    # Inputs to computation graph
    # x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    # Define placeholders for both actor and critic
    # tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

    obs_ph = tf.placeholder(dtype=tf.float32, shape=[None,100,100,3]) # for actor
    stt_ph = tf.placeholder(dtype=tf.float32, shape=[None,24]) # for critic
    act_ph = tf.placeholder(dtype=tf.float32, shape=[None,3]) # used for cnn gaussian policy!
    # adv_ph = tf.placeholder(dtype=tf.float32, shape=()) # used for cnn gaussian policy!
    # ret_ph = tf.placeholder(dtype=tf.float32, shape=()) # used for cnn gaussian policy!
    # logp_old_ph = tf.placeholder(dtype=tf.float32, shape=()) # used for cnn gaussian policy!

    adv_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # used for cnn gaussian policy!
    ret_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # used for cnn gaussian policy!
    logp_old_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # used for cnn gaussian policy!



    rospy.loginfo("placeholder have been created")

    # advtg ftn, return, log_prob
    # adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, value = actor_critic(obs_ph, stt_ph, act_ph) # check for its sanity


    # Need all placeholders in *this* order later (to zip with data from buffer)
    # list of placeholders 
    all_phs = [obs_ph, stt_ph, act_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, value, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # print 'LOCAL STEPS PER EPOCH LOCAL STEPS PER EPOCH LOCAL STEPS PER EPOCH'
    # print local_steps_per_epoch

    buf = PPOBuffer(obs_dim, stt_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph) # clipping
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - value)**2) # PPO learns from MC estimation

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio)) # clipped PPO
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
    # MLE loss?

    rospy.loginfo("computational ops have been created")


    # Optimizers >> divide to enrich info?
    pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
    v_optimizer = MpiAdamOptimizer(learning_rate=vf_lr)

    rospy.loginfo("optimizers have been created")

    # train_pi = pi_optimizer.minimize(pi_loss)
    # train_v = v_optimizer.minimize(v_loss)

    # train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    # train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)


    # for tensorboard monitoring
    grads_and_vars_pi = pi_optimizer.compute_gradients(pi_loss, var_list=get_vars('pi'))
    grads_and_vars_v = v_optimizer.compute_gradients(v_loss, var_list=get_vars('v'))

    rospy.loginfo("gradient ops have been set")


    train_pi = pi_optimizer.apply_gradients(grads_and_vars_pi)
    train_v = v_optimizer.apply_gradients(grads_and_vars_v)

    rospy.loginfo("optimization ops have been set")


    # train_pi = pi_optimizer.minimize(pi_loss)
    # train_v = v_optimizer.minimize(v_loss)


    l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))

    # setup policy summary # setup policy summary # setup policy summary
    pi_s = []
    for pigrad, pivar in grads_and_vars_pi:
        pi_s.append(tf.summary.histogram(pivar.op.name + '', pivar, family='policy_summary'))
        if pigrad is not None and  'log_std' not in pivar.name:
            pi_s.append(tf.summary.histogram(pivar.op.name + '/gradients', pigrad, family='policy_summary'))
            pi_s.append(tf.summary.histogram(pivar.op.name + '/gradients/norm', l2_norm(pigrad), family='policy_summary'))
    pi_s.append(tf.summary.scalar('actor_loss', pi_loss,family='policy_summary'))

    pi_summary_op = tf.summary.merge(pi_s)




    # setup value summary # setup value summary # setup value summary

    v_s = []
    for vgrad, vvar in grads_and_vars_v:
        v_s.append(tf.summary.histogram(vvar.op.name + '', vvar, family='value_summary'))
        if vgrad is not None:
            v_s.append(tf.summary.histogram(vvar.op.name + '/gradients', vgrad, family='value_summary'))
            v_s.append(tf.summary.histogram(vvar.op.name + '/gradients/norm', l2_norm(vgrad), family='value_summary'))
    v_summary_op = tf.summary.merge(v_s)
    pi_s.append(tf.summary.scalar('critic_loss', v_loss,family='value_summary'))

    rospy.loginfo("tensorboard has been set")


    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3




    sess = tf.Session(config=config)

    rospy.loginfo("initializing global variables...")

    sess.run(tf.global_variables_initializer())

    rospy.loginfo("global variables have been initialized, NOW SYNCS ALL PARAMS")


    # Sync params across processes
    sess.run(sync_all_params())

    rospy.loginfo("GLOBAL PARAMS ARE ALL SYNCED")



    summary_writer = tf.summary.FileWriter(path + 'ppo_exp/summary/', sess.graph)
                    #   color obs    pos     vel     eff    obj
    obs_shape_list = [(1,100,100,3), (1,7), (1,7), (1,7), (1,3)]

    # setup statistics summary and normalizers 
    summary_manager = SummaryManager(sess=sess, obs_shape_list=obs_shape_list, summary_writer=summary_writer)
    summary_manager.setup_state_summary()
    summary_manager.setup_stat_summary()
    summary_manager.setup_train_summary()
    summary_manager.setup_ppo_summary() # logs ppo specific values

    logger.setup_tf_saver(sess, inputs={'obs': obs_ph, 'stt': stt_ph}, outputs={'pi': pi, 'v': value})

    # execute behaviour cloning
    if BC_PRETRAIN and TRAIN_BC_ONLY:     # 'pi' is Tensor
        bc_agent = BC(sess=sess, epochs=20, batch_size=128, lr=1e-3, seed=seed,exp_name=exp_name, output_dir=log_path,
         env=env, pi_optimizer=pi_optimizer, summary_writer=summary_writer)
        bc_agent.train_bc()
    # then weight is trained, how to retrieve?

    # execute behaviour cloning

    # Setup model saving
    def update(episode): # train @ the end of each episode
        print ('Now updates weight')

        inputs = {k:v for k,v in zip(all_phs, buf.get())} # code style!
        # for key, value in inputs.iteritems():
            # print key
            # print '================='
            # print value.shape
            # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


        # inputs[]
        # pi_l_old, v_l_old, ent, pi_summary_str, v_summary_str = sess.run([pi_loss, v_loss, 
        #                                     approx_ent, pi_summary_op, v_summary_op], feed_dict=inputs)
   

        pi_l_old, v_l_old, ent, pi_summary_str, v_summary_str = sess.run([pi_loss, v_loss, 
                                            approx_ent, pi_summary_op, v_summary_op], feed_dict=inputs)
       
        # Training : trains both actor and critic
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

        print ('Updates PPO summary')
        summary_manager.update_ppo_summary(KL_d=kl, entropy=ent, clipfrac=cf,
                                           delta_pi=(pi_l_new-pi_l_old), 
                                           delta_v=(v_l_new - v_l_old), episode=episode+1)
        update_summary(summary_writer, pi_summary_str, v_summary_str, episode+1)

    #####################################################################################


    rospy.loginfo("EPISODE STARTS")

    if not TRAIN_BC_ONLY: # 
        # LOAD WEIGHTS FOR THE ACTOR!
        if BC_PRETRAIN and not train_indicator: # loads pre-trained weight
            print ("LOADS PRE-TRAINED WEIGHT FOR ACTOR")
            logger.load_weight(BC_MODEL_LOADDIR, sess=sess,var_list=get_vars('pi'))

            print ("=======================================================")
            print ("=======================================================")
            print ("SUCCESSFULLY LOADED PRE-TRAINED WEIGHTS")
            print ("=======================================================")
            print ("=======================================================")
        if train_indicator: # test time >>> load weight 
            print ("LOADS TRAINED POLICY AND VALUE OF PPO")
            logger.load_weight(PPO_MODEL_LOADDIR, sess=sess,var_list=get_vars('pi'))
            logger.load_weight(PPO_VALUE_MODEL_LOADDIR, sess=sess,var_list=get_vars('v'))
            print ("=======================================================")
            print ("=======================================================")
            print ("SUCCESSFULLY LOADED PRE-TRAINED POLICY AND VALUE")
            print ("=======================================================")
            print ("=======================================================")

        # if i == 0:
        #     if not train_indicator:
        #         print 'Loads the mean and stddev for test time'
        #         summary_manager.s_t0_rms.load_mean_std(path+'mean_std0.bin')
        #         summary_manager.s_t1_rms.load_mean_std(path+'mean_std1.bin')
        #         summary_manager.s_t2_rms.load_mean_std(path+'mean_std2.bin')
        #         summary_manager.s_t3_rms.load_mean_std(path+'mean_std3.bin')
        #         summary_manager.s_t4_rms.load_mean_std(path+'mean_std4.bin')



        step = 0
        for epoch in range(epochs):


            start_time = time.time()
            # what should be the 
            c_obs, j_pos, j_vel, j_eff = env.reset(isReal=isReal)

            obj_state_t = env.getObjPose(isReal=isReal)

            # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            full_stt = np.concatenate([np.array(j_pos + j_vel + j_eff),  obj_state_t])
            rew, dn, ep_ret, ep_len = 0, False, 0, 0     
            t = 0
            while not rospy.is_shutdown() and t <int(local_steps_per_epoch):

                t +=1
                # for t in range(local_steps_per_epoch):
                if not train_indicator:
                    randomize_world() # domain randomization

                # normalize obsrvationsb
                full_stt[:7] = normalize(full_stt[:7], summary_manager.s_t1_rms) # joint pos
                full_stt[7:14] = normalize(full_stt[7:14], summary_manager.s_t2_rms) # joint vel
                full_stt[14:21] = normalize(full_stt[14:21], summary_manager.s_t3_rms) # joint eff
                full_stt[21:] = normalize(full_stt[21:], summary_manager.s_t4_rms) # obj pose
                

                act, value_t, logp_t = sess.run(get_action_ops, feed_dict={obs_ph: c_obs.reshape(1,100,100,3),
                                                                        stt_ph: full_stt.reshape(1,24),})
                
                # [pi, value, logp_pi]            
                # save and log
                # obs, stt, act, rew, val, logp
                if not train_indicator:
                    buf.store(c_obs, full_stt, act, rew, value_t, logp_t)
                    
                    logger.store(VVals=value_t)

                #ORIGINALLY o, r, d, _ 

                # pi, logp, logp_pi, _ = actor_critic(obs_ph, stt_ph, act_ph) # check for its sanity
                # dist, color_obs_t_1, joint_pos_t_1, joint_vels_t_1, joint_efforts_t_1, r_t, done 


                dist, c_obs, j_pos, j_vel, j_eff, rew, dn = env.step(act[0], t, isReal=isReal)
                if not train_indicator: # execute on train time 
                    obj_state_t_1 = env.getObjPose(isReal=isReal)
                    # should concatenate for full state`
                    full_stt = np.concatenate([np.array(j_pos + j_vel + j_eff),  obj_state_t_1])

                    # update normalizer and tensorboard manager

                    if ep_len == 0: # update @ start of every episode
                        summary_manager.update_state_summary(epoch+1, c_obs, 
                            full_stt[:7],
                            full_stt[7:14],
                            full_stt[14:21],
                            full_stt[21:])
                        summary_manager.update_stat_summary(step=epoch+1)


                    # a_t[0] = normalize_action(a_t[0]) # Action normalizations
                    # summary_manager.s_t0_rms.update(_rms_s_t_1[0])
                    summary_manager.s_t1_rms.update(full_stt[:7])
                    summary_manager.s_t2_rms.update(full_stt[7:14])
                    summary_manager.s_t3_rms.update(full_stt[14:21])
                    summary_manager.s_t4_rms.update(full_stt[21:])


                    ep_ret += rew
                    ep_len += 1

                if not train_indicator: # execute on train time 
                    terminal = dn or (ep_len == max_ep_len) # logical OR operation, terminate either one is True!
                    if terminal or (t==local_steps_per_epoch-1):

                        if not(terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                        # if trajectory didn't reach terminal state, bootstrap value target

                        last_val = rew if dn else sess.run(value, feed_dict={stt_ph: full_stt.reshape(1,24)})
                        buf.finish_path(last_val)
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            logger.store(EpRet=ep_ret, EpLen=ep_len)
                            summary_manager.update_episode_summary(epi_ret=ep_ret, epi_len=ep_len, episode=epoch+1)  
                        c_obs, j_pos, j_vel, j_eff = env.reset(isReal=isReal)
                        obj_state_t = env.getObjPose(isReal=isReal)

                        rew, dn, ep_ret, ep_len = 0, False, 0, 0  
                step +=1
                # Save model
            if not train_indicator:    
                if (epoch % save_freq == 0) or (epoch == epochs-1):
                    logger.save_weight(PPO_MODEL_SAVEDIR, sess=sess, 
                        var_list=get_vars('pi'), step=step)
                    logger.save_weight(PPO_VALUE_MODEL_SAVEDIR, sess=sess, 
                        var_list=get_vars('v'), step=step)
                        # logger.save_state({'env': env}, None)
                update(epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)


    # How the learning should proceed


    # Instantiate BC and train!

    # ts = timer.time()
    # print("========================================")
    # print("Running BC with expert demonstrations")
    # print("========================================")
    # bc_agent.train()
    # print("========================================")
    # print("BC training complete !!!")
    # print("time taken = %f" % (timer.time()-ts))
    # print("========================================")

    # score = e.evaluate_policy(policy, num_episodes=10, mean_action=True)
    # print("Score with behavior cloning = %f" % score[0][0])

    # ------------------------------
    # Finetune with DAPG
    # print("========================================")
    # print("Finetuning with PPO")
    # print("========================================")    
    # How the learning should proceed
    ppo(train_indicator=1,isReal=True, logger_kwargs=logger_kwargs) # 0 True / 1 False