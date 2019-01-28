import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
from net_utils import *
## Just port to Keras 


EPS = 1e-8


# def combined_shape(length, shape=None):
#     if shape is None:
#         return (length,)
#     return (length, shape)
def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):

    # how to define action_dimension
    act_dim = (-1,3)

    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


# MUST DEFINE CNN GAUSSIAN POLICY # MUST DEFINE CNN GAUSSIAN POLICY # MUST DEFINE CNN GAUSSIAN POLICY

# cnn feature extractor

def cnn_feature_extractor(img_obs): # let's apply bn , adopted from net_utils
    activ = tf.nn.relu
    layer_1 = activ(conv(img_obs, 'actor_conv1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'actor_conv2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2)))
    layer_3 = activ(conv(layer_2, 'actor_conv3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    layer_4 = conv_to_fc(layer_3)
    return layer_4

# actor
def cnn_gaussian_policy(img_obs=None, act=None):

    # how to define action_dimension
    act_dim = 3
    activ = tf.nn.relu


    feature = cnn_feature_extractor(img_obs)

    mu = tf.layers.dense(feature, 100, name='actor_fc1')
    mu = tf.contrib.layers.layer_norm(mu, center=True, scale=True)
    mu = activ(mu) # relu
    mu = tf.layers.dense(mu, 100, name='actor_fc2')
    mu = tf.contrib.layers.layer_norm(mu, center=True, scale=True)
    mu = activ(mu) # relu
    mu = tf.layers.dense(mu, 3, name='mu')


    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(act, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

# critic
def mlp_critic(full_stt):
    activation = 'relu'
    output_activation = 'linear'
    activ = tf.nn.relu


    val = tf.layers.flatten(full_stt)
    val = tf.layers.dense(val, units=100, name='critic_fc1')
    val = tf.contrib.layers.layer_norm(val, center=True, scale=True)
    val = activ(val) # relu
    val = tf.layers.dense(val, units=100, name='critic_fc2')
    val = tf.contrib.layers.layer_norm(val, center=True, scale=True)
    val = activ(val) # relu
    val = tf.layers.dense(val, units=100, name='critic_fc3')
    val = tf.contrib.layers.layer_norm(val, center=True, scale=True)
    val = activ(val) # relu    

    return tf.layers.dense(val, units=1, activation=None)


def ppo_actor_critic(obs, stt, act):

    # default policy builder depends on action space

    policy = cnn_gaussian_policy

    with tf.variable_scope('pi'): # CNN feature extractor + mlp gaussian policy
        pi, logp, logp_pi = policy(obs, act)
    # policy # policy # policy # policy # policy

    with tf.variable_scope('v'):
        val = tf.squeeze(mlp_critic(stt))

    # value # value # value # value # value
    return pi, logp, logp_pi, val


 
# MUST DEFINE CNN GAUSSIAN POLICY # MUST DEFINE CNN GAUSSIAN POLICY # MUST DEFINE CNN GAUSSIAN POLICY



"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v



    # waht we should return?

