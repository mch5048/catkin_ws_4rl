import tensorflow as tf
import numpy as np
from mpi4py import MPI


class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None, sess=None):
        """
        A parallel MPI implementation of the Adam optimizer for TensorFlow
        https://arxiv.org/abs/1412.6980

        :param var_list: ([TensorFlow Tensor]) the variables
        :param beta1: (float) Adam beta1 parameter
        :param beta2: (float) Adam beta1 parameter
        :param epsilon: (float) to help with preventing arithmetic issues
        :param scale_grad_by_procs: (bool) if the scaling should be done by processes
        :param comm: (MPI Communicators) if None, MPI.COMM_WORLD
        :param sess: (TensorFlow Session) if None, tf.get_default_session()
        """
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(tf_utils.numel(v) for v in var_list)
        # Exponential moving average of gradient values
        # "first moment estimate" m in the paper
        self.exp_avg = np.zeros(size, 'float32')
        # Exponential moving average of squared gradient values
        # "second raw moment estimate" v in the paper
        self.exp_avg_sq = np.zeros(size, 'float32')
        self.step = 0
        self.setfromflat = tf_utils.SetFromFlat(var_list, sess=sess)
        self.getflat = tf_utils.GetFlat(var_list, sess=sess)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, local_grad, learning_rate):
        """
        update the values of the graph

        :param local_grad: (numpy float) the gradient
        :param learning_rate: (float) the learning_rate for the update
        """
        if self.step % 100 == 0:
            self.check_synced()
        local_grad = local_grad.astype('float32')
        global_grad = np.zeros_like(local_grad)
        self.comm.Allreduce(local_grad, global_grad, op=MPI.SUM)
        if self.scale_grad_by_procs:
            global_grad /= self.comm.Get_size()

        self.step += 1
        # Learning rate with bias correction
        step_size = learning_rate * np.sqrt(1 - self.beta2 ** self.step) / (1 - self.beta1 ** self.step)
        # Decay the first and second moment running average coefficient
        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * global_grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (global_grad * global_grad)
        step = (- step_size) * self.exp_avg / (np.sqrt(self.exp_avg_sq) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        """
        syncronize the MPI threads
        """
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        """
        confirm the MPI threads are synced
        """
        if self.comm.Get_rank() == 0:  # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

