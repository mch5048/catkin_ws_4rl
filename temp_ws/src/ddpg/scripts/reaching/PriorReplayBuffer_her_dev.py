#!/usr/bin/env python
from collections import deque
import random
import numpy as np
import rospy
import operator

# PER Beta scheduler # PER Beta scheduler # PER Beta scheduler # PER Beta scheduler

class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError

class ConstantSchedule(Schedule):
    """
    Value remains constant over time.

    :param value: (float) Constant value of the schedule
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.

    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)



class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# PER Beta scheduler # PER Beta scheduler # PER Beta scheduler # PER Beta scheduler



class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        :param prefixsum: (float) upperbound on the sum of array prefix
        :return: (int) highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)

# define hyper parameters # define hyper parameters # define hyper parameters

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 10000
BETE_INCR_PER_SMPL = 1e-5  # annealing the bias
ABS_ERR_UPR = 1   # for stability refer to paper
DEMO = True

# define hyper parameters # define hyper parameters # define hyper parameters

class ReplayBuffer(object):

    '''
    Implementation of prioritized replay buffer! applied HER, so we have place for goal in here
    '''
    def __init__(self, buffer_size, total_train_steps = 500000,prob_alpha=0.8): # focus on prob_alpha
        self.buffer_size = buffer_size
        self.total_train_steps = total_train_steps
        # self.buffer = deque()
        self.buffer = []
        assert prob_alpha >= 0
        self._alpha = prob_alpha
        self.pos = 0
        # self.priorities = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.beta0 = BETA_START
        # adoptation from OpenAI implementation # OpenAI # OpenAI # OpenAI

        self._storage = []
        # self._maxsize = size
        self._next_idx = 0

        self.beta_schedule = LinearSchedule(self.total_train_steps,
                                    initial_p=self.beta0,
                                    final_p=1.0)


        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0





        # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI

    # Reference # Reference # Reference # Reference # Reference # Reference
    # def sample(self, batch_size, beta=0.4):
    #     if len(self.buffer) == self.capacity:
    #         prios = self.priorities
    #     else:
    #         prios = self.priorities[:self.pos]
    #     probs = prios ** self.prob_alpha

    #     probs /= probs.sum()
        # indices = np.random.choice(len(self.buffer), batch_size, p=probs) # get the array of incices of item in buffer, w.r.t. probs
        # samples = [self.buffer[idx] for idx in indices] # using those probs., we sample from buffer to obtain a batch of sampels
    #     ##################################################################

    #     weights /= weights.max()
    #     return samples, indices, np.array(weights, dtype=np.float32)
    # Reference # Reference # Reference # Reference # Reference # Reference

    def _sample_proportional(self, batch_size):
        _res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            _res.append(idx)
            res = np.asarray(_res)
        return res


    def getBatch(self, batch_size, beta=0.7):
        # Randomly sample batch_size examples

        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        # idxes ->  np_array

        weights = []
        # print '=================Sampled batch indices==================='
        # print idxes
        # print '========================================================='


        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)

        # encoded_sample = self._encode_sample(idxes)

            # return tuple(list(encoded_sample) + [weights, idxes])
            # self.beta = np.min([1, self.beta + BETE_INCR_PER_SMPL])  # anneals the bia by beta increment
            # if len(self.buffer) == self.buffer_size:
            #     prios = self.priorities
            # else:
            #     prios = self.priorities[:self.pos]

            # # implementation of P(i)
            # probs = prios ** self.prob_alpha
            # probs /= probs.sum()

            # indices = np.random.choice(len(self.buffer), batch_size, p=probs) # get the array of incices of item in buffer, w.r.t. probs
        samples = [self._storage[idx] for idx in idxes] # using those probs., we sample from buffer to obtain a batch of sampels
            # total = len(self.buffer)
            # weights = (total * probs[indices]) ** (-self.beta)

        return samples, idxes, np.array(weights, dtype=np.float32) # returns a list()
        

    def getActionBatch(self, batch_size):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        
        action_samples = [self._storage[idx] for idx in idxes]


        return action_samples


    #                0          1       2         3       4       5              6                      7               8      9                            
    def add(self, state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo): # same as poluate()
        # max_prio = self.priorities.max() if self._storage else 1.0
        # My implementation seems to be wiped out...


        # Baseline reference # Baseline reference # Baseline reference
            # data = (obs_t, action, reward, obs_tp1, done)

            # if self._next_idx >= len(self._storage):
            #     self._storage.append(data)
            # else:
            #     self._storage[self._next_idx] = data
            # self._next_idx = (self._next_idx + 1) % self._maxsize
        # Baseline reference # Baseline reference # Baseline reference


        experience = (state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo)
        # if self.num_experiences < self.buffer_size:
        if self.pos >= len(self._storage): # If there's space to save
            self._storage.append(experience)
            # self.num_experiences += 1
        else: # if full and index is circulated to the first
            while not rospy.is_shutdown():
                if not self._storage[self.pos][9] == DEMO: # IF NOT DEMO, DELETE
                    self._storage[self.pos] = experience # If full, replace from the oldest experince
                    break
                else: 
                    self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
                    # iterate until no demo data appears
        # self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size # circulater type lists

        self._it_sum[self.pos] = self._max_priority ** self._alpha
        self._it_min[self.pos] = self._max_priority ** self._alpha





    ''' 
        update new priorities for the processed batch
        it should be done after calulating the losses for the batch
    '''
    def update_priorities(self, batch_indices, n_step_priorities, one_step_priorities, isDemos):


        max_n = n_step_priorities.max()
        max_one = one_step_priorities.max()

        shp =  one_step_priorities.shape

        max_coef = max(max_n, max_one)
        # max_coef = max_batch

        max_coef_arr = max_coef*np.ones(shp)

        # for idx, prio1, prio2, mca, isDemo in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr, isDemos):
        #        # self.priorities[idx] = prio2 + 1e-6*np.ones(mca.shape) + 1e-3*mca
        #     print isDemo
        #     print isDemo.shape
        #     self.priorities[idx] = prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca

        # Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(batch_indices) == len(one_step_priorities) and len(batch_indices) == len(n_step_priorities)
        # for idx, priority in zip(idxes, priorities):
        for idx, prio1, prio2, mca, isDemo in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr, isDemos):

            assert prio1 > 0 and prio2 > 0 
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca) ** self._alpha
            self._it_min[idx] = (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca) ** self._alpha
            self._max_priority = max(self._max_priority, (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca))

## W/O HER ## W/O HER ## W/O HER ## W/O HER ## W/O HER ## W/O HER ## W/O HER ## W/O HER



class ReplayBuffer_TD3(object):

    '''
    Implementation of prioritized replay buffer! applied HER, so we have place for goal in here
    '''


    def __init__(self, buffer_size, total_train_steps = 500000,prob_alpha=0.8): # focus on prob_alpha
        self.buffer_size = buffer_size
        self.total_train_steps = total_train_steps
        # self.buffer = deque()
        self.buffer = []
        assert prob_alpha >= 0
        self._alpha = prob_alpha
        self.pos = 0
        # self.priorities = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.beta0 = BETA_START
        # adoptation from OpenAI implementation # OpenAI # OpenAI # OpenAI

        self._storage = []
        # self._maxsize = size
        self._next_idx = 0

        self.beta_schedule = LinearSchedule(self.total_train_steps,
                                    initial_p=self.beta0,
                                    final_p=1.0)


        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0





        # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI

    # Reference # Reference # Reference # Reference # Reference # Reference
    # def sample(self, batch_size, beta=0.4):
    #     if len(self.buffer) == self.capacity:
    #         prios = self.priorities
    #     else:
    #         prios = self.priorities[:self.pos]
    #     probs = prios ** self.prob_alpha

    #     probs /= probs.sum()
        # indices = np.random.choice(len(self.buffer), batch_size, p=probs) # get the array of incices of item in buffer, w.r.t. probs
        # samples = [self.buffer[idx] for idx in indices] # using those probs., we sample from buffer to obtain a batch of sampels
    #     ##################################################################

    #     weights /= weights.max()
    #     return samples, indices, np.array(weights, dtype=np.float32)
    # Reference # Reference # Reference # Reference # Reference # Reference

    def _sample_proportional(self, batch_size):
        _res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            _res.append(idx)
            res = np.asarray(_res)
        return res


    def getBatch(self, batch_size, beta=0.7):
        # Randomly sample batch_size examples

        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        # idxes ->  np_array

        weights = []
        # print '=================Sampled batch indices==================='
        # print idxes
        # print '========================================================='


        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)

        # encoded_sample = self._encode_sample(idxes)

            # return tuple(list(encoded_sample) + [weights, idxes])
            # self.beta = np.min([1, self.beta + BETE_INCR_PER_SMPL])  # anneals the bia by beta increment
            # if len(self.buffer) == self.buffer_size:
            #     prios = self.priorities
            # else:
            #     prios = self.priorities[:self.pos]

            # # implementation of P(i)
            # probs = prios ** self.prob_alpha
            # probs /= probs.sum()

            # indices = np.random.choice(len(self.buffer), batch_size, p=probs) # get the array of incices of item in buffer, w.r.t. probs
        samples = [self._storage[idx] for idx in idxes] # using those probs., we sample from buffer to obtain a batch of sampels
            # total = len(self.buffer)
            # weights = (total * probs[indices]) ** (-self.beta)

        return samples, idxes, np.array(weights, dtype=np.float32) # returns a list()
        

    def getActionBatch(self, batch_size):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        
        action_samples = [self._storage[idx] for idx in idxes]


        return action_samples


    #                0          1       2         3       4       5       6      7                            
    def add(self, state, next_state, action, n_reward, reward, state_n, done, isDemo): # same as poluate()
        # max_prio = self.priorities.max() if self._storage else 1.0
        # My implementation seems to be wiped out...


        # Baseline reference # Baseline reference # Baseline reference
            # data = (obs_t, action, reward, obs_tp1, done)

            # if self._next_idx >= len(self._storage):
            #     self._storage.append(data)
            # else:
            #     self._storage[self._next_idx] = data
            # self._next_idx = (self._next_idx + 1) % self._maxsize
        # Baseline reference # Baseline reference # Baseline reference

                    #   0         1          2       3       4       5       6       7
        experience = (state, next_state, action, n_reward, reward, state_n, done, isDemo)
        # if self.num_experiences < self.buffer_size:
        if self.pos >= len(self._storage): # If there's space to save
            self._storage.append(experience)
            # self.num_experiences += 1
        else: # if full and index is circulated to the first
            while not rospy.is_shutdown():
                if not self._storage[self.pos][7] == DEMO: # IF NOT DEMO, DELETE
                    self._storage[self.pos] = experience # If full, replace from the oldest experince
                    break
                else: 
                    self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
                    # iterate until no demo data appears
        # self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size # circulater type lists

        self._it_sum[self.pos] = self._max_priority ** self._alpha
        self._it_min[self.pos] = self._max_priority ** self._alpha





    ''' 
        update new priorities for the processed batch
        it should be done after calulating the losses for the batch
    '''
    def update_priorities(self, batch_indices, n_step_priorities, one_step_priorities, isDemos):


        max_n = n_step_priorities.max()
        max_one = one_step_priorities.max()

        shp =  one_step_priorities.shape

        max_coef = max(max_n, max_one)
        # max_coef = max_batch

        max_coef_arr = max_coef*np.ones(shp)

        # for idx, prio1, prio2, mca, isDemo in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr, isDemos):
        #        # self.priorities[idx] = prio2 + 1e-6*np.ones(mca.shape) + 1e-3*mca
        #     print isDemo
        #     print isDemo.shape
        #     self.priorities[idx] = prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca

        # Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref# Ref
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(batch_indices) == len(one_step_priorities) and len(batch_indices) == len(n_step_priorities)
        # for idx, priority in zip(idxes, priorities):
        for idx, prio1, prio2, mca, isDemo in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr, isDemos):

            assert prio1 > 0 and prio2 > 0 
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca) ** self._alpha
            self._it_min[idx] = (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca) ** self._alpha
            self._max_priority = max(self._max_priority, (prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca))
