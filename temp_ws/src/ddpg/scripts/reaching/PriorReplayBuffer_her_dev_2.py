#!/usr/bin/env python

import random
import numpy as np

import operator
import rospy
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 10000
BETE_INCR_PER_SMPL = 1e-5  # annealing the bias
ABS_ERR_UPR = 1   # for stability refer to paper



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


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo): # same as poluate()
    # def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo)
        # data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize


    '''
    not necessary for my implementation
    def _encode_sample(self, idxes):

        states, next_states, actions, n_rewards, rewards, state_ns, desired_goal_states, desired_goal_observations, dones, isDemos = [], [], [], [], [], [], [], [], [], []
 
        #obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]    
            state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo = data 
            
            states.append()




            # obs_t, action, reward, obs_tp1, done = data

        '''
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        '''
    
    '''




        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
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
        return self._encode_sample(idxes)



class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    '''

    def add(self, state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo): # same as poluate()
        max_prio = self.priorities.max() if self.buffer else 1.0

        experience = (state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            while True:
                if not self.buffer[self.pos][9] == isDemo: # IF NOT DEMO, DELETE
                    self.buffer[self.pos] = experience # If full, replace from the oldest experince
                    break
                else: 
                    self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
                    # iterate until no demo data appears
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
    '''


    def add(self, state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo): # same as poluate()
    # def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        # super().add(obs_t, action, reward, obs_tp1, done)
        super().add(state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo)
        
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        # encoded_sample = self._encode_sample(idxes)







        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

# Previous code HERE # Previous code HERE # Previous code HERE # Previous code HERE # Previous code HERE
# Previous code HERE # Previous code HERE # Previous code HERE # Previous code HERE # Previous code HERE



# My implementation here # My implementation here # My implementation here # My implementation here # My implementation here

class ReplayBuffer(object):

    '''
    Implementation of prioritized replay buffer! applied HER, so we have place for goal in here
    '''
    def __init__(self, buffer_size, prob_alpha=0.8): # focus on prob_alpha
        self.num_experiences = 0
        self.buffer_size = buffer_size
        self.num_experiences = 0
        # self.buffer = deque()
        self.buffer = []
        self.prob_alpha = prob_alpha
        self.pos = 0
        self.priorities = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.beta = BETA_START

        # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI


    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def __len__(self):
        return len(self.buffer)

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

    def getBatch(self, batch_size, beta=0.7):

        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0


        # Randomly sample batch_size examples
        self.beta = np.min([1, self.beta + BETE_INCR_PER_SMPL])  # anneals the bia by beta increment
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # implementation of P(i)
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs) # get the array of incices of item in buffer, w.r.t. probs
        samples = [self.buffer[idx] for idx in indices] # using those probs., we sample from buffer to obtain a batch of sampels
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        return samples, indices, np.array(weights, dtype=np.float32) # returns a list()

    def size(self):
        return self.buffer_size





    #                0          1       2         3       4       5              6                      7               8      9                            
    def add(self, state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo): # same as poluate()
        
        # following line will be moved to update_priorities
        max_prio = self.priorities.max() if self.buffer else 1.0

        experience = (state, next_state, action, n_reward, reward, state_n, desired_goal_state, desired_goal_observation, done, isDemo)
        
        # CRITICAL ERROR!! Adopt OpenAI's 181026
        # if self.num_experiences < self.buffer_size:
        if self.pos >= len(self.buffer):
            self.buffer.append(experience)
        else:
            while not rospy.is_shutdown():
                if not self.buffer[self.pos][9]: # IF NOT DEMO, DELETE
                    self.buffer[self.pos] = experience # If full, replace from the oldest experince
                    break
                else: # IF DEMO, move to next
                    self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
                    # iterate until no demo data appears
        self.priorities[self.pos] = max_prio
        # self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
        self.pos = (self.pos + 1) % self.buffer_size

        # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI # OpenAI
        pos = self.pos

        self._it_sum[pos] = self._max_priority ** self.prob_alpha
        self._it_min[pos] = self._max_priority ** self.prob_alpha
         # PER # PER # PER # PER # PER # PER # PER # PER # PER



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

        for idx, prio1, prio2, mca, isDemo in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr, isDemos):
               # self.priorities[idx] = prio2 + 1e-6*np.ones(mca.shape) + 1e-3*mca
            print isDemo
            print isDemo.shape
            self.priorities[idx] = prio1 + prio2 + 1e-4*np.ones(mca.shape)*isDemo + 1e-6*mca
