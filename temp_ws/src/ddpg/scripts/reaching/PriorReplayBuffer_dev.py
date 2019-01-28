#!/usr/bin/env python
from collections import deque
import random
import numpy as np

PRIO_REPLAY_ALPHA = 0.8
BETA_START = 0.7
BETA_FRAMES = 10000

class ReplayBuffer(object):

    '''
    Implementation of prioritized replay buffer!
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
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
    #     weights /= weights.max()
    #     return samples, indices, np.array(weights, dtype=np.float32)

    def getBatch(self, batch_size, beta=0.7):
        # Randomly sample batch_size examples
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
        weights = (total * probs[indices]) ** (-beta)

        return samples, indices, np.array(weights, dtype=np.float32) # returns a list()

    def size(self):
        return self.buffer_size

            
    def add(self, state, next_state, action, n_reward, reward, state_n, done, isDemo): # same as poluate()
        max_prio = self.priorities.max() if self.buffer else 1.0

        experience = (state, next_state, action, n_reward, reward, state_n, done, isDemo)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            while True:
                if not self.buffer[self.pos][5] == isDemo: # IF NOT DEMO, DELETE
                    self.buffer[self.pos] = experience # If full, replace from the oldest experince
                    break
                else: 
                    self.pos = (self.pos + 1) % self.buffer_size # circulater type lists
                    # iterate until no demo data appears
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size # circulater type lists

    ''' 
        update new priorities for the processed batch
        it should be done after calulating the losses for the batch
    '''
    def update_priorities(self, batch_indices, n_step_priorities, one_step_priorities):

        # p_i = loss_i_n_step  + epsilon (1e-6)

        max_n = n_step_priorities.max()
        max_one = one_step_priorities.max()

        shp =  one_step_priorities.shape

        max_coef = max(max_n, max_one)
        # max_coef = max_batch

        max_coef_arr = max_coef*np.ones(shp)

        for idx, prio1, prio2, mca in zip(batch_indices, n_step_priorities, one_step_priorities, max_coef_arr):
               # self.priorities[idx] = prio2 + 1e-6*np.ones(mca.shape) + 1e-3*mca
            self.priorities[idx] = prio1 + prio2 + 1e-6*np.ones(mca.shape) + 1e-1*mca
