#!/usr/bin/env python
from collections import deque
import random
import numpy as np

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

class ReplayBuffer(object):

    '''
    Implementation of prioritized replay buffer!
    '''
    def __init__(self, buffer_size, prob_alpha=0.3): # focus on prob_alpha
        self.exp_source_inter = iter()
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

    def getBatch(self, batch_size, beta=1.0):
        # Randomly sample batch_size examples
        if len(self.buffer) == self.capacity:
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

            
    def add(self, state, action, reward, new_state, done): # same as poluate()
        max_prio = self.priorities.max() if self.buffer else 1.0

        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer[self.pos] = experience # If full, replace from the oldest experince
            # self.buffer.append(experience)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size # circulater type lists


    ''' 
        update new priorities for the processed batch
        it should be done after calulating the losses for the batch
    '''
    def update_priorities(self, batch_indices, batch_priorities): # batch priorities ->from 

        # p_i = loss_i_n_step  + epsilon (1e-6)
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

