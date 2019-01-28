import numpy as np

# T = episode_batch['u'].shape[1]
T = 300
# buffer_shapes['ag'] = (self.T, self.dimg)

# rollout_batch_size = episode_batch['u'].shape[0]
rollout_batch_size = 1 # lengh of each episode

replay_k = 4

future_p = 1 - (1. / (1 + replay_k))


# future_p = 0.8
batch_size = 30

#episode_batch is {key: array(buffer_size x T x dim_key)}
 
# Select which episodes and time steps to use.
episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
t_samples = np.random.randint(T, size=batch_size)
# transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
#                    for key in episode_batch.keys()}

# Select future time indexes proportional with probability future_p. These
# will be used for HER replay by substituting in future goals.
her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
future_offset = future_offset.astype(int)
future_t = (t_samples +  future_offset)[her_indexes]


# print episode_idxs
# print len(episode_idxs)
# print '-----------------------------------------'
# print np.random.uniform(size=30)
print '-------------------------------------------------------------------'
print 'HER ratio'
print future_p
print '-------------------------------------------------------------------'
print 'total trajectory'
print T
print '-------------------------------------------------------------------'
print 'batch size'
print batch_size
print '-------------------------------------------------------------------'
print 'samples to be saved in replay buffer'
print t_samples
# print t_samples.size
print '-------------------------------------------------------------------'
# print np.random.uniform(size=batch_size)
print 'sample indices for goal substitution'
print t_samples[her_indexes]
# print her_indexes[0]
print '# of her samples'
print len(her_indexes[0])
# print T-t_samples
print '-------------------------------------------------------------------'

print 'time offset for each sample'
print future_offset
print '-------------------------------------------------------------------'
print 'future goal timestep'
print future_t
# print len(future_t)
print '-------------------------------------------------------------------'


# Replace goal with achieved goal but only for the previously-selected
# # HER transitions (as defined by her_indexes). For the other transitions,
# # keep the original goal.
# future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
# transitions['g'][her_indexes] = future_ag

# # Reconstruct info dictionary for reward  computation.
# info = {}
# for key, value in transitions.items():
#     if key.startswith('info_'):
#         info[key.replace('info_', '')] = value

# # Re-compute reward since we may have substituted the goal.
# reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
# reward_params['info'] = info
# transitions['r'] = reward_fun(**reward_params)

# transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#                for k in transitions.keys()}

# assert(transitions['u'].shape[0] == batch_size_in_transitions)



