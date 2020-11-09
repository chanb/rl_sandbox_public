import numpy as np
import torch

import rl_sandbox.constants as c

from rl_sandbox.buffers.buffer import NoSampleError
from rl_sandbox.buffers.wrappers.buffer_wrapper import BufferWrapper
from rl_sandbox.buffers.utils import SumTree


class PrioritizedExperienceReplay(BufferWrapper):
    def __init__(self,
                 buffer,
                 per_alpha,
                 per_beta,
                 per_epsilon,
                 per_beta_increment=0):
        super().__init__(buffer)
        self._priority_tree = SumTree(capacity=self.buffer.memory_size,
                                      rng=self.buffer.rng)
        self._alpha = per_alpha
        self._beta = per_beta
        self._epsilon = per_epsilon
        self._beta_increment = per_beta_increment
        self._maximal_priority = 1.

    def update_priorities(self, idxes, priorities):
        new_priorities = np.abs(priorities) + self._epsilon
        self._maximal_priority = np.max((self._maximal_priority, new_priorities.max()))
        self._priority_tree.update(idxes, new_priorities ** self._alpha)

    def push(self, obs, h_state, act, rew, done, info, priority=None, **kwargs):
        if priority is None:
            priority = self._maximal_priority

        self._priority_tree.add(priority)
        super().push(obs, h_state, act, rew, done, info, **kwargs)

    def sample(self, batch_size):
        if not len(self):
            raise NoSampleError

        self._beta = min(1, self._beta + self._beta_increment)

        tree_idxes, priorities = self._priority_tree.sample(batch_size)
        idxes = tree_idxes - self._priority_tree.shift

        sample_probs = priorities / self._priority_tree.total_value
        importance_sampling_weights = np.power(len(self) * sample_probs, -self._beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        obss, h_states, acts, rews, dones, infos, lengths = self.buffer.get_transitions(idxes)
        obss = obss[:, None, ...]

        infos[c.PRIORITY] = priorities
        infos[c.IS_WEIGHT] = importance_sampling_weights
        infos[c.SAMPLE_PROB] = sample_probs
        infos[c.TREE_IDX] = tree_idxes

        return obss, h_states, acts, rews, dones, infos, lengths, idxes

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state):
        obss, h_states, acts, rews, dones, infos, lengths, random_idxes = self.sample(batch_size)

        next_idxes = random_idxes + 1
        next_obss, next_h_states = self.buffer.get_next(next_idxes, next_obs, next_h_state)
        obss = obss[:, None, ...]

        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, random_idxes

    def sample_consecutive(self, batch_size, end_with_done=False):
        raise NotImplementedError
