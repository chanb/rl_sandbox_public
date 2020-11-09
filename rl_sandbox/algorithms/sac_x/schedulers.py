import copy
import numpy as np
import torch

from collections import OrderedDict
from torch.distributions import Categorical

import rl_sandbox.constants as c


class QTableScheduler:
    def __init__(self,
                 max_schedule,
                 num_tasks,
                 temperature=1.,
                 temperature_decay=1.,
                 temperature_min=1.,
                 device=torch.device(c.CPU)):
        self.device = device

        self._temperature = temperature
        self._temperature_decay = temperature_decay
        self._temperature_min = temperature_min

        self._max_schedule = max_schedule
        self._num_tasks = num_tasks

        self.table = OrderedDict()
        self._initialize_qtable()

    @property
    def max_obs_len(self):
        return self._max_schedule - 1

    def state_dict(self):
        return {c.Q_TABLE: torch.stack([v for v in self.table.values()])}

    def load_state_dict(self, state_dict):
        for idx, key in enumerate(self.table.keys()):
            self.table[key].data = state_dict[c.Q_TABLE][idx].data

    def _initialize_qtable(self, state=None):
        if state is None:
            state = [-1] * self.max_obs_len
            self.table[self.check_state(state)] = torch.zeros(self._num_tasks)

        try:
            curr_idx = state.index(-1)
        except ValueError:
            return
        
        for ii in range(self._num_tasks):
            state = copy.deepcopy(state)
            state[curr_idx] = ii
            self.table[self.check_state(state)] = torch.zeros(self._num_tasks)
            self._initialize_qtable(state=state)

    def check_state(self, state):
        state = list(copy.deepcopy(state))
        for _ in range(len(state), self.max_obs_len):
            state.append(-1)
        return tuple(state)

    def compute_action(self, state, h):
        state = self.check_state(state)
        dist = Categorical(logits=self.table[state] / self._temperature)
        action = dist.sample()
        lprob = dist.log_prob(action)
        value = torch.sum(self.table[state] * dist.probs)

        self._temperature = max(self._temperature_min, self._temperature * self._temperature_decay)

        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy(), dist.mean.cpu().numpy(), dist.variance.cpu().numpy()

    def deterministic_action(self, state, h):
        state = self.check_state(state)
        dist = Categorical(logits=self.table[state] / self._temperature)
        action = torch.argmax(self.table[state])
        lprob = dist.log_prob(action)
        value = torch.sum(self.table[state] * dist.probs)
        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy()

    def update_qsa(self, state, action, q_value):
        state = self.check_state(state)
        self.table[state][action] = q_value

    def compute_qsa(self, state, action):
        state = self.check_state(state)
        return self.table[state][action]

    def compute_qs(self, state):
        state = self.check_state(state)
        return self.table[state]


class FixedScheduler:
    def __init__(self,
                 intention_i):
        self._intention_i = intention_i

    def compute_action(self, state, h):
        return self._intention_i, None, [np.nan], None, None, None, None


class UScheduler:
    def __init__(self,
                 num_tasks):
        self._num_tasks = num_tasks

    def compute_action(self, state, h):
        action = np.random.randint(0, self._num_tasks, size=(1,))
        return action, None, [np.nan], None, None, None, None
