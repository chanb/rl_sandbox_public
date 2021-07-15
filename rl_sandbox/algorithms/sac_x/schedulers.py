import copy
import numpy as np
import torch

from collections import OrderedDict
from torch.distributions import Categorical

import rl_sandbox.constants as c


class Scheduler:
    def __init__(self, max_schedule, num_tasks):
        self._max_schedule = max_schedule
        self._num_tasks = num_tasks

    @property
    def max_obs_len(self):
        return self._max_schedule - 1

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def compute_action(self, state, h):
        raise NotImplementedError

    def deterministic_action(self, state, h):
        raise NotImplementedError

    def update_qsa(self, state, action, q_value):
        pass

    def compute_qsa(self, state, action):
        return 0.

    def compute_qs(self, state):
        return torch.zeros(self._num_tasks)


class QTableScheduler(Scheduler):
    def __init__(self,
                 max_schedule,
                 num_tasks,
                 temperature=1.,
                 temperature_decay=1.,
                 temperature_min=1.):
        super().__init__(max_schedule, num_tasks)

        self._temperature = temperature
        self._temperature_decay = temperature_decay
        self._temperature_min = temperature_min

        self.table = OrderedDict()
        self._initialize_qtable()

    def state_dict(self):
        state_dict = {
            c.Q_TABLE: self.table,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.table = state_dict[c.Q_TABLE]

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


# TODO: Add Q-table if we want
class FixedScheduler(Scheduler):
    def __init__(self,
                 intention_i,
                 num_tasks,
                 max_schedule=0):
        super().__init__(max_schedule, num_tasks)
        assert intention_i < num_tasks
        self._intention_i = np.array(intention_i, dtype=np.int)
        self.zero = np.zeros((1, 1))

    def compute_action(self, state, h):
        return self._intention_i, np.zeros(1), h.cpu().numpy(), self.zero, self.zero, None, None

    def deterministic_action(self, state, h):
        action, value, h, lprob, entropy, _, _ = self.compute_action(state, h)
        return action, value, h, lprob, entropy


class RecycleScheduler(Scheduler):
    def __init__(self,
                 num_tasks,
                 scheduling,
                 max_schedule=0):
        super().__init__(max_schedule, num_tasks)
        self.zero = np.zeros((1, 1))
        assert np.all(np.asarray(scheduling) >= 1)
        assert num_tasks == len(scheduling)
        self.count = 0
        self.scheduling = np.cumsum(scheduling)

    def state_dict(self):
        return {
            c.COUNT: self.count,
            c.SCHEDULING: self.scheduling,
        }

    def load_state_dict(self, state_dict):
        self.count = state_dict[c.COUNT]
        self.scheduling = state_dict[c.SCHEDULING]

    def compute_action(self, state, h):
        intention = np.where(self.count < self.scheduling)[0][0]
        self.count = (self.count + 1) % self.scheduling[-1]
        return np.array(intention, dtype=np.int), np.zeros(1), h.cpu().numpy(), self.zero, self.zero, None, None

    def deterministic_action(self, state, h):
        action, value, h, lprob, entropy, _, _ = self.compute_action(state, h)
        return action, value, h, lprob, entropy


class UScheduler(Scheduler):
    def __init__(self,
                 num_tasks,
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        if task_options is not None:
            num_tasks = len(task_options)
        super().__init__(max_schedule, num_tasks)
        self._intention_i = np.array(intention_i, dtype=np.int)
        self.zero = np.zeros((1, 1))
        self.lprob = np.log(1 / num_tasks)
        self.entropy = np.array([-num_tasks * (1 / num_tasks) * self.lprob])
        if task_options is None:
            self.task_options = list(range(self._num_tasks))
        else:
            self.task_options = task_options

    def compute_action(self, state, h):
        action = np.array(np.random.choice(self.task_options))
        return action, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None

    def deterministic_action(self, state, h):
        return self._intention_i, np.zeros(1), h.cpu().numpy(), self.zero, self.entropy
