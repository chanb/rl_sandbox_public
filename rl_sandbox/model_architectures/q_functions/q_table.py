import torch

from torch.distributions import Categorical

import rl_sandbox.constants as c


class QTable:
    def __init__(self,
                 num_states,
                 num_actions,
                 temperature=1.,
                 temperature_decay=1.,
                 temperature_min=1.,
                 device=torch.device(c.CPU)):
        self._num_states = num_states
        self._num_actions = num_actions
        self._temperature = temperature
        self._temperature_decay = temperature_decay
        self._temperature_min = temperature_min
        self.device = device

    def _initialize_table(self):
        self.table = torch.zeros(size=(self._num_states, self._num_actions),
                                 device=self.device)

    def compute_action(self, x, h):
        dist = Categorical(logits=self.table[x] / self._temperature)
        action = dist.sample()
        lprob = dist.log_prob(action)
        val = torch.sum(self.table[state] * dist.probs)

        self._temperature = max(self._temperature_min, self._temperature * self._temperature_decay)

        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy(), dist.mean.cpu().numpy(), dist.variance.cpu().numpy()

    def deterministic_action(self, x, h):
        dist = Categorical(logits=self.table[x] / self._temperature)
        action = torch.argmax(self.table[x])
        lprob = dist.log_prob(action)
        val = torch.sum(self.table[state] * dist.probs)
        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy()

    def update_qsa(self, state, h, action, q_value):
        self.table[state, action] = q_value

    def qsa(self, state, h, action):
        return self.table[state, action]
