import numpy as np
import torch

from torch.distributions import Categorical

import rl_sandbox.constants as c

from rl_sandbox.agents.rl_agents import ACAgent


class HierarchicalRLAgent(ACAgent):
    def __init__(self, high_level_model, low_level_model, learning_algorithm, preprocess=lambda obs: obs):
        self.high_level_model = high_level_model
        self.curr_high_level_obs = None
        self.curr_high_level_act = None
        self.curr_high_level_h_state = None

        super().__init__(model=low_level_model,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)

    def update(self, curr_obs, curr_h_state, action, reward, done, info, next_obs, next_h_state):
        info[c.HIGH_LEVEL_OBSERVATION] = self.curr_high_level_obs
        info[c.HIGH_LEVEL_HIDDEN_STATE] = self.curr_high_level_h_state
        info[c.HIGH_LEVEL_ACTION] = self.curr_high_level_act

        return self.learning_algorithm.update(curr_obs,
                                              curr_h_state,
                                              action,
                                              reward,
                                              done,
                                              info,
                                              next_obs,
                                              next_h_state)


class SACXAgent(HierarchicalRLAgent):
    def __init__(self, scheduler, intentions, learning_algorithm, scheduler_period, preprocess=lambda obs: obs):
        super().__init__(high_level_model=scheduler,
                         low_level_model=intentions,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)
        assert scheduler_period > 0
        self._scheduler_period = scheduler_period

    def compute_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy, self.curr_high_level_mean, self.curr_high_level_variance = \
                    self.high_level_model.compute_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy,
                                   c.MEAN: self.curr_high_level_mean,
                                   c.VARIANCE: self.curr_high_level_variance}

        action, hidden_state, act_info = super().compute_action(obs, hidden_state)

        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]
        act_info[c.MEAN] = act_info[c.MEAN][self.curr_high_level_act]
        act_info[c.VARIANCE] = act_info[c.VARIANCE][self.curr_high_level_act]

        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def deterministic_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy = \
                    self.high_level_model.deterministic_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy}

        action, hidden_state, act_info = super().deterministic_action(obs, hidden_state)
        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]

        # Use the action from the main task
        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def reset(self):
        self._curr_timestep = 0
        self.curr_high_level_obs = None
        self.curr_high_level_h_state = np.nan
        self.curr_high_level_act = None
        self.curr_high_level_value = None
        self.curr_high_level_log_prob = None
        self.curr_high_level_entropy = None
        self.curr_high_level_mean = None
        self.curr_high_level_variance = None
        return super().reset()


class DIAYNAgent(HierarchicalRLAgent):
    """ One may consider sampling skill from prior distribution as a high level action
    """
    def __init__(self, prior, model, learning_algorithm, preprocess=lambda obs: obs):
        super().__init__(high_level_model=prior,
                         low_level_model=model,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)

    def compute_action(self, obs, hidden_state):
        return super().compute_action(obs, hidden_state)

    def reset(self):
        self.curr_high_level_act = self.high_level_model.sample(num_samples=(1,))
        return super().reset()
