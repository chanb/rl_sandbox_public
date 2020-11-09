import numpy as np
import torch

from torch.distributions import Categorical, Normal

import rl_sandbox.constants as c


class RLAgent():
    def __init__(self, model, learning_algorithm):
        self.model = model
        self.learning_algorithm = learning_algorithm

    def update(self, curr_obs, curr_h_state, action, reward, done, info, next_obs, next_h_state):
        return self.learning_algorithm.update(curr_obs,
                                              curr_h_state,
                                              action,
                                              reward,
                                              done,
                                              info,
                                              next_obs,
                                              next_h_state)

    def compute_action(self, obs, **kwargs):
        raise NotImplementedError

    def reset(self):
        # Returns initial hidden state
        if hasattr(self.model, c.INITIALIZE_HIDDEN_STATE):
            return self.model.initialize_hidden_state().numpy().astype(np.float32)
        return np.array([np.nan], dtype=np.float32)


class ACAgent(RLAgent):
    def __init__(self, model, learning_algorithm, preprocess=lambda obs: obs):
        super().__init__(model=model,
                         learning_algorithm=learning_algorithm)
        self.preprocess = preprocess

    def preprocess(self, obs):
        return obs

    def compute_action(self, obs, hidden_state):
        obs = torch.tensor(obs).unsqueeze(0)
        obs = self.preprocess(obs)
        hidden_state = torch.tensor(hidden_state).unsqueeze(0)
        action, value, hidden_state, log_prob, entropy, mean, variance = self.model.compute_action(
            obs, hidden_state)
        act_info = {c.VALUE: value,
                    c.LOG_PROB: log_prob,
                    c.ENTROPY: entropy,
                    c.MEAN: mean,
                    c.VARIANCE: variance}
        return action, hidden_state, act_info

    def deterministic_action(self, obs, hidden_state):
        obs = torch.tensor(obs).unsqueeze(0)
        obs = self.preprocess(obs)
        hidden_state = torch.tensor(hidden_state).unsqueeze(0)
        action, value, hidden_state, log_prob, entropy = self.model.deterministic_action(
            obs, hidden_state)
        act_info = {c.VALUE: value,
                    c.LOG_PROB: log_prob,
                    c.ENTROPY: entropy}
        return action, hidden_state, act_info
