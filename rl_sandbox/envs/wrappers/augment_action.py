import numpy as np

import rl_sandbox.constants as c

from rl_sandbox.envs.wrappers.wrapper import Wrapper


class AugmentActionWrapper(Wrapper):
    def __init__(self, env, action_dim):
        assert action_dim > 0
        super().__init__(env)
        self._action_dim = action_dim

    def reset(self):
        # Assumes initial action is a zero vector.
        obs = self._env.reset()
        return np.concatenate((obs, np.zeros(self._action_dim)))

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = np.concatenate((obs, action))

        return obs, reward, done, info

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)
