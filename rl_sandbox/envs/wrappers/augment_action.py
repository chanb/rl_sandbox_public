import numpy as np

import rl_sandbox.constants as c


class AugmentActionWrapper:
    def __init__(self, env, action_dim):
        assert action_dim > 0
        self._env = env
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
