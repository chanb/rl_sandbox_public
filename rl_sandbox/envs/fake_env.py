import numpy as np

class FakeEnv:
    def __init__(self, obs_dim):
        self._obs_dim = obs_dim

    def reset(self):
        return np.zeros(self._obs_dim)

    def step(self, action):
        return np.zeros(self._obs_dim), 0., False, {}

    def render(self):
        pass
