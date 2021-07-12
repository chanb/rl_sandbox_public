import numpy as np

from copy import deepcopy

from rl_sandbox.constants import ORIGINAL_OBS, WINDOW
from rl_sandbox.envs.wrappers.wrapper import Wrapper


class DictToNumPyWrapper(Wrapper):
    def __init__(self, env, keys, preprocess=dict()):
        assert len(keys)
        super().__init__(env)
        self._obs = None
        self._keys = keys
        self._preprocess = preprocess

    def _get_obs(self):
        obs = []
        for key in self._keys:
            if key in self._preprocess:
                obs.append(self._preprocess[key](self._obs[key]).reshape(-1))
            else:
                obs.append(self._obs[key].reshape(-1))
        return np.concatenate(obs, axis=0)

    def reset(self):
        self._obs = self._env.reset()
        return self._get_obs()

    def step(self, action):
        self._obs, reward, done, info = self._env.step(action)
        info[ORIGINAL_OBS] = deepcopy(self._obs)
        return self._get_obs(), reward, done, info

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        pass


class DMControlDictToNumPyWrapper(DictToNumPyWrapper):
    def __init__(self, env, keys):
        super().__init__(env, keys)

    def reset(self):
        timestep = self._env.reset()
        self._obs = timestep.observation
        return self._get_obs()

    def step(self, action):
        timestep = self._env.step(action)
        self._obs = timestep.observation
        return self._get_obs(), timestep.reward, timestep.last(), {ORIGINAL_OBS: deepcopy(self._obs)}


class GymDictToNumPyWrapper(DictToNumPyWrapper):
    def __init__(self, env, keys):
        super().__init__(env, keys)

    def seed(self, seed):
        self._env.seed(seed)
