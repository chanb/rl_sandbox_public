import numpy as np

import rl_sandbox.constants as c


class ActionRepeatWrapper:
    def __init__(self, env, action_repeat, discount_factor=1, enable_discounting=False):
        assert action_repeat > 0
        self._env = env
        self._action_repeat = action_repeat
        self._enable_discounting = enable_discounting
        self._discount_factor = discount_factor if enable_discounting else 1.

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        done = False
        cum_reward = 0
        num_repeated = 0
        infos = {
            c.INFOS: []
        }

        while not done and num_repeated < self._action_repeat:
            obs, reward, done, info = self._env.step(action)
            cum_reward += (self._discount_factor ** num_repeated) * reward
            num_repeated += 1
            infos[c.INFOS].append(info)

        infos[c.DISCOUNTING] = np.array([num_repeated if self._enable_discounting else 1])

        return obs, cum_reward, done, infos

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)
