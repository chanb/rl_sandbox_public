import numpy as np

import rl_sandbox.constants as c

class AbsorbingStateWrapper:
    def __init__(self, env, create_absorbing_state, max_episode_length):
        self._env = env
        self._done = False
        self._obs = None
        self._max_episode_length = max_episode_length
        self._create_absorbing_state = create_absorbing_state

    def _get_obs(self):
        if self._done:
            # Return absorbing state which is [0, ..., 0, 1]
            return np.eye(self._obs.size + 1)[-1]

        return np.concatenate((self._obs.reshape(-1), [0]), axis=0)

    def reset(self):
        self._curr_timestep = 0
        self._obs = self._env.reset()
        return self._get_obs()

    def step(self, action):
        self._curr_timestep += 1
        if self._create_absorbing_state and self._done:
            obs = self._get_obs()
            self._done = False
            return obs, 0., True, {c.ABSORBING_STATE: True, c.DONE: False}

        self._obs, reward, done, info = self._env.step(action)
        info[c.ABSORBING_STATE] = False
        info[c.DONE] = done
        if self._create_absorbing_state and self._curr_timestep < self._max_episode_length and done:
            self._done = True
            done = False
        return self._get_obs(), reward, done, info

    def render(self):
        self._env.render()

    def seed(self, seed):
        self._env.seed(seed)
