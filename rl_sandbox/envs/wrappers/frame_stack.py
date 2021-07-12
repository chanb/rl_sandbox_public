import numpy as np

from collections import deque

from rl_sandbox.envs.wrappers.wrapper import Wrapper


class FrameStackWrapper(Wrapper):
    def __init__(self, env, num_frames):
        assert num_frames > 0
        super().__init__(env)
        self._num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def _get_obs(self):
        assert len(self.frames) == self._num_frames
        return np.stack(self.frames)

    def reset(self):
        obs = self._env.reset()
        for _ in range(self._num_frames):
            self.frames.append(obs)

        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.frames.append(obs)

        return self._get_obs(), reward, done, info

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)
