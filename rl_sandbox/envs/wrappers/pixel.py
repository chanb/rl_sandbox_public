import cv2
import numpy as np

from gym.wrappers.pixel_observation import PixelObservationWrapper


class PixelWrapper:
    def __init__(self, env, render_h=84, render_w=84):
        self._env = env
        self.render_h = render_h
        self.render_w = render_w

    def _get_obs(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        pass


class GymPixelWrapper(PixelWrapper):
    def __init__(self, env, render_h=84, render_w=84):
        super().__init__(env, render_h, render_w)
        self._wrapped_env = None
        self._obs = None

    def _get_obs(self):
        img = cv2.resize(self._obs['pixels'], (self.render_h, self.render_w), interpolation = cv2.INTER_AREA)
        img *= 255
        return img.astype(np.uint8)

    def reset(self):
        gt_obs = self._env.reset()
        if self._wrapped_env is None:
            self._wrapped_env = PixelObservationWrapper(self._env, pixels_only=True)

        self._obs = self._wrapped_env.observation(gt_obs)

        return self._get_obs()

    def step(self, action):
        self._obs, reward, done, info = self._wrapped_env.step(action)
        return self._get_obs(), reward, done, info

    def seed(self, seed):
        self._env.seed(seed)


class DMControlPixelWrapper(PixelWrapper):
    def __init__(self, env, render_h=84, render_w=84):
        super().__init__(env, render_h, render_w)
        self._obs = None

    def _get_obs(self):
        return self._env.physics.render(camera_id=0, height=self.render_h, width=self.render_w)

    def reset(self):
        self._env.reset()
        return self._get_obs()

    def step(self, action):
        timestep = self._env.step(action)
        return self._get_obs(), timestep.reward, timestep.last(), {}
