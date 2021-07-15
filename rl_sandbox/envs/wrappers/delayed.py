import numpy as np

from collections import deque

import rl_sandbox.constants as c


class ConstantDelayWrapper:
    def __init__(self, env, observation_delay, action_delay, reward_delay, default_action, default_reward):
        assert observation_delay >= 0
        assert action_delay >= 0
        assert reward_delay >= 0

        self._env = env

        self._observation_delay = observation_delay
        self._action_delay = action_delay
        self._reward_delay = reward_delay

        # This is the set of applied action to the environment before first observation arrives
        self._default_action = default_action
        self._default_reward = default_reward

        self.observation_buffer = deque([], maxlen=self._observation_delay)
        self.action_buffer = deque([], maxlen=self._action_delay)
        self.reward_buffer = deque([], maxlen=self._reward_delay)

    def reset(self):
        done = True
        for _ in range(self._action_delay):
            self.action_buffer.append(self._default_action)

        while done:
            for _ in range(self._reward_delay):
                self.reward_buffer.append(self._default_reward)

            obs = self._env.reset()
            self.observation_buffer.append(obs)
            for _ in range(self._observation_delay - 1):
                obs, reward, done, _ = self._env.step(self.action_buffer[0])
                self.observation_buffer.append(obs)
                self.reward_buffer.append(reward)
                if done:
                    self.observation_buffer.clear()
                    self.reward_buffer.clear()
                    break
        return self.observation_buffer[0]

    def step(self, action):
        self.action_buffer.append(action)
        obs, reward, done, info = self._env.step(self.action_buffer[0])
        self.observation_buffer.append(obs)
        self.reward_buffer.append(reward)
        return self.observation_buffer[0], self.reward_buffer[0], done, info

    def render(self, **kwargs):
        self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)
