import gym

from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.envs.wrappers.pixel import GymPixelWrapper

env = ActionRepeatWrapper(FrameStackWrapper(GymPixelWrapper(gym.make("Hopper-v2")), num_frames=2), action_repeat=2)
env.seed(1)
env.reset()

for _ in range(10):
    obs, reward, done, info = env.step([1, 1, 1])
    print(obs.shape, obs.dtype, done, info)
