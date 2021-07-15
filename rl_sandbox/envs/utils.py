import numpy as np
import os

os.environ["MUJOCO_GL"] = "egl"

import rl_sandbox.constants as c

def make_env(env_config, seed=None):
    assert env_config[c.ENV_TYPE] in c.VALID_ENV_TYPE
    if env_config[c.ENV_TYPE] == c.GYM:
        import gym
        import pybullet_envs
        env = gym.make(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.DM_CONTROL:
        from dm_control import suite
        env = suite.load(**env_config[c.ENV_BASE])
    else:
        raise NotImplementedError

    for wrapper_config in env_config[c.ENV_WRAPPERS]:
        env = wrapper_config[c.WRAPPER](env, **wrapper_config[c.KWARGS])

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    env.seed(seed)

    return env
