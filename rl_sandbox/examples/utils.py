import _pickle as pickle
import os
import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler
from rl_sandbox.envs.utils import make_env
from rl_sandbox.model_architectures.utils import make_model


def load_model(seed, config_path, model_path, intention=0):
    assert os.path.isfile(model_path)
    assert os.path.isfile(config_path)

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    env = make_env(env_setting, seed=seed)

    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]
    if config[c.ALGO] == c.SACX:
        intentions = make_model(config[c.INTENTIONS_SETTING])
        intentions.load_state_dict(torch.load(model_path)[c.INTENTIONS][c.STATE_DICT])

        scheduler = FixedScheduler(intention_i=intention)
        agent = SACXAgent(scheduler=scheduler,
                          intentions=intentions,
                          learning_algorithm=None,
                          scheduler_period=config[c.MAX_EPISODE_LENGTH],
                          preprocess=config[c.EVALUATION_PREPROCESSING])
        agent.use_intention(intention)
    else:
        model = make_model(config[c.MODEL_SETTING])

        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model[c.STATE_DICT])
        if hasattr(model, c.OBS_RMS):
            model.obs_rms = saved_model[c.OBS_RMS]
        
        agent = ACAgent(model=model,
                        learning_algorithm=None,
                        preprocess=config[c.EVALUATION_PREPROCESSING])

    return config, env, buffer_preprocessing, agent
