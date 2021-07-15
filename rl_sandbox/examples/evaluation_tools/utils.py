import _pickle as pickle
import os
import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler
from rl_sandbox.envs.utils import make_env
from rl_sandbox.model_architectures.utils import make_model


def load_model(seed, config_path, model_path, device, intention=0):
    assert os.path.isfile(model_path)
    assert os.path.isfile(config_path)

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    env = make_env(env_setting, seed=seed)

    if device is None:
        device = config[c.DEVICE]
    else:
        device = torch.device(device)

    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]
    if config[c.ALGO] in (c.SACX, c.DACX):
        config[c.INTENTIONS_SETTING][c.KWARGS]['device'] = device
        intentions = make_model(config[c.INTENTIONS_SETTING])
        intentions_model = torch.load(model_path, map_location=device.type)[c.INTENTIONS]
        if c.ALGORITHM in intentions_model.keys():
            intentions.load_state_dict(intentions_model[c.ALGORITHM][c.STATE_DICT])
        else:
            intentions.load_state_dict(intentions_model[c.STATE_DICT])

        scheduler = FixedScheduler(intention_i=intention,
                                   num_tasks=config[c.SCHEDULER_SETTING][c.TRAIN][c.KWARGS][c.NUM_TASKS])
        agent = SACXAgent(scheduler=scheduler,
                          intentions=intentions,
                          learning_algorithm=None,
                          scheduler_period=c.MAX_INT,
                          preprocess=config[c.EVALUATION_PREPROCESSING])
    else:
        model = make_model(config[c.MODEL_SETTING])

        saved_model = torch.load(model_path)
        if config[c.ALGO] == c.DAC:
            saved_model = saved_model[c.ALGORITHM]
        model.load_state_dict(saved_model[c.STATE_DICT])
        if hasattr(model, c.OBS_RMS):
            model.obs_rms = saved_model[c.OBS_RMS]
        
        if config[c.ALGO] == c.MULTITASK_BC:
            scheduler = FixedScheduler(intention_i=intention,
                                       num_tasks=config[c.NUM_TASKS])
            agent = SACXAgent(scheduler=scheduler,
                              intentions=model,
                              learning_algorithm=None,
                              scheduler_period=c.MAX_INT,
                              preprocess=config[c.EVALUATION_PREPROCESSING])
        else:
            agent = ACAgent(model=model,
                            learning_algorithm=None,
                            preprocess=config[c.EVALUATION_PREPROCESSING])

    return config, env, buffer_preprocessing, agent
