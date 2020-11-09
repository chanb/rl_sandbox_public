"""
This script loads up an expert model and generate an expert buffer.

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python create_expert_data.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --save_path=./expert_buffer.pkl \
        --num_episodes=5 --num_steps=10000
"""

import _pickle as pickle
import argparse
import os
import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import buffer_warmup
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.utils import set_seed


def create_trajectories(args):
    assert args.num_episodes > 0
    assert os.path.isfile(args.model_path)
    assert os.path.isfile(args.config_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    set_seed(args.seed)
    with open(args.config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    env_setting[c.ENV_WRAPPERS][0][c.KWARGS][c.CREATE_ABSORBING_STATE] = True
    env_setting[c.ENV_WRAPPERS][0][c.KWARGS][c.MAX_EPISODE_LENGTH] = 1000
    env = make_env(env_setting, seed=args.seed)
    model = make_model(config[c.MODEL_SETTING])
    model.load_state_dict(torch.load(args.model_path)[c.STATE_DICT])
    
    agent = ACAgent(model=model,
                    learning_algorithm=None,
                    preprocess=config[c.EVALUATION_PREPROCESSING])

    config[c.BUFFER_SETTING][c.KWARGS][c.MEMORY_SIZE] = args.num_steps
    config[c.BUFFER_SETTING][c.STORE_NEXT_OBSERVATION] = True
    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]

    expert_buffer = make_buffer(config[c.BUFFER_SETTING], args.seed)

    config[c.NUM_STEPS] = args.num_steps
    config[c.NUM_EPISODES] = args.num_episodes

    def transition_preprocess(obs,
                              h_state,
                              action,
                              reward,
                              done,
                              info,
                              next_obs,
                              next_h_state):
        if obs[:, -1] == 1:
            action[:] = 0

        return {
            "obs": obs,
            "h_state": h_state,
            "act": action,
            "rew": [reward],
            "done": False,
            "info": info,
            "next_obs": next_obs,
            "next_h_state": next_h_state,
        }

    buffer_warmup(agent=agent,
                  env=env,
                  buffer=expert_buffer,
                  buffer_preprocess=buffer_preprocessing,
                  transition_preprocess=transition_preprocess,
                  experiment_settings=config)

    expert_buffer.save(save_path=args.save_path, end_with_done=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the trajectories")
    parser.add_argument("--num_episodes", required=True, type=int, help="The maximum number of episodes")
    parser.add_argument("--num_steps", required=True, type=int, help="The maximum number of steps")
    args = parser.parse_args()

    create_trajectories(args)
