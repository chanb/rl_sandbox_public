"""
This script loads up a trained model and performs evaluation using the model.
During evaluation, we use deterministic action (usually the mean of the action).

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python create_expert_data.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --num_episodes=5
"""

import _pickle as pickle
import argparse
import os
import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.model_architectures.utils import make_model
from rl_sandbox.utils import set_seed


def create_trajectories(args):
    assert args.num_episodes > 0
    assert os.path.isfile(args.model_path)
    assert os.path.isfile(args.config_path)

    set_seed(args.seed)
    with open(args.config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    env = make_env(env_setting, seed=args.seed)

    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]
    if config[c.ALGO] == c.SACX:
        intentions = make_model(config[c.INTENTIONS_SETTING])
        intentions.load_state_dict(torch.load(args.model_path)[c.INTENTIONS][c.STATE_DICT])

        agent = SACXAgent(scheduler=None,
                          intentions=intentions,
                          learning_algorithm=None,
                          scheduler_period=config[c.MAX_EPISODE_LENGTH],
                          preprocess=config[c.EVALUATION_PREPROCESSING])
    else:
        model = make_model(config[c.MODEL_SETTING])

        saved_model = torch.load(args.model_path)
        model.load_state_dict(saved_model[c.STATE_DICT])
        if hasattr(model, c.OBS_RMS):
            model.obs_rms = saved_model[c.OBS_RMS]
        
        agent = ACAgent(model=model,
                        learning_algorithm=None,
                        preprocess=config[c.EVALUATION_PREPROCESSING])

    (ret_mean, ret_std) = evaluate_policy(agent=agent,
                                          env=env,
                                          buffer_preprocess=buffer_preprocessing,
                                          num_episodes=args.num_episodes,
                                          clip_action=config[c.CLIP_ACTION],
                                          min_action=config[c.MIN_ACTION],
                                          max_action=config[c.MAX_ACTION],
                                          render=True,)

    print("=" * 100)
    print("Interacted with {} episodes".format(args.num_episodes))
    print("Average Return: {} - Std: {}".format(ret_mean, ret_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--num_episodes", required=True, type=int, help="The maximum number of episodes")
    args = parser.parse_args()

    create_trajectories(args)
