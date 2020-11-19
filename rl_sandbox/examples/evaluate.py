"""
This script loads up a trained model and performs evaluation using the model.
During evaluation, we use deterministic action (usually the mean of the action).

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python evaluate.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --num_episodes=5
"""

import _pickle as pickle
import argparse
import torch

import rl_sandbox.constants as c

from rl_sandbox.examples.utils import load_model
from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.utils import set_seed


def evaluate(args):
    set_seed(args.seed)
    assert args.num_episodes > 0

    config, env, buffer_preprocessing, agent = load_model(args.seed,
                                                          args.config_path,
                                                          args.model_path,
                                                          args.intention)

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
    parser.add_argument("--intention", type=int, default=0, help="The intention to use for SAC-X")
    args = parser.parse_args()

    evaluate(args)
