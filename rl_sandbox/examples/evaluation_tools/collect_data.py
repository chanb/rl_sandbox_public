"""
This script loads up a trained model and collects trajectories using the model.
We choose how much data is generated by the model and random uniform policy.

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python collect_data.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --num_episodes=5 \
        --num_samples=1000 --save_path=./data.pkl
"""

import _pickle as pickle
import argparse
import gzip
import numpy as np
import os
import torch

from pprint import pprint
from tqdm import tqdm

import rl_sandbox.constants as c

from rl_sandbox.examples.evaluation_tools.utils import load_model
from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.utils import set_seed


def collect_data(args):
    set_seed(args.seed)
    assert args.num_episodes > 0
    assert args.num_samples > 0
    assert 0 <= args.mixture_ratio <= 1

    dir_exists = os.path.isdir(args.save_path)
    assert dir_exists or not os.path.exists(args.save_path)

    if not dir_exists:
        os.makedirs(args.save_path, exist_ok=True)

    config, env, buffer_preprocess, agent = load_model(args.seed,
                                                       args.config_path,
                                                       args.model_path,
                                                       args.device,
                                                       args.intention)

    init_observations = []
    observations = []
    actions = []
    rewards = []
    dones = []

    episodes_pbar = tqdm(total=args.num_episodes)
    samples_pbar = tqdm(total=args.num_samples)

    sample_i = 0
    eval_returns = []
    for episode_i in range(args.num_episodes):
        eval_returns.append(0)
        obs = env.reset()

        init_observations.append(obs)

        buffer_preprocess.reset()
        obs = buffer_preprocess(obs)
        h_state = agent.reset()
        done = False

        while not done:
            if hasattr(env, c.RENDER) and args.render:
                env.render()

            if args.deterministic:
                action, h_state, act_info = agent.deterministic_action(
                    obs=obs, hidden_state=h_state)
            else:
                action, h_state, act_info = agent.compute_action(
                    obs=obs, hidden_state=h_state)

            if np.random.uniform() < args.mixture_ratio:
                action = np.random.uniform(config[c.MIN_ACTION], config[c.MAX_ACTION], config[c.ACTION_DIM])

            actions.append(action)

            if config[c.CLIP_ACTION]:
                action = np.clip(action, a_min=config[c.MIN_ACTION], a_max=config[c.MAX_ACTION])

            obs, reward, done, _ = env.step(action)

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            obs = buffer_preprocess(obs)

            eval_returns[-1] += reward
            sample_i += 1
            samples_pbar.update(1)
            if sample_i >= args.num_samples:
                break
        else:
            episodes_pbar.update(1)
            continue
        break

    ret_mean = np.mean(eval_returns)
    ret_std = np.std(eval_returns)
    ret_max = np.max(eval_returns)
    ret_min = np.min(eval_returns)

    print("=" * 100)
    print("Interacted with {} complete episodes ({} timesteps)".format(episode_i, sample_i))
    print("Average Return: {} - Std: {}".format(ret_mean, ret_std))
    print("Max Return: {} - Min Return: {}".format(ret_max, ret_min))

    for (filename, data) in zip(("init_obss", "obss", "acts", "rews", "dones"),
                                (init_observations, observations, actions, rewards, dones)):
        with gzip.open(f"{args.save_path}/{filename}.pkl", "wb") as f:
            pickle.dump(data, f)

    with gzip.open(f"{args.save_path}/metadata.pkl", "wb") as f:
            pickle.dump({
                "returns": eval_returns,
                "min": ret_min,
                "max": ret_max,
                "avg": ret_mean,
                "std": ret_std,
                **args.__dict__,
            }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the environment")

    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--save_path", type=str, required=True, help="The directory to save the trajectories")
    parser.add_argument("--mixture_ratio", required=True, type=float, help="Amount of data sampled using random uniform policy")
    parser.add_argument("--deterministic", action="store_true", help="Whether or not to use deterministic action (the action mean) from the agent")
    parser.add_argument("--num_episodes", required=True, type=int, help="The maximum number of episodes")
    parser.add_argument("--num_samples", required=True, type=int, help="The maximum number of samples")

    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--intention", type=int, default=0, help="The intention to use for SAC-X")
    parser.add_argument("--device", type=str, help="Device to load models on")
    args = parser.parse_args()

    pprint(args)

    collect_data(args)