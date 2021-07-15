import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.train.train_ppo import train_ppo
from rl_sandbox.model_architectures.actor_critics.fully_connected_actor_critic import FullyConnectedGaussianAC
from rl_sandbox.model_architectures.layers_definition import POLICY_BASED_LINEAR_LAYERS

# This is for script run
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

seed = args.seed

obs_dim = 15
action_dim = 3
device = torch.device("cuda:0")

action_repeat = 1
num_frames = 1

max_total_steps = 1000000
steps_between_update = 2048

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.AsType(),
    c.BUFFER_SETTING: {
        c.KWARGS: {
            c.MEMORY_SIZE: steps_between_update,
            c.OBS_DIM: (obs_dim,),
            c.H_STATE_DIM: (1,),
            c.ACTION_DIM: (action_dim,),
            c.REWARD_DIM: (1,),
            c.INFOS: {c.MEAN: ((action_dim,), np.float32),
                      c.VARIANCE: ((action_dim,), np.float32),
                      c.ENTROPY: ((action_dim,), np.float32),
                      c.LOG_PROB: ((1,), np.float32),
                      c.VALUE: ((1,), np.float32),
                      c.DISCOUNTING: ((1,), np.float32)},
        },
        c.STORAGE_TYPE: c.RAM,
        c.BUFFER_WRAPPERS: [
            {
                c.WRAPPER: TorchBuffer,
                c.KWARGS: {},
            },
        ],
    },

    # Environment
    c.CLIP_ACTION: True,
    c.ENV_SETTING: {
        c.ENV_BASE: {
            c.ENV_NAME: "HopperBulletEnv-v0"
        },
        c.ENV_TYPE: c.GYM,
        c.ENV_WRAPPERS: [
            {
                c.WRAPPER: ActionRepeatWrapper,
                c.KWARGS: {
                    c.ACTION_REPEAT: action_repeat,
                    c.DISCOUNT_FACTOR: 1.,
                    c.ENABLE_DISCOUNTING: False,
                }
            },
            {
                c.WRAPPER: FrameStackWrapper,
                c.KWARGS: {
                    c.NUM_FRAMES: num_frames,
                }
            }
        ]
    },
    c.MIN_ACTION: -np.ones(action_dim),
    c.MAX_ACTION: np.ones(action_dim),

    # Evaluation
    c.EVALUATION_FREQUENCY: 5000,
    c.EVALUATION_RENDER: False,
    c.EVALUATION_RETURNS: [],
    c.NUM_EVALUATION_EPISODES: 5,
    
    # General
    c.DEVICE: device,
    c.SEED: seed,

    # Load
    c.LOAD_MODEL: False,

    # Logging
    c.PRINT_INTERVAL: 5000,
    c.SAVE_INTERVAL: 1000000,

    # Model
    c.MODEL_SETTING: {
        c.MODEL_ARCHITECTURE: FullyConnectedGaussianAC,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: POLICY_BASED_LINEAR_LAYERS(in_dim=obs_dim),
            c.DEVICE: device,
            c.NORMALIZE_OBS: True,
            c.NORMALIZE_VALUE: True,
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.POLICY: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
    },

    # PPO
    c.ACCUM_NUM_GRAD: 1,
    c.STEPS_BETWEEN_UPDATE: steps_between_update,
    c.CLIP_PARAM: 0.2,
    c.CLIP_VALUE: True,
    c.ENT_COEF: 0,
    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.GAE_LAMBDA: 0.95,
    c.GAMMA: 0.99,
    c.MAX_GRAD_NORM: 1e10,
    c.NORMALIZE_ADVANTAGE: True,
    c.OPT_BATCH_SIZE: 256,
    c.OPT_EPOCHS: 5,
    c.PG_COEF: 1,
    c.TRAIN_PREPROCESSING: gt.Identity(),
    c.V_COEF: 1,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [],

    # Save
    c.SAVE_PATH: f"../results/pybullet/hopper/gt-ppo/{seed}",

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_ppo(experiment_config=experiment_setting)
