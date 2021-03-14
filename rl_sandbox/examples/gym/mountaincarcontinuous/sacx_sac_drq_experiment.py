import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.algorithms.sac_x.schedulers import QTableScheduler
from rl_sandbox.auxiliary_rewards.mountain_car_continuous import MountainCarContinuousAuxiliaryReward
from rl_sandbox.train.train_sacx_sac_drq import train_sacx_sac_drq
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import MultiTaskFullyConnectedSquashedGaussianSAC
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS

# This is for script run
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

seed = args.seed

obs_dim = 2
action_dim = 1
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)
device = torch.device("cuda:0")
# device = torch.device(c.CPU)

action_repeat = 1
num_frames = 1

aux_reward = MountainCarContinuousAuxiliaryReward()
num_tasks = 1 + aux_reward.num_auxiliary_rewards

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.AsType(),
    c.BUFFER_SETTING: {
        c.KWARGS: {
            c.MEMORY_SIZE: 50000 // action_repeat,
            c.OBS_DIM: (obs_dim,),
            c.H_STATE_DIM: (1,),
            c.ACTION_DIM: (action_dim,),
            c.REWARD_DIM: (num_tasks,),
            c.INFOS: {c.MEAN: ((action_dim,), np.float32),
                      c.VARIANCE: ((action_dim,), np.float32),
                      c.ENTROPY: ((action_dim,), np.float32),
                      c.LOG_PROB: ((1,), np.float32),
                      c.VALUE: ((1,), np.float32),
                      c.DISCOUNTING: ((1,), np.float32)},
            c.CHECKPOINT_INTERVAL: 0,
            c.CHECKPOINT_PATH: None,
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
    c.ACTION_DIM: action_dim,
    c.CLIP_ACTION: True,
    c.ENV_SETTING: {
        c.ENV_BASE: {
            c.ENV_NAME: "MountainCarContinuous-v0"
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
    c.MIN_ACTION: min_action,
    c.MAX_ACTION: max_action,
    c.MAX_EPISODE_LENGTH: 1000,
    c.OBS_DIM: obs_dim,

    # Evaluation
    c.EVALUATION_FREQUENCY: 5000,
    c.EVALUATION_RENDER: False,
    c.EVALUATION_RETURNS: [],
    c.NUM_EVALUATION_EPISODES: 5,

    # Exploration
    c.EXPLORATION_STEPS: 1000,
    c.EXPLORATION_STRATEGY: UniformContinuousAgent(min_action,
                                                   max_action,
                                                   np.random.RandomState(seed)),
    
    # General
    c.DEVICE: device,
    c.SEED: seed,

    # Load
    c.LOAD_MODEL: False,

    # Logging
    c.PRINT_INTERVAL: 5000,
    c.SAVE_INTERVAL: 51200,

    # Model
    c.INTENTIONS_SETTING: {
        c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianSAC,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.TASK_DIM: num_tasks,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: VALUE_BASED_LINEAR_LAYERS(in_dim=obs_dim),
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.INTENTIONS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-3,
            },
        },
        c.QS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-3,
            },
        },
        c.ALPHA: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-3,
            },
        },
    },

    c.SCHEDULER_SETTING: {
        c.MODEL_ARCHITECTURE: QTableScheduler,
        c.KWARGS: {
            c.MAX_SCHEDULE: 4,
            c.NUM_TASKS: num_tasks,
            c.TEMPERATURE: 10.,
            c.TEMPERATURE_DECAY: 0.9999,
            c.TEMPERATURE_MIN: 1.,
            c.DEVICE: device,
        },
    },

    # DrQ
    c.K: 2,
    c.M: 2,

    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.TRAIN_PREPROCESSING: gt.Identity(),

    # SAC
    c.ACCUM_NUM_GRAD: 1,
    c.BATCH_SIZE: 256,
    c.BUFFER_WARMUP: 1000,
    c.GAMMA: 0.99,
    c.INITIAL_ALPHA: 1.,
    c.LEARN_ALPHA: True,
    c.MAX_GRAD_NORM: 1e10,
    c.NUM_GRADIENT_UPDATES: 1,
    c.NUM_PREFETCH: 1,
    c.REWARD_SCALING: 1.,
    c.STEPS_BETWEEN_UPDATE: 1,
    c.TARGET_ENTROPY: -1.,
    c.TARGET_UPDATE_INTERVAL: 1,
    c.TAU: 0.005,
    c.UPDATE_NUM: 0,

    # SACX
    c.AUXILIARY_REWARDS: aux_reward,
    c.NUM_TASKS: num_tasks,
    c.SCHEDULER_PERIOD: 250,
    c.SCHEDULER_TAU: 0.4,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [],

    # Save
    c.SAVE_PATH: f"../results/mountaincarcontinuous/gt/sac-x/sac_drq/{seed}",

    # train parameters
    c.MAX_TOTAL_STEPS: 50000 // action_repeat,
    c.TRAIN_RENDER: False,
}

train_sacx_sac_drq(experiment_config=experiment_setting)
