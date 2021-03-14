import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.train.train_sac_diayn import train_sac_diayn
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import FullyConnectedSeparate, FullyConnectedSquashedGaussianSAC
from rl_sandbox.model_architectures.variational_distributions.fully_connected_gaussian import FullyConnectedGaussian
from rl_sandbox.model_architectures.layers_definition import POLICY_BASED_LINEAR_LAYERS, VALUE_BASED_LINEAR_LAYERS
from rl_sandbox.priors.gaussian import GaussianPrior

# This is for script run
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

seed = args.seed

task_dim = 1
obs_dim = 2 + task_dim
action_dim = 1
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)

device = torch.device("cuda:0")
device = torch.device(c.CPU)

action_repeat = 1
num_frames = 1

memory_size = max_total_steps = 50000 // action_repeat

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.AsType(),
    c.BUFFER_SETTING: {
        c.KWARGS: {
            c.MEMORY_SIZE: memory_size,
            c.OBS_DIM: (obs_dim,),
            c.H_STATE_DIM: (1,),
            c.ACTION_DIM: (action_dim,),
            c.REWARD_DIM: (1,),
            c.INFOS: {c.MEAN: ((action_dim,), np.float32),
                      c.VARIANCE: ((action_dim,), np.float32),
                      c.ENTROPY: ((action_dim,), np.float32),
                      c.LOG_PROB: ((1,), np.float32),
                      c.VALUE: ((1,), np.float32),
                      c.DISCOUNTING: ((1,), np.float32),},
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

    # DIAYN
    c.KL_APPROXIMATION_SAMPLES: 100,
    c.TASK_DIM: task_dim,

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
    c.OBS_DIM: obs_dim,

    # Evaluation
    c.EVALUATION_FREQUENCY: 5000,
    c.EVALUATION_RENDER: False,
    c.EVALUATION_RETURNS: [],
    c.NUM_EVALUATION_EPISODES: 5,

    # Exploration
    c.EXPLORATION_STEPS: 10000,
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
    c.SAVE_INTERVAL: 1000000,

    # Model
    c.DISCRIMINATOR_SETTING: {
        c.MODEL_ARCHITECTURE: FullyConnectedGaussian,
        c.KWARGS: {
            c.INPUT_DIM: obs_dim - task_dim,
            c.OUTPUT_DIM: task_dim,
            c.LAYERS: POLICY_BASED_LINEAR_LAYERS(in_dim=obs_dim - task_dim),
            c.DEVICE: device,
        },
    },

    c.MODEL_SETTING: {
        c.MODEL_ARCHITECTURE: FullyConnectedSeparate,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: VALUE_BASED_LINEAR_LAYERS(in_dim=obs_dim),
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.POLICY: {
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
        c.DISCRIMINATOR: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-3,
            },
        },
    },

    c.PRIOR: GaussianPrior(loc=torch.zeros((task_dim,)),
                           scale=torch.ones((task_dim,)),
                           device=device),

    # SAC
    c.ACCUM_NUM_GRAD: 1,
    c.BATCH_SIZE: 256,
    c.BUFFER_WARMUP: 1000,
    c.EVALUATION_PREPROCESSING: gt.Identity(),
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
    c.TRAIN_PREPROCESSING: gt.Identity(),
    c.UPDATE_NUM: 0,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [],

    # Save
    c.SAVE_PATH: f"../results/mountaincarcontinuous/gt-diayn-separate/{seed}",

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_sac_diayn(experiment_config=experiment_setting)
