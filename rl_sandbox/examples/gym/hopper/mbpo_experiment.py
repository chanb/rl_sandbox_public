import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.dict_to_numpy import GymDictToNumPyWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.envs.wrappers.renderer import GymRenderer
from rl_sandbox.train.train_mbpo_sac import train_mbpo_sac
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import FullyConnectedSeparate, FullyConnectedSquashedGaussianSAC
from rl_sandbox.model_architectures.layers_definition import MBPO_DYNAMICS_LINEAR_LAYERS, MBPO_DYNAMICS_WD_LINEAR_LAYERS, VALUE_BASED_LINEAR_LAYERS
from rl_sandbox.model_architectures.predictive_models.fully_connected_predictive_models import Ensemble, FullyConnectedDeterministic, FullyConnectedGaussian

# This is for script run
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

seed = args.seed

obs_dim = 11
action_dim = 3
reward_dim = 1
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)
# device = torch.device(c.CPU)
device = torch.device("cuda:0")

action_repeat = 1
num_frames = 1
num_models = 7
buffer_warmup = 5000

max_total_steps = 250000 // action_repeat
env_memory_size = 250000
fake_memory_size = 1000000

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.AsType(),
    c.BUFFER_SETTING: {
        c.DYNAMICS: {
            c.KWARGS: {
                c.MEMORY_SIZE: env_memory_size,
                c.OBS_DIM: (obs_dim,),
                c.H_STATE_DIM: (1,),
                c.ACTION_DIM: (action_dim,),
                c.REWARD_DIM: (reward_dim,),
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
            c.STORE_NEXT_OBSERVATION: True,
            c.BUFFER_WRAPPERS: [
                {
                    c.WRAPPER: TorchBuffer,
                    c.KWARGS: {},
                },
            ],
        },
        c.POLICY: {
            c.KWARGS: {
                c.MEMORY_SIZE: fake_memory_size,
                c.OBS_DIM: (obs_dim,),
                c.H_STATE_DIM: (1,),
                c.ACTION_DIM: (action_dim,),
                c.REWARD_DIM: (reward_dim,),
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
            c.STORE_NEXT_OBSERVATION: True,
            c.BUFFER_WRAPPERS: [
                {
                    c.WRAPPER: TorchBuffer,
                    c.KWARGS: {},
                },
            ],
        },
    },

    # Environment
    c.ACTION_DIM: action_dim,
    c.CLIP_ACTION: True,
    c.ENV_SETTING: {
        c.ENV_BASE: {
            c.ENV_NAME: "Hopper-v2"
        },
        c.ENV_TYPE: c.GYM,
        c.ENV_WRAPPERS: [
            {
                c.WRAPPER: GymRenderer,
                c.KWARGS: {}
            },
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
    c.EXPLORATION_STEPS: buffer_warmup,
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
    c.LOG_INTERVAL: 1,

    # Model
    c.MODEL_SETTING: {
        c.DYNAMICS: {
            c.MODEL_ARCHITECTURE: Ensemble,
            c.KWARGS: {
                c.NUM_MODELS: num_models,
                c.STATE_DIM: obs_dim,
                c.REWARD_DIM: reward_dim,
                c.MODEL_CONSTRUCTOR: FullyConnectedGaussian,
                c.MODEL_KWARGS: {
                    c.STATE_DIM: obs_dim,
                    c.ACTION_DIM: action_dim,
                    c.REWARD_DIM: reward_dim,
                    c.LAYERS: MBPO_DYNAMICS_LINEAR_LAYERS(in_dim=obs_dim + action_dim),
                    c.DEVICE: device,
                },
                c.DEVICE: device,
            },
        },
        c.POLICY: {
            c.MODEL_ARCHITECTURE: FullyConnectedSeparate,
            c.KWARGS: {
                c.OBS_DIM: obs_dim,
                c.ACTION_DIM: action_dim,
                c.SHARED_LAYERS: VALUE_BASED_LINEAR_LAYERS(in_dim=obs_dim),
                c.INITIAL_ALPHA: 0.2,
                c.DEVICE: device,
                c.NORMALIZE_OBS: False,
                c.NORMALIZE_VALUE: False,
                c.EPS: 1e-5,
            },
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.DYNAMICS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-3,
            },
        },
        c.POLICY: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
        c.QS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
        c.ALPHA: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
    },

    # Preprocessing
    c.EVALUATION_PREPROCESSING: gt.Identity(),

    # MBPO
    c.MBPO: {
        c.ACCUM_NUM_GRAD: 1,
        c.BATCH_SIZE: 256,
        c.VALIDATION_RATIO: 0.2,
        c.BUFFER_WARMUP: buffer_warmup,
        c.MAX_GRAD_NORM: 10000,
        c.NUM_GRADIENT_UPDATES: 10000000,
        c.NUM_PREFETCH: 1,
        c.STEPS_BETWEEN_UPDATE: 250,
        # Rollout length
        c.K: 1,
        # Number of rollouts
        c.M: 400,
        c.DEVICE: device,
        c.EVALUATION_PREPROCESSING: gt.Identity(),
        c.TRAIN_PREPROCESSING: gt.Identity(),
    },

    # SAC
    c.SAC: {
        c.ACCUM_NUM_GRAD: 1,
        c.ACTOR_UPDATE_INTERVAL: 2,
        # c.BATCH_SIZE: 100,
        c.BATCH_SIZE: 256,
        c.BUFFER_WARMUP: 0,
        c.GAMMA: 0.99,
        c.LEARN_ALPHA: False,
        c.MAX_GRAD_NORM: 1e10,
        c.NUM_GRADIENT_UPDATES: 20,
        c.NUM_PREFETCH: 1,
        c.REWARD_SCALING: 1.,
        c.STEPS_BETWEEN_UPDATE: 1,
        # c.TARGET_ENTROPY: -action_dim,
        c.TARGET_ENTROPY: -1.,
        c.TARGET_UPDATE_INTERVAL: 2,
        # c.TAU: 0.01,
        c.TAU: 5e-3,
        c.UPDATE_NUM: 0,
        c.DEVICE: device,
        c.EVALUATION_PREPROCESSING: gt.Identity(),
        c.TRAIN_PREPROCESSING: gt.Identity(),
    },
    

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [],

    # Save
    # c.SAVE_PATH: f"../results/mujoco/hopper-v2/gt-mbpo-sac-separate/learn_done/{seed}",
    c.SAVE_PATH: None,

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_mbpo_sac(experiment_config=experiment_setting)
