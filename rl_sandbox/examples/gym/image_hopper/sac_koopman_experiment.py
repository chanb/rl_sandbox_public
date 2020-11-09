import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt
import rl_sandbox.transforms.image_transforms as it

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.envs.wrappers.pixel import GymPixelWrapper
from rl_sandbox.train.train_sac import train_sac
from rl_sandbox.model_architectures.actor_critics.conv_soft_actor_critic import EarlyFusionConv2DGaussianSAC
from rl_sandbox.model_architectures.shared import Conv2DDecoder
from rl_sandbox.model_architectures.layers_definition import SAC_DECODER, SAC_ENCODER

# This is for script run
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

seed = args.seed

action_repeat = 4
num_frames = 3
scalar_feature_dim = 0
action_dim = 3
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)

render_h, render_w = 100, 100
processed_h, processed_w = 84, 84
raw_img_dim = (3 * num_frames, render_h, render_w)
img_dim = (1, 3 * num_frames, processed_h, processed_w)
obs_dim = int(np.product(raw_img_dim) + scalar_feature_dim)
latent_dim = 50

memory_size = max_total_steps = 1000000 // action_repeat

device = torch.device("cuda:0")
# device = torch.device(c.CPU)

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {
        c.KOOPMAN: {
            c.BATCH_SIZE: 256,
            c.DEVICE: device,

            # Dimensions
            c.REC_DIM: img_dim,
            c.Z_DIM: latent_dim,
            c.U_DIM: action_dim,

            # Loss
            c.LOSS_COEF: 1.,
            c.REDUCTION: c.MEAN,

            # Model
            c.MODEL_SETTING: {
                c.MODEL_ARCHITECTURE: Conv2DDecoder,
                c.KWARGS:{
                    c.LAYERS: SAC_DECODER(out_channels=img_dim[1]),
                    c.INPUT_SIZE: latent_dim,
                }
            },
            
            # Optimizer
            c.OPTIMIZER_SETTING: {
                c.OPTIMIZER: torch.optim.Adam,
                c.KWARGS: {
                    c.LR: 3e-4,
                },
            },
        },
    },

    # Buffer
    c.BUFFER_PREPROCESSING: gt.Compose([
        gt.Transpose((0, 3, 1, 2)),
        gt.Reshape(),
    ]),
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
                      c.DISCOUNTING: ((1,), np.float32)},
            c.DTYPE: np.uint8,
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
            c.ENV_NAME: "Hopper-v2"
        },
        c.ENV_TYPE: c.GYM,
        c.ENV_WRAPPERS: [
            {
                c.WRAPPER: GymPixelWrapper,
                c.KWARGS: {
                    c.RENDER_H: render_h,
                    c.RENDER_W: render_w,
                },
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
    c.MIN_ACTION: -np.ones(action_dim),
    c.MAX_ACTION: np.ones(action_dim),

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
    c.MODEL_SETTING: {
        c.MODEL_ARCHITECTURE: EarlyFusionConv2DGaussianSAC,
        c.KWARGS: {
            c.IMG_DIM: img_dim,
            c.SCALAR_FEATURE_DIM: scalar_feature_dim,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: SAC_ENCODER(in_channels=img_dim[1]),
            c.SHARED_OUT_DIM: latent_dim,
            c.INITIAL_ALPHA: 0.2,
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.OPTIMIZER: torch.optim.Adam,
        c.KWARGS: {
            c.LR: 3e-4,
        },
    },

    # SAC
    c.ACCUM_NUM_GRAD: 1,
    c.BATCH_SIZE: 100,
    c.BUFFER_WARMUP: 1000,
    c.EVALUATION_PREPROCESSING: gt.Compose([it.CenterCrop(raw_img_dim, height=processed_h, width=processed_w), gt.Normalize(mean=0., var=255.)]),
    c.GAMMA: 0.99,
    c.LEARN_ALPHA: False,
    c.MAX_GRAD_NORM: 1e10,
    c.NUM_GRADIENT_UPDATES: 50,
    c.NUM_PREFETCH: 1,
    c.REWARD_SCALING: 1.,
    c.STEPS_BETWEEN_UPDATE: 50,
    c.TARGET_ENTROPY: -3.,
    c.TARGET_UPDATE_INTERVAL: 1,
    c.TAU: 0.005,
    c.TRAIN_PREPROCESSING: gt.Compose([it.CenterCrop(raw_img_dim, height=processed_h, width=processed_w), gt.Normalize(mean=0., var=255.)]),
    c.UPDATE_NUM: 0,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [0],

    # Save
    c.SAVE_PATH: f"../results/mujoco/hopper-v2/image-sac-koopman/curl_encoder/{seed}",

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_sac(experiment_config=experiment_setting)
