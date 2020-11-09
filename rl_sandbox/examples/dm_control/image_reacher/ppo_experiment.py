from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import torch

import rl_sandbox.constants as c
import rl_sandbox.transforms.general_transforms as gt
import rl_sandbox.transforms.image_transforms as it

from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.envs.wrappers.pixel import DMControlPixelWrapper
from rl_sandbox.train.train_ppo import train_ppo
from rl_sandbox.model_architectures.actor_critics.conv_actor_critic import EarlyFusionConv2DGaussianAC
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
action_dim = 2
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)

render_h, render_w = 100, 100
processed_h, processed_w = 84, 84
raw_img_dim = (3 * num_frames, render_h, render_w)
img_dim = (1, 3 * num_frames, processed_h, processed_w)
obs_dim = int(np.product(raw_img_dim) + scalar_feature_dim)
latent_dim = 50

max_total_steps = 100000 // action_repeat
steps_between_update = 256

device = torch.device("cuda:0")
# device = torch.device(c.CPU)

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.Compose([
        gt.Transpose((0, 3, 1, 2)),
        gt.Reshape(),
    ]),
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
            c.DOMAIN_NAME: "reacher",
            c.TASK_NAME: "easy",
        },
        c.ENV_TYPE: c.DM_CONTROL,
        c.ENV_WRAPPERS: [
            {
                c.WRAPPER: DMControlPixelWrapper,
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
    c.MIN_ACTION: min_action,
    c.MAX_ACTION: max_action,

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
        c.MODEL_ARCHITECTURE: EarlyFusionConv2DGaussianAC,
        c.KWARGS: {
            c.IMG_DIM: img_dim,
            c.SCALAR_FEATURE_DIM: scalar_feature_dim,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: SAC_ENCODER(in_channels=img_dim[1]),
            c.SHARED_OUT_DIM: latent_dim,
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: True,
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.OPTIMIZER: torch.optim.Adam,
        c.KWARGS: {
            c.LR: 3e-4,
        },
    },

    c.EVALUATION_PREPROCESSING: gt.Compose([it.CenterCrop(raw_img_dim, height=processed_h, width=processed_w), gt.Normalize(mean=0., var=255.)]),
    c.TRAIN_PREPROCESSING: gt.Compose([it.CenterCrop(raw_img_dim, height=processed_h, width=processed_w), gt.Normalize(mean=0., var=255.)]),

    # PPO
    c.ACCUM_NUM_GRAD: 2,
    c.STEPS_BETWEEN_UPDATE: steps_between_update,
    c.CLIP_PARAM: 0.2,
    c.CLIP_VALUE: True,
    c.ENT_COEF: 0,
    c.GAE_LAMBDA: 0.95,
    c.GAMMA: 0.99,
    c.MAX_GRAD_NORM: 1e10,
    c.NORMALIZE_ADVANTAGE: True,
    c.OPT_BATCH_SIZE: 128,
    c.OPT_EPOCHS: 10,
    c.PG_COEF: 1,
    c.V_COEF: 1,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [0],

    # Save
    c.SAVE_PATH: f"/home/chanb/scratch/results/dm_control/reacher/easy/img/ppo/{seed}",

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_ppo(experiment_config=experiment_setting)
