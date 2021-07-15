import torch.nn as nn

import rl_sandbox.constants as c
import rl_sandbox.model_architectures.shared as snn


NATURE_CNN_ENCODER = lambda in_channels: (
    [in_channels,   32,     (8, 8), (4, 4), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32,            64,     (4, 4), (2, 2), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [64,            64,     (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
)

NATURE_CNN_DECODER = lambda out_channels: (
    [64,            64,             (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [64,            32,             (4, 4), (2, 2), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32,            out_channels,   (8, 8), (4, 4), (0, 0), (1, 1), nn.Identity(), False, 0, False],
)

# SAC-AE: https://arxiv.org/abs/1910.01741
SAC_ENCODER = lambda in_channels: (
    [in_channels, 32, (3, 3), (2, 2), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
)

SAC_DECODER = lambda out_channels: (
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, 32, (3, 3), (1, 1), (0, 0), (1, 1), nn.ReLU(), False, 0, False],
    [32, out_channels, (3, 3), (2, 2), (0, 0), (1, 1), nn.Identity(), False, 0, False],
)

POLICY_BASED_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  128,      nn.ReLU(), True, 0],
    [128,     128,      nn.ReLU(), True, 0],
)

VALUE_BASED_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  256,      nn.ReLU(), True, 0],
    [256,     256,      nn.ReLU(), True, 0],
)

DISCRIMINATOR_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  100,      nn.Tanh(), True, 0],
    [100,     100,      nn.Tanh(), True, 0],
)

SAC_DISCRIMINATOR_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  256,      nn.Tanh(), True, 0],
    [256,     256,      nn.Tanh(), True, 0],
)

MBPO_DYNAMICS_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  200,      snn.Swish(), True, 0],
    [200,     200,      snn.Swish(), True, 0],
    [200,     200,      snn.Swish(), True, 0],
    [200,     200,      snn.Swish(), True, 0],
)
