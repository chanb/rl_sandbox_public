import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Normal

from rl_sandbox.constants import OBS_RMS, CPU
from rl_sandbox.model_architectures.actor_critics.actor_critic import ActorCritic
from rl_sandbox.model_architectures.shared import Conv2DEncoder, Flatten, Fuse, Split
from rl_sandbox.model_architectures.utils import construct_linear_layers


class EarlyFusionConv2DGaussianAC(ActorCritic):
    def __init__(self,
                 img_dim,
                 scalar_feature_dim,
                 action_dim,
                 shared_layers,
                 shared_out_dim,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        assert len(img_dim) == 4
        super().__init__(obs_dim=scalar_feature_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._eps = eps
        self._img_dim = img_dim
        self._scalar_feature_dim = scalar_feature_dim
        self.split = Split([int(np.product(img_dim)), scalar_feature_dim])
        self.fuse = Fuse()
        self.encoder = Conv2DEncoder(*img_dim[1:], shared_out_dim, shared_layers, nn.LayerNorm(50))
        self.action_net = nn.Sequential(nn.Linear(shared_out_dim * self._img_dim[0] + scalar_feature_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, action_dim * 2))
        self.value = nn.Sequential(nn.Linear(shared_out_dim * self._img_dim[0] + scalar_feature_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
        self.to(self.device)

    def forward(self, x, h, **kwargs):
        batch_size = x.shape[0]

        if self._scalar_feature_dim > 0:
            (imgs, scalars) = self.split(x)
            if hasattr(self, OBS_RMS):
                scalars = self.obs_rms.normalize(scalars)
        else:
            imgs = x
            scalars = torch.empty(batch_size, 0, device=self.device)

        imgs = imgs.reshape(batch_size * self._img_dim[0], *self._img_dim[1:]).to(self.device)
        z = self.encoder(imgs)

        x = self.fuse((z.reshape(batch_size, -1), scalars.to(self.device)))
        a_mean, a_log_std = torch.chunk(self.action_net(x), chunks=2, dim=1)
        a_std = torch.nn.functional.softplus(a_log_std) + self._eps
        dist = Normal(a_mean, a_std)

        val = self.value(x)

        return dist, val, h
