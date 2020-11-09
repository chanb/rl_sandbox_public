import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform

from rl_sandbox.constants import OBS_RMS, CPU
from rl_sandbox.model_architectures.actor_critics.actor_critic import SquashedGaussianSoftActorCritic
from rl_sandbox.model_architectures.shared import Conv2DEncoder, Flatten, Fuse, Split
from rl_sandbox.model_architectures.utils import construct_linear_layers


class EarlyFusionConv2DGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 img_dim,
                 scalar_feature_dim,
                 action_dim,
                 shared_layers,
                 shared_out_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        assert len(img_dim) == 4
        super().__init__(obs_dim=scalar_feature_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._img_dim = img_dim
        self._scalar_feature_dim = scalar_feature_dim
        self.split = Split([int(np.product(img_dim)), scalar_feature_dim])
        self.fuse = Fuse()
        self.encoder = Conv2DEncoder(*img_dim[1:], shared_out_dim, shared_layers, nn.LayerNorm(50))

        encoded_dim = shared_out_dim * self._img_dim[0] + scalar_feature_dim
        self._policy = nn.Sequential(nn.Linear(encoded_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(encoded_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(encoded_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._squash_gaussian = TanhTransform()
        self.to(self.device)

    def _extract_features(self, x):
        batch_size = x.shape[0]

        if self._scalar_feature_dim > 0:
            (imgs, scalars) = self.split(x)

            if hasattr(self, OBS_RMS):
                scalars = self.obs_rms.normalize(scalars)
        else:
            imgs = x
            scalars = torch.empty(batch_size, 0, device=self.device)

        imgs = imgs.reshape(batch_size * self._img_dim[0], *self._img_dim[1:]).to(self.device)
        # if batch_size > 1:
        #     print(imgs.shape, scalars.shape)
        #     import matplotlib.pyplot as plt
        #     for i in range(batch_size):
        #         plt.imshow(imgs[i, 0].cpu().numpy())
        #         plt.show()
        z = self.encoder(imgs)
        x = self.fuse((z.reshape(batch_size, -1), scalars.to(self.device)))
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self.encoder.parameters())


class MultiTaskEarlyFusionConv2DGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 img_dim,
                 scalar_feature_dim,
                 action_dim,
                 task_dim,
                 shared_layers,
                 shared_out_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=scalar_feature_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False)
        self._task_dim = task_dim
        self._action_dim = action_dim

        self._img_dim = img_dim
        self._scalar_feature_dim = scalar_feature_dim
        self.split = Split([int(np.product(img_dim)), scalar_feature_dim])
        self.fuse = Fuse()
        self.encoder = Conv2DEncoder(*img_dim[1:], shared_out_dim, shared_layers, nn.LayerNorm(50))

        encoded_dim = shared_out_dim * self._img_dim[0] + scalar_feature_dim
        self._policy = nn.Sequential(nn.Linear(encoded_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim * action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(encoded_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, task_dim))
        self._q2 = nn.Sequential(nn.Linear(encoded_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, task_dim))
        self._log_alpha = nn.Parameter(torch.ones(task_dim) * torch.log(torch.tensor(initial_alpha)))

        self.to(self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(self._task_dim,), norm_dim=(0,))

    def _extract_features(self, x):
        batch_size = x.shape[0]

        if self._scalar_feature_dim > 0:
            (imgs, scalars) = self.split(x)

            if hasattr(self, OBS_RMS):
                scalars = self.obs_rms.normalize(scalars)
        else:
            imgs = x
            scalars = torch.empty(batch_size, 0, device=self.device)

        imgs = imgs.reshape(batch_size * self._img_dim[0], *self._img_dim[1:]).to(self.device)
        # if batch_size > 1:
        #     print(imgs.shape, scalars.shape)
        #     import matplotlib.pyplot as plt
        #     for i in range(batch_size):
        #         plt.imshow(imgs[i, 0].cpu().numpy())
        #         plt.show()
        z = self.encoder(imgs)
        x = self.fuse((z.reshape(batch_size, -1), scalars.to(self.device)))
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self.encoder.parameters())

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_mean = a_mean.reshape(-1, self._task_dim, self._action_dim)
        a_raw_std = a_raw_std.reshape(-1, self._task_dim, self._action_dim)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)[:, 0]
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha[0] * self._lprob(dist, a_mean, t_a_mean)[:, 0]

        return dist, val, h
