import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

from rl_sandbox.constants import OBS_RMS, CPU
from rl_sandbox.model_architectures.actor_critics.actor_critic import QActorCritic
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers


class FullyConnectedGaussianQACSeparate(QActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._eps = eps
        self._action_dim = action_dim
        self._flatten = Flatten()

        # NOTE: Separate architecture grants stable learning for GRAC
        self._policy = nn.Sequential(nn.Linear(obs_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(obs_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.to(self.device)

    def _extract_features(self, x):
        x = self._flatten(x)

        obs, extra_features = x[:, :self._obs_dim], x[:, self._obs_dim:]
        if hasattr(self, OBS_RMS):
            obs = self.obs_rms.normalize(obs)
        x = torch.cat((obs, extra_features), dim=1)
        x = x.to(self.device)
        return x

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        # NOTE: This hyperbolic tangent is important to get reasonable action log prob
        a_mean = torch.tanh(a_mean)
        # NOTE: If self._eps is too small, we risk running into bad log prob with CEM's choice of action...
        a_std = F.softplus(a_raw_std) + self._eps
        min_q, _, _, _ = self._q_vals(x, h, a_mean)

        return Normal(loc=a_mean, scale=a_std), min_q, h

    @property
    def policy_parameters(self):
        return list(self._policy.parameters())

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters())


class FullyConnectedGaussianQAC(QActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._eps = eps
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)
        self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.to(self.device)

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        # NOTE: This hyperbolic tangent is important to get reasonable action log prob
        a_mean = torch.tanh(a_mean)
        # NOTE: If self._eps is too small, we risk running into bad log prob with CEM's choice of action...
        a_std = F.softplus(a_raw_std) + self._eps
        min_q, _, _, _ = self._q_vals(x, h, a_mean)

        return Normal(loc=a_mean, scale=a_std), min_q, h

    @property
    def policy_parameters(self):
        return list(self._policy.parameters())

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters()) + list(self._shared_network.parameters())


class FullyConnectedGaussianCEMQAC(QActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 cem,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._eps = eps
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)
        self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.to(self.device)

        self._cem = cem

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        # NOTE: This hyperbolic tangent is important to get reasonable action log prob
        a_mean = torch.tanh(a_mean)
        a_std = F.softplus(a_raw_std) + self._eps
        min_q, _, _, _ = self._q_vals(x, h, a_mean)

        return Normal(loc=a_mean, scale=a_std), min_q, h

    @property
    def policy_parameters(self):
        return list(self._policy.parameters())

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters()) + list(self._shared_network.parameters())

    def compute_cem_score(self, x, h, a, lengths):
        return self.q_vals(x, h, a, length=lengths)[1]

    def compute_action(self, x, h, **kwargs):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample().clamp(min=self._cem.min_action, max=self._cem.max_action)
            cem_action = self._cem.compute_action(self.compute_cem_score, x, h, dist.mean, dist.variance, None)
            pi_min_q, _, _, _ = self.q_vals(x, h, action)
            cem_min_q, _, _, _ = self.q_vals(x, h, cem_action)

            if cem_min_q > pi_min_q:
                action = cem_action

            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()
