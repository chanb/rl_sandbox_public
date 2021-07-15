import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from rl_sandbox.constants import CPU, OBS_RMS
from rl_sandbox.model_architectures.actor_critics.actor_critic import SquashedGaussianSoftActorCritic, LSTMActorCritic
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers, RunningMeanStd

class FullyConnectedSeparate(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._action_dim = action_dim
        self._flatten = Flatten()
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


class FullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
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

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters())


class LSTMSquashedGaussianSAC(SquashedGaussianSoftActorCritic, LSTMActorCritic):
    def __init__(self,
                 obs_dim,
                 hidden_state_dim,
                 action_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         hidden_state_dim=hidden_state_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False)
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)
        self.lstm_layer = nn.LSTM(input_size=shared_layers[-1][1],
                                  hidden_size=self.hidden_state_dim,
                                  batch_first=True)
        self._policy = nn.Sequential(nn.Linear(self.hidden_state_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(self.hidden_state_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(self.hidden_state_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.to(self.device)

    def _extract_features(self, x, h, lengths=None):
        x = x.to(self.device)
        for layer in self._shared_network:
            x = layer(x)
        x, h = self.lstm_forward(x, h, lengths=lengths)
        return x, h

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters()) + list(self.lstm_layer.parameters())

    @property
    def soft_update_parameters(self):
        return self.qs_parameters + list(self._shared_network.parameters()) + list(self.lstm_layer.parameters())

    def q_vals(self, x, h, a, lengths=None, **kwargs):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size * seq_len, -1)

        a = a.to(self.device)
        x, h = self._extract_features(x, h, lengths=lengths)
        min_q, q1_val, q2_val = self._q_vals(x, a)
        return min_q, q1_val, q2_val, h

    def forward(self, x, h, lengths=None, **kwargs):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size * seq_len, -1)
        x, h = self._extract_features(x, h, lengths=lengths)
        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha * self._lprob(dist, a_mean, t_a_mean)

        return dist, val, h

    def _lprob(self, dist, a, t_a):
        return torch.sum(dist.log_prob(a) - self._squash_gaussian.log_abs_det_jacobian(a, t_a), dim=-1, keepdim=True)

    def act_lprob(self, x, h, lengths=None):
        dist, _, _ = self(x, h, lengths)
        action = dist.rsample()
        t_action = self._squash_gaussian(action)
        log_prob = self._lprob(dist, action, t_action)
        return t_action, log_prob

    def flatten_parameters(self):
        self.lstm_layer.flatten_parameters()


class MultiTaskFullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 task_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False)
        self._task_dim = task_dim
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)
        self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim * action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, task_dim))
        self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, task_dim))
        self._log_alpha = nn.Parameter(torch.ones(task_dim) * torch.log(torch.tensor(initial_alpha)))

        self.to(self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(self._task_dim,), norm_dim=(0,))

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters())

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
