import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal

from rl_sandbox.constants import OBS_RMS, CPU
from rl_sandbox.model_architectures.actor_critics.actor_critic import ActorCritic, LSTMActorCritic
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers

class FullyConnectedDiscreteActorCritic(ActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)

        self._flatten = Flatten()
        self.shared_network = construct_linear_layers(shared_layers)
        self.action = nn.Linear(shared_layers[-1][1], action_dim)
        self.value = nn.Linear(shared_layers[-1][1], 1)
        self.to(self.device)

    def forward(self, x, h, **kwargs):
        x = self._flatten(x)

        if hasattr(self, OBS_RMS):
            x = self.obs_rms.normalize(x)

        x = x.to(self.device)
        for layer in self.shared_network:
            x = layer(x)
        logits = self.action(x)
        value = self.value(x)

        return Categorical(logits=logits), value, h


class FullyConnectedGaussianAC(ActorCritic):
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
        self._flatten = Flatten()
        self.shared_network = construct_linear_layers(shared_layers)
        self.action_mean = nn.Linear(shared_layers[-1][1], action_dim)
        self.action_raw_std = nn.Linear(shared_layers[-1][1], action_dim)
        self.value = nn.Linear(shared_layers[-1][1], 1)
        self.to(self.device)

    def forward(self, x, h, **kwargs):
        x = self._flatten(x)

        if hasattr(self, OBS_RMS):
            x = self.obs_rms.normalize(x)

        x = x.to(self.device)
        for layer in self.shared_network:
            x = layer(x)
        mean = self.action_mean(x)
        std = F.softplus(self.action_raw_std(x)) + self._eps
        value = self.value(x)

        return Normal(loc=mean, scale=std), value, h


class LSTMGaussianAC(LSTMActorCritic):
    def __init__(self,
                 obs_dim,
                 hidden_state_dim,
                 action_dim,
                 shared_layers,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         hidden_state_dim=hidden_state_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)

        self._flatten = Flatten()
        self.shared_network = construct_linear_layers(shared_layers)
        self.lstm_layer = nn.LSTM(input_size=shared_layers[-1][1],
                                  hidden_size=self.hidden_state_dim,
                                  batch_first=True)
        self.action_mean = nn.Linear(self.hidden_state_dim, action_dim)
        self.action_raw_std = nn.Linear(self.hidden_state_dim, action_dim)
        self.value = nn.Linear(self.hidden_state_dim, 1)
        self.to(self.device)

    def forward(self, x, h, lengths=None, **kwargs):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size * seq_len, -1)

        if hasattr(self, OBS_RMS):
            x = self.obs_rms.normalize(x)

        x = x.to(self.device)
        for layer in self.shared_network:
            x = layer(x)

        x, h = self.lstm_forward(x, h, lengths=lengths)

        mean = self.action_mean(x)
        std = F.softplus(self.action_raw_std(x))
        value = self.value(x)

        return Normal(loc=mean, scale=std), value, h
