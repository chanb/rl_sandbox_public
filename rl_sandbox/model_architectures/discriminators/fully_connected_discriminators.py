import torch
import torch.nn as nn

from torch.distributions import Normal

from rl_sandbox.constants import CPU
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers


class ActionConditionedFullyConnectedDiscriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, output_dim, layers, device=torch.device(CPU)):
        super().__init__()
        self.device = device

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._output_dim = output_dim

        self._flatten = Flatten()
        self.fc_layers = construct_linear_layers(layers)

        self.output = nn.Linear(layers[-1][1], output_dim)

        self.to(device)

    def forward(self, obss, acts):
        batch_size = obss.shape[0]

        obss = obss.reshape(batch_size, -1)
        x = torch.cat((obss, acts), dim=-1)
        x = self._flatten(x)

        x = x.to(self.device)
        for layer in self.fc_layers:
            x = layer(x)

        logits = self.output(x)

        return logits
