import torch
import torch.nn as nn

from torch.distributions import Normal

from rl_sandbox.constants import CPU
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers


class FullyConnectedGaussian(nn.Module):
    def __init__(self, input_dim, output_dim, layers, device=torch.device(CPU)):
        super().__init__()
        self.device = device

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._flatten = Flatten()
        self.fc_layers = construct_linear_layers(layers)

        # Assume independence between dimensions
        self.gaussian_parameters = nn.Linear(layers[-1][1], 2 * output_dim)

        self.to(device)

    def forward(self, x):
        x = self._flatten(x)

        x = x.to(self.device)
        for layer in self.fc_layers:
            x = layer(x)

        out_mean, out_raw_std = torch.chunk(self.gaussian_parameters(x), chunks=2, dim=1)
        out_std = torch.nn.functional.softplus(out_raw_std)

        return Normal(out_mean, out_std)

    def lprob(self, x, samples):
        dist = self.forward(x)
        return dist.log_prob(samples)
