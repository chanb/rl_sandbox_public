import torch

from torch.distributions import Uniform

from rl_sandbox.constants import CPU


class UniformPrior:
    def __init__(self, low, high, device=torch.device(CPU)):
        self.device = device
        self.dist = Uniform(low=low, high=high)

    def sample(self, num_samples):
        return self.dist.rsample(sample_shape=num_samples).to(self.device)

    def lprob(self, samples):
        return self.dist.log_prob(samples)
