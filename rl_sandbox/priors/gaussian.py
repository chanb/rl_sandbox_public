import torch

from torch.distributions import Normal

from rl_sandbox.constants import CPU


class GaussianPrior:
    def __init__(self, loc, scale, device=torch.device(CPU)):
        self.device = device
        self.dist = Normal(loc=loc, scale=scale)

    def sample(self, num_samples):
        return self.dist.rsample(sample_shape=num_samples).to(self.device)

    def lprob(self, samples):
        return self.dist.log_prob(samples)
