import numpy as np
import torch

from torch.distributions import MultivariateNormal


class CEMQ:
    """ CEM-RL: https://arxiv.org/abs/1810.01222
    CEM: https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf
    This uses Q-function as the scorer. We prefer higher value in this case.
    Assumes independent Gaussians
    """
    def __init__(self,
                 cov_noise_init,
                 cov_noise_end,
                 cov_noise_tau,
                 action_dim,
                 batch_size,
                 num_iters,
                 pop_size,
                 elite_size,
                 device,
                 min_action,
                 max_action):
        assert num_iters > 0

        self._action_dim = action_dim
        self._batch_size = batch_size
        self._num_iters = num_iters
        self._pop_size = pop_size
        self._elite_size = elite_size
        self.min_action = min_action[0]
        self.max_action = max_action[0]
        self.device = device

        # Initialize parameters
        self.weights = torch.tensor([np.log((elite_size + 1) / ii) for ii in range(1, elite_size + 1)], dtype=torch.float, device=device)[None, ..., None]
        self.weights /= torch.sum(self.weights)
        self.cov_noise_init = cov_noise_init
        self.cov_noise_end = cov_noise_end
        self.cov_noise_tau = cov_noise_tau
    
    def compute_action(self, q_function, obss, h_states, act_mean, act_var, lengths):
        obss = obss.repeat(*([1] * (obss.ndim - 1)), self._pop_size).reshape(self._batch_size * self._pop_size, *obss.shape[1:])
        h_states = h_states.repeat(*([1] * (h_states.ndim - 1)), self._pop_size).reshape(self._batch_size * self._pop_size, *h_states.shape[1:])
        if lengths is not None:
            lengths = lengths.repeat(*([1] * (lengths.ndim - 1)), self._pop_size).reshape(self._batch_size * self._pop_size, *lengths.shape[1:])

        cov_noise = self.cov_noise_init
        with torch.no_grad():
            for iter_i in range(self._num_iters):
                eps = torch.randn(self._batch_size, self._pop_size, self._action_dim, device=self.device)
                acts = act_mean[:, None] + eps * act_var[:, None]
                acts = torch.clamp(acts, min=self.min_action, max=self.max_action)
                values = q_function(obss, h_states, acts.reshape(self._batch_size * self._pop_size, self._action_dim), lengths).reshape(self._batch_size, self._pop_size)

                top_vals, top_idxes = torch.sort(values, dim=1)
                top_idxes = top_idxes[:, -self._elite_size:]
                top_vals = top_vals[:, -self._elite_size:]
                top_acts = torch.gather(acts, 1, top_idxes.unsqueeze(-1).expand(self._batch_size, self._elite_size, self._action_dim))

                # CMA-ES uses old mean to compute covariance
                shifted_acts = top_acts - act_mean[:, None]
                act_var = torch.sum(self.weights * (shifted_acts ** 2), dim=1) / self._elite_size + cov_noise
                cov_noise = self.cov_noise_tau * cov_noise + (1 - self.cov_noise_tau) * self.cov_noise_end

                act_mean = torch.mean(top_acts, dim=1)
                act_mean = torch.sum(self.weights * top_acts, dim=1)

                if iter_i + 1 == self._num_iters:
                    return top_acts[:, -1]
