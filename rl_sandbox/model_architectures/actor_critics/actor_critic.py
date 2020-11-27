import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform

from rl_sandbox.constants import OBS_RMS, VALUE_RMS, CPU
from rl_sandbox.model_architectures.utils import RunningMeanStd


class ActorCritic(nn.Module):
    def __init__(self,
                 obs_dim,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self._obs_dim = obs_dim

        if normalize_obs:
            if isinstance(obs_dim, int):
                obs_dim = (obs_dim,)
            self.obs_rms = RunningMeanStd(shape=obs_dim, norm_dim=norm_dim)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(1,), norm_dim=(0,))

    def _extract_features(self, x):
        x = self._flatten(x)

        obs, extra_features = x[:, :self._obs_dim], x[:, self._obs_dim:]
        if hasattr(self, OBS_RMS):
            obs = self.obs_rms.normalize(obs)
        x = torch.cat((obs, extra_features), dim=1)
        x = x.to(self.device)
        return x

    def forward(self, x, **kwargs):
        raise NotImplementedError()

    def evaluate_action(self, x, h, a, **kwargs):
        dist, value, _ = self.forward(x, h, **kwargs)
        log_prob = dist.log_prob(a.clone().detach().to(self.device)).sum(dim=-1, keepdim=True)
        return log_prob, value, dist.entropy()

    def compute_action(self, x, h, **kwargs):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h, **kwargs):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.mean
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()


class LSTMActorCritic(ActorCritic):
    def __init__(self,
                 obs_dim,
                 hidden_state_dim,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        self.hidden_state_dim = hidden_state_dim

    def _convert_hidden_state_to_tuple(self, h):
        hidden_state = h[..., :self.hidden_state_dim].contiguous()
        cell_state = h[..., self.hidden_state_dim:].contiguous()
        return (hidden_state, cell_state)

    def _convert_tuple_to_hidden_state(self, h):
        return torch.cat((h[0], h[1]), dim=-1)

    def initialize_hidden_state(self):
        return torch.zeros((1, self.hidden_state_dim * 2))

    def lstm_forward(self, x, h, lengths, **kwargs):
        batch_size = h.shape[0]
        seq_len = h.shape[1]
        if lengths is None:
            lengths = torch.ones(batch_size, dtype=torch.int)

        h = h.transpose(0, 1)[[0]]
        x = x.reshape(batch_size, seq_len, -1)
        h = self._convert_hidden_state_to_tuple(h.to(self.device))

        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, h = self.lstm_layer(x, h)
        output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = output[range(output.shape[0]), input_sizes - 1, :]

        h = self._convert_tuple_to_hidden_state(h).transpose(0, 1)

        return x, h

    def evaluate_action(self, x, h, a, lengths, **kwargs):
        dist, value, _ = self.forward(x, h, lengths=lengths)
        log_prob = dist.log_prob(a.clone().detach().to(self.device)).sum(dim=-1, keepdim=True)
        return log_prob, value, dist.entropy()


class QActorCritic(ActorCritic):
    def __init__(self,
                 obs_dim,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)

    def _q_vals(self, x, h, a):
        input = torch.cat((x, a), dim=1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        min_q = torch.min(q1_val, q2_val)

        return min_q, q1_val, q2_val, h

    def q_vals(self, x, h, a, **kwargs):
        x = self._extract_features(x)
        a = a.to(self.device)
        return self._q_vals(x, h, a)

    def act_lprob(self, x, h, **kwargs):
        dist, _, _ = self(x, h)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def act_stats(self, x, h, **kwargs):
        dist, val, _ = self(x, h)
        action = dist.rsample()

        return action, dist.mean, dist.variance, dist.entropy(), val

    def lprob(self, x, h, a, **kwargs):
        dist, _, _ = self(x, h)
        return dist.log_prob(a).sum(dim=-1, keepdim=True)

    def forward(self, x, h, **kwargs):
        raise NotImplementedError

    @property
    def policy_parameters(self):
        return self._policy.parameters()

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters())


class SoftActorCritic(ActorCritic):
    def __init__(self,
                 obs_dim,
                 initial_alpha=1.,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        assert initial_alpha > 0.
        self._log_alpha = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(initial_alpha)))

    def _q_vals(self, x, h, a):
        input = torch.cat((x, a), dim=1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        min_q = torch.min(q1_val, q2_val)

        return min_q, q1_val, q2_val, h

    def q_vals(self, x, h, a, **kwargs):
        x = self._extract_features(x)
        a = a.to(self.device)
        return self._q_vals(x, h, a)

    def act_lprob(self, x, h, **kwargs):
        dist, _, _ = self(x, h)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def forward(self, x, h, **kwargs):
        raise NotImplementedError

    @property
    def log_alpha(self):
        return self._log_alpha

    @property
    def alpha(self):
        return torch.exp(self._log_alpha)

    @property
    def policy_parameters(self):
        return self._policy.parameters()

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters())

    @property
    def soft_update_parameters(self):
        return self.qs_parameters


class SquashedGaussianSoftActorCritic(SoftActorCritic):
    def __init__(self,
                 obs_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        self._eps = eps
        self._squash_gaussian = TanhTransform()

    def _q_vals(self, x, a):
        input = torch.cat((x, a), dim=1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        min_q = torch.min(q1_val, q2_val)
        return min_q, q1_val, q2_val

    def _lprob(self, dist, a, t_a):
        return torch.sum(dist.log_prob(a) - self._squash_gaussian.log_abs_det_jacobian(a, t_a), dim=-1, keepdim=True)

    def q_vals(self, x, h, a, **kwargs):
        a = a.to(self.device)
        x = self._extract_features(x)
        min_q, q1_val, q2_val = self._q_vals(x, a)
        return min_q, q1_val, q2_val, h

    def act_lprob(self, x, h, **kwargs):
        dist, _, _ = self.forward(x, h)
        action = dist.rsample()
        t_action = self._squash_gaussian(action)
        log_prob = self._lprob(dist, action, t_action)
        return t_action, log_prob

    def compute_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample()
            t_action = self._squash_gaussian(action)
            log_prob = self._lprob(dist, action, t_action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.mean
            t_action = self._squash_gaussian(action)
            log_prob = self._lprob(dist, action, t_action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha * self._lprob(dist, a_mean, t_a_mean)

        return dist, val, h
