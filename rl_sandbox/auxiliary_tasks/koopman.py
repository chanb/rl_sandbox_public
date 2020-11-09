import numpy as np
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask
from rl_sandbox.model_architectures.shared import Flatten

class Koopman(AuxiliaryTask):
    def __init__(self,
                 rec_dim,
                 batch_size,
                 encoder,
                 decoder,
                 dynamics,
                 opt,
                 buffer,
                 algo_params,
                 reduction=c.SUM,
                 loss_coef=1.,
                 device=torch.device(c.CPU),
                 **kwargs):
        # Image dim: (num_images, num_frames, height, width)
        assert len(rec_dim) == 4
        super().__init__()
        self._flat = Flatten()

        self._rec_dim = rec_dim
        self._flatten_dim = int(np.product(rec_dim))
        self._batch_size = batch_size

        self._buffer = buffer
        self._encoder = encoder
        self._decoder = decoder
        self._dynamics = dynamics
        self._opt = opt

        self._loss_coef = loss_coef
        self._mse = torch.nn.MSELoss(reduction=reduction)

        self.device = device
        self.algo_params = algo_params
        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

    def state_dict(self):
        return {
            c.DECODER: self._decoder.state_dict(),
            c.KOOPMAN_DYNAMICS: self._dynamics.state_dict(),
            c.KOOPMAN_OPTIMIZER: self._opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._decoder.load_state_dict(state_dict[c.DECODER])
        self._dynamics.load_state_dict(state_dict[c.KOOPMAN_DYNAMICS])
        self._opt.load_state_dict(state_dict[c.KOOPMAN_OPTIMIZER])

    @property
    def opt(self):
        return self._opt

    def compute_loss(self, next_obs, next_h_state):
        obss, _, acts, _, dones, next_obss, _, _ = self._buffer.sample_with_next_obs(
            self._batch_size, next_obs, next_h_state)

        obss = self.train_preprocessing(obss)
        next_obss = self.train_preprocessing(next_obss)

        batch_size = obss.shape[0]

        x = obss[:, :self._flatten_dim].reshape(
            batch_size * self._rec_dim[0], *self._rec_dim[1:]).to(self.device)

        z_hat = self._encoder(x)
        x_hat = self._decoder(z_hat)

        # Compute autoencoder reconstruction loss
        ae_loss = self._mse(x_hat, x)

        # This only looks at observations with valid transitions
        valid_ind = torch.where(dones == 0)[0]

        z_hat = z_hat[valid_ind]
        x_hat = x_hat[valid_ind]

        # Compute MSE K(g(x{n})) + B(u_{n}) and g(x_{n+1})
        next_x = next_obss[valid_ind, :self._flatten_dim].reshape(
            len(valid_ind) * self._rec_dim[0], *self._rec_dim[1:]).to(self.device)
        z_next_hat = self._encoder(next_x)

        z_next_trans = self._dynamics(z_hat, acts[valid_ind])
        transition_loss = self._mse(z_next_hat, z_next_trans)

        # Compute MSE of future state reconstruction
        # Compute reconstruction of K(g(x{n})) + B(u_{n}), which is approximately = g(x_{n+1})
        x_next_hat = self._decoder(z_next_hat)
        x_next_trans = self._decoder(z_next_trans)

        # Compute reconstruction from z_{n+1}
        future_rec_loss = self._mse(x_next_hat, x_next_trans)

        return self._loss_coef * (ae_loss + transition_loss + future_rec_loss)


class KoopmanDynamics(nn.Module):
    def __init__(self, z_dim, u_dim, device=torch.device(c.CPU)):
        super().__init__()
        self.device = device

        self.K = torch.nn.Linear(z_dim, z_dim)
        self.B = torch.nn.Linear(u_dim, z_dim)

        self.to(device)

    def forward(self, z, u):
        z, u = z.to(self.device), u.to(self.device)
        Kz = self.K(z)
        Bu = self.B(u)

        return Kz + Bu
