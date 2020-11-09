import timeit
import torch

import rl_sandbox.constants as c


class DIAYN:
    def __init__(self, discriminator, prior, discriminator_opt, learning_algorithm, algo_params):
        """ DIAYN Algorithm: https://arxiv.org/abs/1802.06070
        """
        self.learning_algorithm = learning_algorithm
        self.learning_algorithm.diayn_discriminator = discriminator
        self.learning_algorithm.diayn_prior = prior
        self.learning_algorithm.diayn_task_dim = algo_params[c.TASK_DIM]

        self.buffer = self.learning_algorithm.buffer

        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self.prior = prior
        self.kl_approx_samples = algo_params[c.KL_APPROXIMATION_SAMPLES]
        self.task_dim = algo_params[c.TASK_DIM]
        self.device = algo_params[c.DEVICE]

    def state_dict(self):
        state_dict = dict()
        state_dict[c.ALGORITHM] = self.learning_algorithm.state_dict()
        state_dict[c.DISCRIMINATOR] = self.discriminator.state_dict()
        state_dict[c.DISCRIMINATOR_OPTIMIZER] = self.discriminator_opt.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.learning_algorithm.load_state_dict(state_dict[c.ALGORITHM])
        self.discriminator.load_state_dict(state_dict[c.DISCRIMINATOR])
        self.discriminator_opt.load_state_dict(state_dict[c.DISCRIMINATOR_OPTIMIZER])

    def update_discriminator(self, curr_obs):
        update_info = dict()
        tic = timeit.default_timer()
        self.discriminator_opt.zero_grad()

        samples = self.prior.sample((self.kl_approx_samples,))

        curr_obs = torch.as_tensor(curr_obs[:, :-self.task_dim], device=self.device)
        curr_obs = curr_obs.repeat(self.kl_approx_samples, *[1] * (len(curr_obs.shape) - 1))

        q_zs = self.discriminator.lprob(curr_obs, samples)
        p_z = self.prior.lprob(samples)
        loss = -(q_zs - p_z).mean()
        loss.backward()
        self.discriminator_opt.step()
        update_info[f"{c.DIAYN}/{c.DISCRIMINATOR_UPDATE_TIME}"] = timeit.default_timer() - tic
        update_info[f"{c.DIAYN}/{c.DISCRIMINATOR_LOSS}"] = loss.detach().cpu()
        return update_info

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        discriminator_update_info  = self.update_discriminator(curr_obs)
        updated, update_info = self.learning_algorithm.update(curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state)
        update_info.update(discriminator_update_info)

        return updated, update_info
