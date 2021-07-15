import copy
import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

class BC:
    def __init__(self, model, optimizer, expert_buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ Basic behavioral cloning algorithm that minimizes the negative log likelihood
        """

        self._optimizer = optimizer
        self.model = model
        self.expert_buffer = expert_buffer
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_BC_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_BC_PARAMS[c.OPT_BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_BC_PARAMS[c.ACCUM_NUM_GRAD])
        self._overfit_tolerance = algo_params.get(
            c.OVERFIT_TOLERANCE, c.DEFAULT_BC_PARAMS[c.OVERFIT_TOLERANCE])
        self._aux_tasks = aux_tasks

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._train_val_ratio = algo_params.get(
            c.VALIDATION_RATIO, c.DEFAULT_BC_PARAMS[c.VALIDATION_RATIO])
        self.num_val = int(len(self.expert_buffer) * self._train_val_ratio)
        self.num_train = len(self.expert_buffer) - self.num_val
        idxes = np.random.permutation(np.arange(len(self.expert_buffer)))
        self._val_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[self.num_train:]),
                                         batch_size=self._opt_batch_size,
                                         drop_last=False)
        self._train_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[:self.num_train]),
                                           batch_size=self._opt_batch_size,
                                           drop_last=False)

        self.best_validation_loss = np.inf
        self._overfit_count = 0
        self.overfitted = False
        self._curr_best_model = copy.deepcopy(self.model.state_dict())

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])

    def _compute_bc_loss(self, obss, h_states, acts, lengths):
        dist, _, _ = self.model(obss, h_states, lengths=lengths)
        return torch.sum((dist.mean - acts.to(self.device)) ** 2)

    def clone_policy(self, update_info):
        sampler = self._train_sampler.__iter__()
        for idxes in sampler:
            tic = timeit.default_timer()
            obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffer.sample(self._opt_batch_size, idxes=idxes)
            update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
            tic = timeit.default_timer()
    
            self._optimizer.zero_grad()
            total_bc_loss = 0.
            for accum_i in range(self._accum_num_grad):
                start_idxes = accum_i * self._num_samples_per_accum
                end_idxes = (accum_i + 1) * self._num_samples_per_accum
                bc_loss = self._compute_bc_loss(obss[start_idxes:end_idxes],
                                                h_states[start_idxes:end_idxes],
                                                acts[start_idxes:end_idxes],
                                                lengths[start_idxes:end_idxes])
                bc_loss /= self.num_train
                bc_loss.backward()
                total_bc_loss += bc_loss.detach().cpu()
    
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self._max_grad_norm)
            self._optimizer.step()
    
            update_info[c.BC_LOSS].append(total_bc_loss.numpy())
            update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)

    def check_overfit(self, update_info):
        with torch.no_grad():
            self.model.eval()
            sampler = self._val_sampler.__iter__()
            total_validation_loss = 0.
            for idxes in sampler:
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffer.sample(self._opt_batch_size, idxes=idxes)
                update_info[c.VALIDATION_SAMPLE_TIME].append(timeit.default_timer() - tic)
                for accum_i in range(self._accum_num_grad):
                    start_idxes = accum_i * self._num_samples_per_accum
                    end_idxes = (accum_i + 1) * self._num_samples_per_accum
                    validation_loss = self._compute_bc_loss(obss[start_idxes:end_idxes],
                                                            h_states[start_idxes:end_idxes],
                                                            acts[start_idxes:end_idxes],
                                                            lengths[start_idxes:end_idxes])
                    validation_loss /= self.num_val
                    total_validation_loss += validation_loss.detach().cpu()
            if total_validation_loss > self.best_validation_loss:
                self._overfit_count += 1
                if self._overfit_count == self._overfit_tolerance:
                    self.overfitted = True
                    self.model.load_state_dict(self._curr_best_model)
            else:
                self._overfit_count = 0
                self.best_validation_loss = total_validation_loss
                update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
                self._curr_best_model = copy.deepcopy(self.model.state_dict())
            self.model.train()

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1
        update_info = {}

        update_info[c.BC_LOSS] = []
        update_info[c.SAMPLE_TIME] = []
        update_info[c.POLICY_UPDATE_TIME] = []
        update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
        update_info[c.VALIDATION_SAMPLE_TIME] = []
        if self.overfitted:
            return False, update_info

        for e in range(self._opt_epochs):
            self.check_overfit(update_info)
            if self.overfitted:
                break
            self.clone_policy(update_info)

        return True, update_info
