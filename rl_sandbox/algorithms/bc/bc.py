import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

class BC:
    def __init__(self, model, optimizer, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ Basic behavioral cloning algorithm that minimizes the negative log likelihood
        """

        self._optimizer = optimizer
        self.model = model
        self.buffer = buffer
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._steps_between_update = algo_params.get(
            c.STEPS_BETWEEN_UPDATE, c.DEFAULT_BC_PARAMS[c.STEPS_BETWEEN_UPDATE])
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_BC_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_BC_PARAMS[c.OPT_BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_BC_PARAMS[c.ACCUM_NUM_GRAD])
        self._aux_tasks = aux_tasks

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._train_val_ratio = algo_params.get(
            c.VALIDATION_RATIO, c.DEFAULT_BC_PARAMS[c.VALIDATION_RATIO])
        num_val = int(len(self.buffer) * self._train_val_ratio)
        num_train = len(self.buffer) - num_val
        idxes = np.random.permutation(np.arange(len(self.buffer)))
        # self._val_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[num_train:]),
        #                                  batch_size=self._opt_batch_size,
        #                                  drop_last=True)
        # self._train_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[:num_train]),
        #                                    batch_size=self._opt_batch_size,
        #                                    drop_last=True)

        self._train_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes),
                                           batch_size=self._opt_batch_size,
                                           drop_last=True)

        self.best_validation_loss = np.inf
        self.overfitted = False

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()
        state_dict[c.AUXILIARY_TASKS] = self._aux_tasks.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])
        self._aux_tasks.load_state_dict(state_dict[c.AUXILIARY_TASKS])

    def _compute_bc_loss(self, obss, h_states, acts, lengths):
        # NOTE: Deterministic policy is better?
        # NOTE: If we want to use this BC policy for RL, then most likely we want to train the variance as well.
        # log_probs, _, entropies = self.model.evaluate_action(obss, h_states, acts, lengths=lengths)
        # return -log_probs.sum()
        dist, _, _ = self.model(obss, h_states, lengths=lengths)
        return torch.sum((dist.mean - acts.to(self.device)) ** 2)

    def clone_policy(self, obss, h_states, acts, lengths, update_info):
        tic = timeit.default_timer()

        self._optimizer.zero_grad()
        total_bc_loss = 0.
        for accum_i in range(self._accum_num_grad):
            opt_idxes = range(accum_i * self._num_samples_per_accum,
                              (accum_i + 1) * self._num_samples_per_accum)
            bc_loss = self._compute_bc_loss(obss[opt_idxes],
                                            h_states[opt_idxes],
                                            acts[opt_idxes],
                                            lengths[opt_idxes])
            bc_loss /= self._opt_batch_size
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
                obss, h_states, acts, rews, dones, infos, lengths = self.buffer.sample(self._opt_batch_size, idxes=idxes)
                update_info[c.VALIDATION_SAMPLE_TIME].append(timeit.default_timer() - tic)
                for accum_i in range(self._accum_num_grad):
                    opt_idxes = range(accum_i * self._num_samples_per_accum,
                                    (accum_i + 1) * self._num_samples_per_accum)
                    validation_loss = self._compute_bc_loss(obss[opt_idxes],
                                                            h_states[opt_idxes],
                                                            acts[opt_idxes],
                                                            lengths[opt_idxes])
                    validation_loss /= self._opt_batch_size
                    total_validation_loss += validation_loss.detach().cpu()
            if total_validation_loss > self.best_validation_loss:
                self.overfitted = True
            else:
                self.best_validation_loss = total_validation_loss
                update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
            self.model.train()

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step = (self.step + 1) % self._steps_between_update
        update_info = {}

        # if not self.overfitted and self.step == 0:
        if self.step == 0:
            update_info[c.BC_LOSS] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.POLICY_UPDATE_TIME] = []
            update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
            update_info[c.VALIDATION_SAMPLE_TIME] = []
            for _ in range(self._opt_epochs):
                # NOTE: So far, this mechanism stops training too soon. Why?
                # self.check_overfit(update_info)
                # if self.overfitted:
                #     break

                sampler = self._train_sampler.__iter__()
                for idxes in sampler:
                    tic = timeit.default_timer()
                    obss, h_states, acts, rews, dones, infos, lengths = self.buffer.sample(self._opt_batch_size, idxes=idxes)
                    update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
                    self.clone_policy(obss,
                                      h_states,
                                      acts,
                                      lengths,
                                      update_info)
            return True, update_info
        return False, update_info
