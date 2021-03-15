import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class PPO:
    def __init__(self, model, optimizer, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ PPO Algorithm: https://arxiv.org/abs/1707.06347
        """
        self._optimizer = optimizer
        self.model = model
        self.buffer = buffer
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))

        self._gamma = algo_params.get(c.GAMMA, c.DEFAULT_PPO_PARAMS[c.GAMMA])
        self._gae_lambda = algo_params.get(
            c.GAE_LAMBDA, c.DEFAULT_PPO_PARAMS[c.GAE_LAMBDA])
        self._batch_size = algo_params.get(
            c.STEPS_BETWEEN_UPDATE, c.DEFAULT_PPO_PARAMS[c.STEPS_BETWEEN_UPDATE])
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_PPO_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_PPO_PARAMS[c.OPT_BATCH_SIZE])
        self._clip_param = algo_params.get(
            c.CLIP_PARAM, c.DEFAULT_PPO_PARAMS[c.CLIP_PARAM])
        self._clip_value = algo_params.get(
            c.CLIP_VALUE, c.DEFAULT_PPO_PARAMS[c.CLIP_VALUE])
        self._normalize_advantage = algo_params.get(
            c.NORMALIZE_ADVANTAGE, c.DEFAULT_PPO_PARAMS[c.NORMALIZE_ADVANTAGE])

        self._pg_coef = algo_params.get(
            c.PG_COEF, c.DEFAULT_PPO_PARAMS[c.PG_COEF])
        self._v_coef = algo_params.get(
            c.V_COEF, c.DEFAULT_PPO_PARAMS[c.V_COEF])
        self._ent_coef = algo_params.get(
            c.ENT_COEF, c.DEFAULT_PPO_PARAMS[c.ENT_COEF])

        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_PPO_PARAMS[c.ACCUM_NUM_GRAD])

        self._aux_tasks = aux_tasks

        assert self._batch_size % self._opt_batch_size == 0
        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_PPO_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._sampler = BatchSampler(sampler=SubsetRandomSampler(range(self._batch_size)),
                                     batch_size=self._opt_batch_size,
                                     drop_last=True)

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()
        state_dict[c.AUXILIARY_TASKS] = self._aux_tasks.state_dict()
        if hasattr(self.model, c.OBS_RMS):
            state_dict[c.OBS_RMS] = self.model.obs_rms
        if hasattr(self.model, c.VALUE_RMS):
            state_dict[c.VALUE_RMS] = self.model.value_rms

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])
        self._aux_tasks.load_state_dict(state_dict[c.AUXILIARY_TASKS])
        if hasattr(self.model, c.OBS_RMS) and c.OBS_RMS in state_dict:
            self.model.obs_rms = state_dict[c.OBS_RMS]
        if hasattr(self.model, c.VALUE_RMS) and c.VALUE_RMS in state_dict:
            self.model.value_rms = state_dict[c.VALUE_RMS]

    def _compute_returns_advantages(self, rewards, discounting, values, dones):
        returns = torch.zeros(values.shape, dtype=torch.float)

        # PopArt: https://arxiv.org/abs/1602.07714
        if hasattr(self.model, c.VALUE_RMS):
            values = self.model.value_rms.unnormalize(values)

        if self._gae_lambda is not None:
            # GAE: https://arxiv.org/abs/1506.02438
            gae = 0
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + (self._gamma ** discounting[step]) * \
                    values[step + 1] * (1 - dones[step]) - values[step]
                gae = delta + self._gamma * \
                    self._gae_lambda * (1 - dones[step]) * gae
                returns[step] = gae + values[step]
        else:
            for step in reversed(range(len(rewards))):
                returns[step] = returns[step + 1] * \
                    (self._gamma ** discounting[step]) * \
                    (1 - dones[step]) + rewards[step]

        advantages = returns[:-1] - values[:-1]

        if self._normalize_advantage:
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-5)

        if hasattr(self.model, c.VALUE_RMS):
            self.model.value_rms.update(returns)
            returns = self.model.value_rms.normalize(returns)

        return returns, advantages

    def _compute_losses(self, obss, h_states, acts, rets, vals, advs, old_lprobs, lengths):
        new_log_probs, new_vals, entropies = self.model.evaluate_action(obss, h_states, acts, lengths=lengths)
        entropies = entropies.sum()

        ratio = torch.exp(new_log_probs - old_lprobs)
        pg_surr_1 = ratio * advs
        pg_surr_2 = torch.clamp(
            ratio, 1 - self._clip_param, 1 + self._clip_param) * advs
        pg_loss = -torch.min(pg_surr_1, pg_surr_2).sum()

        if self._clip_value:
            clipped_value = vals + \
                torch.clamp(new_vals - vals, -
                            self._clip_param, self._clip_param)
            v_surr_1 = ((rets - new_vals) ** 2)
            v_surr_2 = ((rets - clipped_value) ** 2)
            v_loss = torch.max(v_surr_1, v_surr_2).sum()
        else:
            v_loss = ((rets - new_vals) ** 2).sum()

        return pg_loss, v_loss, entropies

    def update_ac(self, idxes, obss, h_states, acts, rets, vals, advs, old_lprobs, lengths, update_info):
        tic = timeit.default_timer()
        total_pg_loss = 0.
        total_v_loss = 0.
        total_entropy = 0.

        self._optimizer.zero_grad()
        for accum_i in range(self._accum_num_grad):
            opt_idxes = idxes[accum_i * self._num_samples_per_accum: (accum_i + 1) * self._num_samples_per_accum]
            pg_loss, v_loss, entropies = self._compute_losses(obss[opt_idxes],
                                                              h_states[opt_idxes],
                                                              acts[opt_idxes],
                                                              rets[opt_idxes],
                                                              vals[opt_idxes],
                                                              advs[opt_idxes],
                                                              old_lprobs[opt_idxes],
                                                              lengths[opt_idxes])
            pg_loss /= self._opt_batch_size
            v_loss /= self._opt_batch_size
            entropies /= self._opt_batch_size

            total_loss = self._pg_coef * pg_loss + \
                self._v_coef * v_loss - self._ent_coef * entropies
            total_loss.backward()

            total_pg_loss += pg_loss.detach().cpu()
            total_v_loss += v_loss.detach().cpu()
            total_entropy += entropies.detach().cpu()

        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self._max_grad_norm)
        self._optimizer.step()
        update_info[c.PG_LOSS].append(total_pg_loss.numpy())
        update_info[c.V_LOSS].append(total_v_loss.numpy())
        update_info[c.ENTROPIES].append(total_entropy.numpy())
        update_info[c.ACTOR_CRITIC_UPDATE_TIME].append(timeit.default_timer() - tic)

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.buffer.push(curr_obs, curr_h_state, act, rew, [done], info)
        self.step = (self.step + 1) % self._batch_size

        update_info = {}

        # Perform PPO update
        if self.step == 0:
            update_info[c.PG_LOSS] = []
            update_info[c.V_LOSS] = []
            update_info[c.ENTROPIES] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.ADVANTAGE_COMPUTE_TIME] = []
            update_info[c.ACTOR_CRITIC_UPDATE_TIME] = []

            tic = timeit.default_timer()
            obss, h_states, acts, rews, dones, infos, lengths = self.buffer.sample_consecutive(
                self._batch_size)
            obss = self.train_preprocessing(obss)
            update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
            discounting = infos[c.DISCOUNTING]
            old_lprobs = infos[c.LOG_PROB]
            values = infos[c.VALUE]

            if hasattr(self.model, c.OBS_RMS):
                idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
                x_gather = torch.gather(obss, 1, index=idxes - 1)
                self.model.obs_rms.update(x_gather.squeeze(1))

            with torch.no_grad():
                _, last_value, _ = self.model(
                    self.train_preprocessing(torch.tensor([next_obs])), h=torch.tensor([next_h_state]))
                last_value = last_value.to(torch.device(c.CPU))

            vals = torch.cat((values, last_value.detach()), dim=0)
            tic = timeit.default_timer()
            rets, advs = self._compute_returns_advantages(
                rews, discounting, vals, dones)
            update_info[c.ADVANTAGE_COMPUTE_TIME].append(timeit.default_timer() - tic)

            if hasattr(self.model, c.VALUE_RMS):
                update_info[f"{c.VALUE_RMS}/{c.MEAN}"] = self.model.value_rms.mean.numpy()
                update_info[f"{c.VALUE_RMS}/{c.VARIANCE}"] = self.model.value_rms.var.numpy()

            discounting, old_lprobs, vals, rets, advs = discounting.to(self.device), old_lprobs.to(self.device), vals.to(self.device), rets.to(self.device), advs.to(self.device)

            for _ in range(self._opt_epochs):
                sampler = self._sampler.__iter__()
                for idxes in sampler:
                    self._aux_tasks.zero_grad()
                    # NOTE: For now, we just perform auxiliary task update based on gradient step
                    aux_loss, aux_update_info = self._aux_tasks.compute_loss(next_obs, next_h_state)
                    if hasattr(aux_loss, c.BACKWARD):
                        aux_loss.backward()

                    self.update_ac(idxes,
                                   obss,
                                   h_states,
                                   acts,
                                   rets,
                                   vals,
                                   advs,
                                   old_lprobs,
                                   lengths,
                                   update_info)

                    self._aux_tasks.step()

                    update_info.update(aux_update_info)

            self.buffer.clear()
            return True, update_info
        return False, update_info
