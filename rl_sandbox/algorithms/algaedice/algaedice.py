import copy
import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class AlgaeDICE:
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ AlgaeDICE algorithm: https://arxiv.org/abs/1912.02074
        NOTE: We combine the Q1 and Q2 loss and update both Q functions at the same time.
        """
        self.model = model
        self.policy_opt = policy_opt
        self.qs_opt = qs_opt
        self.alpha_opt = alpha_opt
        self.learn_alpha = learn_alpha
        self.buffer = buffer
        self.algo_params = algo_params
        self.step = 0
        self._update_num = algo_params.get(c.UPDATE_NUM, 0)

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))

        self._actor_update_interval = algo_params.get(
            c.ACTOR_UPDATE_INTERVAL, c.DEFAULT_SAC_PARAMS[c.ACTOR_UPDATE_INTERVAL])
        self._target_update_interval = algo_params.get(
            c.TARGET_UPDATE_INTERVAL, c.DEFAULT_SAC_PARAMS[c.TARGET_UPDATE_INTERVAL])
        self._tau = algo_params.get(
            c.TAU, c.DEFAULT_SAC_PARAMS[c.TAU])
        self._steps_between_update = algo_params.get(
            c.STEPS_BETWEEN_UPDATE, c.DEFAULT_SAC_PARAMS[c.STEPS_BETWEEN_UPDATE])
        self._target_entropy = algo_params.get(
            c.TARGET_ENTROPY, c.DEFAULT_SAC_PARAMS[c.TARGET_ENTROPY])
        self._buffer_warmup = algo_params.get(
            c.BUFFER_WARMUP, c.DEFAULT_SAC_PARAMS[c.BUFFER_WARMUP])
        self._reward_scaling = algo_params.get(
            c.REWARD_SCALING, c.DEFAULT_SAC_PARAMS[c.REWARD_SCALING])

        self._gamma = algo_params.get(c.GAMMA, c.DEFAULT_SAC_PARAMS[c.GAMMA])

        self._num_gradient_updates = algo_params.get(
            c.NUM_GRADIENT_UPDATES, c.DEFAULT_SAC_PARAMS[c.NUM_GRADIENT_UPDATES])
        self._batch_size = algo_params.get(
            c.BATCH_SIZE, c.DEFAULT_SAC_PARAMS[c.BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_SAC_PARAMS[c.ACCUM_NUM_GRAD])
        self._num_prefetch = algo_params.get(
            c.NUM_PREFETCH, 1)

        self._aux_tasks = aux_tasks

        assert self._batch_size % self._accum_num_grad == 0
        assert self._num_gradient_updates % self._num_prefetch == 0

        self._num_samples_per_accum = self._batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_SAC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._initialize_target_network()

        # AlgaeDICE parameters
        self._critic_mixture_ratio = algo_params.get(
            c.CRITIC_MIXTURE_RATIO, c.DEFAULT_ALGAEDICE_PARAMS[c.CRITIC_MIXTURE_RATIO])
        self._algae_alpha = algo_params.get(c.ALGAE_ALPHA, c.DEFAULT_ALGAEDICE_PARAMS[c.ALGAE_ALPHA])
        self._p_norm_exp = algo_params.get(c.P_NORM_EXP, c.DEFAULT_ALGAEDICE_PARAMS[c.P_NORM_EXP])
        assert self._p_norm_exp >= 1, f"p-norm needs to be at least 1. Got: {self._p_norm_exp}"
        self._f = lambda x: torch.abs(x) ** self._p_norm_exp / self._p_norm_exp

        # NOTE: What if we remove this? This is forcing the gradient to be non-negative
        self._f_grad = lambda resid: torch.pow(torch.clamp(resid, min=0.), self._p_norm_exp - 1)


    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.POLICY_OPTIMIZER] = self.policy_opt.state_dict()
        state_dict[c.QS_OPTIMIZER] = self.qs_opt.state_dict()
        state_dict[c.ALPHA_OPTIMIZER] = self.alpha_opt.state_dict()
        if hasattr(self.model, c.OBS_RMS):
            state_dict[c.OBS_RMS] = self.model.obs_rms
        if hasattr(self.model, c.VALUE_RMS):
            state_dict[c.VALUE_RMS] = self.model.value_rms
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self.policy_opt.load_state_dict(state_dict[c.POLICY_OPTIMIZER])
        self.qs_opt.load_state_dict(state_dict[c.QS_OPTIMIZER])
        self.alpha_opt.load_state_dict(state_dict[c.ALPHA_OPTIMIZER])
        if hasattr(self.model, c.OBS_RMS) and c.OBS_RMS in state_dict:
            self.model.obs_rms = state_dict[c.OBS_RMS]
        if hasattr(self.model, c.VALUE_RMS) and c.VALUE_RMS in state_dict:
            self.model.value_rms = state_dict[c.VALUE_RMS]

        self._initialize_target_network()

    def _initialize_target_network(self):
        self._target_model = copy.deepcopy(self.model)
        for param in self._target_model.parameters():
            param.requires_grad = False

    def _update_target_network(self):
        for (param, target_param) in zip(self.model.soft_update_parameters, self._target_model.soft_update_parameters):
            target_param.data.mul_(1. - self._tau)
            target_param.data.add_(param.data * self._tau)

        if hasattr(self.model, c.OBS_RMS):
            self._target_model.obs_rms = copy.deepcopy(self.model.obs_rms)

    def _critic_mix(self, obss, h_states, targ_h_states, acts):
        min_q_targ, _, _, _ = self._target_model.q_vals(obss, targ_h_states, acts)
        _, q1, q2, _ = self.model.q_vals(obss, h_states, acts)
        return q1 * self._critic_mixture_ratio + min_q_targ * (1 - self._critic_mixture_ratio), q2 * self._critic_mixture_ratio + min_q_targ * (1 - self._critic_mixture_ratio)

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, init_obss, init_h_states, discounting, lengths):
        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            init_acts, _ = self.model.act_lprob(init_obss, init_h_states)
            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)

            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)

        q1_targ, q2_targ = self._critic_mix(next_obss, next_h_states, targ_next_h_states, next_acts)
        v1_next = (q1_targ - self.model.alpha.detach() * next_lprobs)
        v2_next = (q2_targ - self.model.alpha.detach() * next_lprobs)

        targ_1 = rews + (self._gamma ** discounting) * (1 - dones) * v1_next
        targ_2 = rews + (self._gamma ** discounting) * (1 - dones) * v2_next

        _, init_q1_val, init_q2_val, _ = self.model.q_vals(init_obss, init_h_states, init_acts)

        # NOTE: This is different from the original implementation. This might be mathematically more correct?
        q1_loss = self._algae_alpha * (self._f((targ_1 - q1_val) / self._algae_alpha)).sum() + (1 - self._gamma) * init_q1_val.sum()
        q2_loss = self._algae_alpha * (self._f((targ_2 - q2_val) / self._algae_alpha)).sum() + (1 - self._gamma) * init_q2_val.sum()

        return q1_loss, q2_loss

    def _compute_pi_loss(self, obss, h_states, acts, rews, dones, next_obss, init_obss, init_h_states, discounting, lengths):
        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        init_acts, _ = self.model.act_lprob(init_obss, init_h_states)
        next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)

        with torch.no_grad():
            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)
        q1_targ, q2_targ = self._critic_mix(next_obss, next_h_states, targ_next_h_states, next_acts)
        v1_next = (q1_targ - self.model.alpha.detach() * next_lprobs)
        v2_next = (q2_targ - self.model.alpha.detach() * next_lprobs)

        targ_1 = rews + (self._gamma ** discounting) * (1 - dones) * v1_next
        targ_2 = rews + (self._gamma ** discounting) * (1 - dones) * v2_next

        _, init_q1_val, init_q2_val, _ = self.model.q_vals(init_obss, init_h_states, init_acts)

        # NOTE: This is different from the original implementation. This might be mathematically more correct?
        # The AlgaeDICE's alpha gets cancelled out from chain rule.
        with torch.no_grad():
            grad_1 = self._f_grad(self._f((targ_1 - q1_val) / self._algae_alpha)).detach()
            grad_2 = self._f_grad(self._f((targ_2 - q2_val) / self._algae_alpha)).detach()
        pi1_loss = (grad_1 * (targ_1 - q1_val)).sum() + (1 - self._gamma) * init_q1_val.sum()
        pi2_loss = (grad_2 * (targ_2 - q2_val)).sum() + (1 - self._gamma) * init_q2_val.sum()

        return -(pi1_loss + pi2_loss) / 2

    def _compute_alpha_loss(self, obss, h_states, lengths):
        with torch.no_grad():
            _, lprobs = self.model.act_lprob(obss, h_states, lengths=lengths)
        alpha_loss = (-self.model.alpha * (lprobs + self._target_entropy).detach()).sum()

        return alpha_loss

    def update_qs(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, init_obss, init_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.qs_opt.zero_grad()
        total_q1_loss = 0.
        total_q2_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            q1_loss, q2_loss = self._compute_qs_loss(obss[opt_idxes],
                                                     h_states[opt_idxes],
                                                     acts[opt_idxes],
                                                     rews[opt_idxes],
                                                     dones[opt_idxes],
                                                     next_obss[opt_idxes],
                                                     init_obss[opt_idxes],
                                                     init_h_states[opt_idxes],
                                                     discounting[opt_idxes],
                                                     lengths[opt_idxes])
            q1_loss /= self._batch_size
            q2_loss /= self._batch_size
            qs_loss = q1_loss + q2_loss
            total_q1_loss += q1_loss.detach().cpu()
            total_q2_loss += q2_loss.detach().cpu()
            qs_loss.backward()

        nn.utils.clip_grad_norm_(self.model.qs_parameters,
                                self._max_grad_norm)
        self.qs_opt.step()
        update_info[c.Q_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.Q1_LOSS].append(total_q1_loss.numpy())
        update_info[c.Q2_LOSS].append(total_q2_loss.numpy())

    def update_policy(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, init_obss, init_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.policy_opt.zero_grad()
        total_pi_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            pi_loss = self._compute_pi_loss(obss[opt_idxes],
                                            h_states[opt_idxes],
                                            acts[opt_idxes],
                                            rews[opt_idxes],
                                            dones[opt_idxes],
                                            next_obss[opt_idxes],
                                            init_obss[opt_idxes],
                                            init_h_states[opt_idxes],
                                            discounting[opt_idxes],
                                            lengths[opt_idxes])
            pi_loss /= self._batch_size
            total_pi_loss += pi_loss.detach().cpu()
            pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.policy_parameters,
                                self._max_grad_norm)
        self.policy_opt.step()
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.PI_LOSS].append(total_pi_loss.numpy())

    def update_alpha(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, init_obss, init_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.alpha_opt.zero_grad()
        total_alpha_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            alpha_loss = self._compute_alpha_loss(obss[opt_idxes],
                                                h_states[opt_idxes],
                                                lengths[opt_idxes])
            alpha_loss /= self._batch_size
            total_alpha_loss += alpha_loss.detach().cpu()
            alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.model.log_alpha,
                                self._max_grad_norm)
        self.alpha_opt.step()
        update_info[c.ALPHA_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.ALPHA_LOSS].append(total_alpha_loss.numpy())

    def _store_to_buffer(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.buffer.push(curr_obs, curr_h_state, act, rew, [done], info, next_obs=next_obs, next_h_state=next_h_state)

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self._store_to_buffer(curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state)
        self.step += 1

        update_info = {}

        if hasattr(self.model, c.OBS_RMS):
            self.model.obs_rms.update(self.eval_preprocessing(torch.tensor(curr_obs)))

        # Perform SAC update
        if self.step >= self._buffer_warmup and self.step % self._steps_between_update == 0:
            update_info[c.PI_LOSS] = []
            update_info[c.Q1_LOSS] = []
            update_info[c.Q2_LOSS] = []
            update_info[c.ALPHA] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.Q_UPDATE_TIME] = []
            update_info[c.POLICY_UPDATE_TIME] = []
            update_info[c.ALPHA_LOSS] = []
            update_info[c.ALPHA_UPDATE_TIME] = []

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths = self.buffer.sample_with_next_obs(
                    self._batch_size * self._num_prefetch, next_obs, next_h_state)
                
                init_obss, init_h_states = self.buffer.sample_init_obs(self._batch_size * self._num_prefetch)

                obss = self.train_preprocessing(obss)
                next_obss = self.train_preprocessing(next_obss)
                init_obss = self.train_preprocessing(init_obss)

                rews = rews * self._reward_scaling
                discounting = infos[c.DISCOUNTING]
                update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)

                for batch_i in range(self._num_prefetch):
                    self._update_num += 1
                    batch_start_idx = batch_i * self._batch_size

                    # Update Q functions
                    # Auxiliary tasks are usually for shared layers, which is updated along with Q
                    aux_loss, aux_update_info = self._aux_tasks.compute_loss(next_obs, next_h_state)
                    if hasattr(aux_loss, c.BACKWARD):
                        aux_loss.backward()
                    self.update_qs(batch_start_idx,
                                   obss,
                                   h_states,
                                   acts,
                                   rews,
                                   dones,
                                   next_obss,
                                   next_h_states,
                                   init_obss,
                                   init_h_states,
                                   discounting,
                                   infos,
                                   lengths,
                                   update_info)
                    self._aux_tasks.step()
                    update_info.update(aux_update_info)

                    if self._update_num % self._actor_update_interval == 0:
                        # Update policy
                        self.update_policy(batch_start_idx,
                                           obss,
                                           h_states,
                                           acts,
                                           rews,
                                           dones,
                                           next_obss,
                                           next_h_states,
                                           init_obss,
                                           init_h_states,
                                           discounting,
                                           infos,
                                           lengths,
                                           update_info)

                        # Update Alpha
                        if self.learn_alpha:
                            self.update_alpha(batch_start_idx,
                                              obss,
                                              h_states,
                                              acts,
                                              rews,
                                              dones,
                                              next_obss,
                                              next_h_states,
                                              init_obss,
                                              init_h_states,
                                              discounting,
                                              infos,
                                              lengths,
                                              update_info)

                    if self._update_num % self._target_update_interval == 0:
                        update_info[c.TARGET_UPDATE_TIME] = []
                        tic = timeit.default_timer()
                        self._update_target_network()
                        update_info[c.TARGET_UPDATE_TIME].append(timeit.default_timer() - tic)

                    update_info[c.ALPHA].append(self.model.alpha.detach().cpu().numpy())

            if hasattr(self.model, c.VALUE_RMS):
                update_info[f"{c.VALUE_RMS}/{c.MEAN}"] = self.model.value_rms.mean.numpy()
                update_info[f"{c.VALUE_RMS}/{c.VARIANCE}"] = self.model.value_rms.var.numpy()

            return True, update_info
        return False, update_info
