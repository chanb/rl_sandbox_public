import copy
import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.algorithms.cem.cem import CEMQ
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class GRAC:
    def __init__(self, model, policy_opt, qs_opt, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ GRAC  Algorithm: https://arxiv.org/abs/2009.08973
        """
        self.model = model
        self.policy_opt = policy_opt
        self.qs_opt = qs_opt

        self.buffer = buffer
        self.algo_params = algo_params
        self.step = 0
        self.action_dim = algo_params[c.ACTION_DIM]

        # TODO: They have a scehduler for alpha
        self._alpha = algo_params.get(
            c.ALPHA, c.DEFAULT_GRAC_PARAMS[c.ALPHA])
        self._cov_noise_init = algo_params.get(
            c.COV_NOISE_INIT, c.DEFAULT_GRAC_PARAMS[c.COV_NOISE_INIT])
        self._cov_noise_end = algo_params.get(
            c.COV_NOISE_END, c.DEFAULT_GRAC_PARAMS[c.COV_NOISE_END])
        self._cov_noise_tau = algo_params.get(
            c.COV_NOISE_TAU, c.DEFAULT_GRAC_PARAMS[c.COV_NOISE_TAU])
        self._num_iters = algo_params.get(
            c.NUM_ITERS, c.DEFAULT_GRAC_PARAMS[c.NUM_ITERS])
        self._pop_size = algo_params.get(
            c.POP_SIZE, c.DEFAULT_GRAC_PARAMS[c.POP_SIZE])
        self._elite_size = algo_params.get(
            c.ELITE_SIZE, c.DEFAULT_GRAC_PARAMS[c.ELITE_SIZE])
        self._min_action = algo_params.get(
            c.MIN_ACTION, c.DEFAULT_GRAC_PARAMS[c.MIN_ACTION])
        self._max_action = algo_params.get(
            c.MAX_ACTION, c.DEFAULT_GRAC_PARAMS[c.MAX_ACTION])

        self._update_num = algo_params.get(c.UPDATE_NUM, 0)

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))

        self._num_q_updates = algo_params.get(
            c.NUM_Q_UPDATES, c.DEFAULT_GRAC_PARAMS[c.NUM_Q_UPDATES])
        self._steps_between_update = algo_params.get(
            c.STEPS_BETWEEN_UPDATE, c.DEFAULT_GRAC_PARAMS[c.STEPS_BETWEEN_UPDATE])
        self._buffer_warmup = algo_params.get(
            c.BUFFER_WARMUP, c.DEFAULT_GRAC_PARAMS[c.BUFFER_WARMUP])
        self._reward_scaling = algo_params.get(
            c.REWARD_SCALING, c.DEFAULT_GRAC_PARAMS[c.REWARD_SCALING])

        self._gamma = algo_params.get(c.GAMMA, c.DEFAULT_GRAC_PARAMS[c.GAMMA])

        self._num_gradient_updates = algo_params.get(
            c.NUM_GRADIENT_UPDATES, c.DEFAULT_GRAC_PARAMS[c.NUM_GRADIENT_UPDATES])
        self._batch_size = algo_params.get(
            c.BATCH_SIZE, c.DEFAULT_GRAC_PARAMS[c.BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_GRAC_PARAMS[c.ACCUM_NUM_GRAD])
        self._num_prefetch = algo_params.get(
            c.NUM_PREFETCH, 1)

        self._aux_tasks = aux_tasks

        assert self._batch_size % self._accum_num_grad == 0
        assert self._num_gradient_updates % self._num_prefetch == 0

        self._num_samples_per_accum = self._batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_GRAC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self.cem = CEMQ(cov_noise_init=self._cov_noise_init,
                        cov_noise_end=self._cov_noise_end,
                        cov_noise_tau=self._cov_noise_tau,
                        action_dim=self.action_dim,
                        batch_size=self._num_samples_per_accum,
                        num_iters=self._num_iters,
                        pop_size=self._pop_size,
                        elite_size=self._elite_size,
                        device=self.device,
                        min_action=self._min_action,
                        max_action=self._max_action,)

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.POLICY_OPTIMIZER] = self.policy_opt.state_dict()
        state_dict[c.QS_OPTIMIZER] = self.qs_opt.state_dict()
        if hasattr(self.model, c.OBS_RMS):
            state_dict[c.OBS_RMS] = self.model.obs_rms
        if hasattr(self.model, c.VALUE_RMS):
            state_dict[c.VALUE_RMS] = self.model.value_rms
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self.policy_opt.load_state_dict(state_dict[c.POLICY_OPTIMIZER])
        self.qs_opt.load_state_dict(state_dict[c.QS_OPTIMIZER])
        if hasattr(self.model, c.OBS_RMS) and c.OBS_RMS in state_dict:
            self.model.obs_rms = state_dict[c.OBS_RMS]
        if hasattr(self.model, c.VALUE_RMS) and c.VALUE_RMS in state_dict:
            self.model.value_rms = state_dict[c.VALUE_RMS]

    def construct_q_function(self, q_i):
        def q_function(obss, h_states, acts, lengths):
            res = self.model.q_vals(obss, h_states, acts, lengths=lengths)
            return res[q_i + 1]
        return q_function

    def _compute_qs_loss(self, obss, h_states, acts, dones, best_next_acts, target, targ_q1_best, targ_q2_best, next_obss, lengths):
        best_next_acts, target, targ_q1_best, targ_q2_best, dones = best_next_acts.to(self.device), target.to(self.device), targ_q1_best.to(self.device), targ_q2_best.to(self.device), dones.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)
        _, q1_best, q2_best, _ = self.model.q_vals(next_obss, next_h_states, best_next_acts)

        q1_loss = ((q1_val - target) ** 2).sum()
        q2_loss = ((q2_val - target) ** 2).sum()

        # NOTE: Supposedly we shouldn't be concerned about state at timestep T + 1, assuming the episode ends at timestep T.
        q1_reg = (((1 - dones) * (q1_best - targ_q1_best)) ** 2).sum()
        q2_reg = (((1 - dones) * (q2_best - targ_q2_best)) ** 2).sum()

        return q1_loss, q2_loss, q1_reg, q2_reg

    def _compute_acts_targets(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths):
        with torch.no_grad():
            rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
            _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

            # Compute next actions with policy and CEM
            next_acts_pi, next_acts_pi_mean, next_acts_pi_var, _, _ = self.model.act_stats(next_obss, next_h_states)

            # NOTE: It is important to clip this action. Otherwise the Q-function gets OOD data
            next_acts_pi = torch.clamp(next_acts_pi, min=self._min_action[0], max=self._max_action[0])
            
            best_next_acts = self.cem.compute_action(self.construct_q_function(q_i=1),
                                                     next_obss,
                                                     next_h_states,
                                                     next_acts_pi_mean,
                                                     next_acts_pi_var,
                                                     lengths=None)

            # Get best actions and best Q values
            min_q_targs_pi, _, _, _ = self.model.q_vals(next_obss, next_h_states, next_acts_pi)
            min_q_targs_cem, _, _, _ = self.model.q_vals(next_obss, next_h_states, best_next_acts)

            best_q_targs = torch.max(min_q_targs_pi, min_q_targs_cem)
            target = rews + (self._gamma ** discounting) * (1 - dones) * best_q_targs

            replace_idxes = (min_q_targs_pi > min_q_targs_cem).squeeze()
            best_next_acts[replace_idxes] = next_acts_pi[replace_idxes]

            _, q1_best, q2_best, _ = self.model.q_vals(next_obss, next_h_states, best_next_acts)

        return best_next_acts.cpu().detach(), target.cpu().detach(), q1_best.cpu().detach(), q2_best.cpu().detach(), (q2_val - q1_val).sum().cpu().detach(), q1_best.max().cpu().detach(), q2_best.max().cpu().detach()

    def _compute_pi_loss(self, obss, h_states, acts, lengths):
        acts_pi, acts_pi_mean, acts_pi_var, entropies, v_pi = self.model.act_stats(obss, h_states, lengths=lengths)
        _, q1_pi, _, _ = self.model.q_vals(obss, h_states, acts_pi, lengths=lengths)
        acts_cem = self.cem.compute_action(self.construct_q_function(q_i=0),
                                           obss,
                                           h_states,
                                           acts_pi_mean,
                                           acts_pi_var,
                                           lengths=lengths)
        with torch.no_grad():
            _, q1_cem, _, _ = self.model.q_vals(obss, h_states, acts_cem, lengths=lengths)
            score = q1_cem - v_pi
            score = torch.clamp(score.detach(), min=0.)

        acts_cem_lprob = self.model.lprob(obss, h_states, acts_cem, lengths=lengths)
        cem_loss = (score * acts_cem_lprob).sum() / self.action_dim

        q_loss = q1_pi.sum()
        pi_loss = -(q_loss + cem_loss)
        
        return pi_loss, acts_cem_lprob.max().detach().cpu(), acts_cem_lprob.min().detach().cpu()

    def update_qs(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        init_qs_loss = None

        best_next_acts = []
        targets = []
        q1_bests = []
        q2_bests = []
        total_qs_descrepancy = 0.
        total_q2_val = 0.
        max_q1 = -np.inf
        max_q2 = -np.inf
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                            batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            best_next_act, target, q1_best, q2_best, qs_descrepancy, q1_max, q2_max = self._compute_acts_targets(obss[opt_idxes],
                                                                                                                 h_states[opt_idxes],
                                                                                                                 acts[opt_idxes],
                                                                                                                 rews[opt_idxes],
                                                                                                                 dones[opt_idxes],
                                                                                                                 next_obss[opt_idxes],
                                                                                                                 discounting[opt_idxes],
                                                                                                                 lengths[opt_idxes])
            best_next_acts.append(best_next_act)
            targets.append(target)
            q1_bests.append(q1_best)
            q2_bests.append(q2_best)
            total_qs_descrepancy += qs_descrepancy
            max_q1 = max(q1_max, max_q1)
            max_q2 = max(q2_max, max_q2)

        best_next_acts = torch.cat(best_next_acts, dim=0)
        targets = torch.cat(targets, dim=0)
        q1_bests = torch.cat(q1_bests, dim=0)
        q2_bests = torch.cat(q2_bests, dim=0)

        q1_losses = []
        q2_losses = []
        total_update_time = 0.
        q1_regs = []
        q2_regs = []

        for update_i in range(self._num_q_updates):
            tic = timeit.default_timer()
            self.qs_opt.zero_grad()
            total_q1_loss = 0.
            total_q2_loss = 0.
            total_q1_reg = 0.
            total_q2_reg = 0.
            for grad_i in range(self._accum_num_grad):
                q1_loss, q2_loss, q1_reg, q2_reg = self._compute_qs_loss(obss[opt_idxes],
                                                                         h_states[opt_idxes],
                                                                         acts[opt_idxes],
                                                                         dones[opt_idxes],
                                                                         best_next_acts[opt_idxes],
                                                                         target[opt_idxes],
                                                                         q1_best[opt_idxes],
                                                                         q2_best[opt_idxes],
                                                                         next_obss[opt_idxes],
                                                                         lengths[opt_idxes])
                q1_loss /= self._batch_size
                q2_loss /= self._batch_size
                q1_reg /= self._batch_size
                q2_reg /= self._batch_size
                qs_loss = q1_loss + q2_loss + q1_reg + q2_reg

                total_q1_loss += q1_loss.detach().cpu()
                total_q2_loss += q2_loss.detach().cpu()
                total_q1_reg += q1_reg.detach().cpu()
                total_q2_reg += q2_reg.detach().cpu()
                qs_loss.backward()

            nn.utils.clip_grad_norm_(self.model.qs_parameters,
                                    self._max_grad_norm)
            self.qs_opt.step()
            total_update_time += timeit.default_timer() - tic
            q1_losses.append(total_q1_loss.numpy())
            q2_losses.append(total_q2_loss.numpy())
            q1_regs.append(total_q1_reg.numpy())
            q2_regs.append(total_q2_reg.numpy())

            if init_qs_loss is None:
                init_qs_loss = qs_loss.detach()

            # This seems to be a hack for not overfitting?
            if qs_loss.detach() < init_qs_loss * self._alpha:
                break

        update_info[c.Q1_MAX].append(max_q1)
        update_info[c.Q2_MAX].append(max_q2)
        update_info[c.Q_UPDATE_TIME].append(total_update_time)
        update_info[c.Q1_LOSS].append(np.mean(q1_losses))
        update_info[c.Q2_LOSS].append(np.mean(q2_losses))
        update_info[c.Q1_REG].append(np.mean(q1_regs))
        update_info[c.Q2_REG].append(np.mean(q2_regs))
        update_info[c.AVG_Q_DISCREPANCY].append(qs_descrepancy / self._batch_size)


    def update_policy(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.policy_opt.zero_grad()
        total_pi_loss = 0.
        max_lprob = -np.inf
        min_lprob = np.inf
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            pi_loss, lprob_max, lprob_min = self._compute_pi_loss(obss[opt_idxes],
                                            h_states[opt_idxes],
                                            acts[opt_idxes],
                                            lengths[opt_idxes])
            max_lprob = max(lprob_max, max_lprob)
            min_lprob = min(lprob_min, min_lprob)
            pi_loss /= self._batch_size
            total_pi_loss += pi_loss.detach().cpu()
            pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.policy_parameters,
                                 self._max_grad_norm)
        self.policy_opt.step()
        update_info[c.LPROB_MAX].append(max_lprob)
        update_info[c.LPROB_MIN].append(min_lprob)
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.PI_LOSS].append(total_pi_loss.numpy())

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
            update_info[c.Q1_REG] = []
            update_info[c.Q2_REG] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.Q_UPDATE_TIME] = []
            update_info[c.POLICY_UPDATE_TIME] = []
            update_info[c.AVG_Q1_VAL] = []
            update_info[c.AVG_Q2_VAL] = []
            update_info[c.AVG_Q_DISCREPANCY] = []
            update_info[c.LPROB_MAX] = []
            update_info[c.LPROB_MIN] = []
            update_info[c.Q1_MAX] = []
            update_info[c.Q2_MAX] = []

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths = self.buffer.sample_with_next_obs(
                    self._batch_size * self._num_prefetch, next_obs, next_h_state)
                
                obss = self.train_preprocessing(obss)
                next_obss = self.train_preprocessing(next_obss)
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
                                   discounting,
                                   infos,
                                   lengths,
                                   update_info)
                    self._aux_tasks.step()
                    update_info.update(aux_update_info)

                    # Update policy
                    self.update_policy(batch_start_idx,
                                       obss,
                                       h_states,
                                       acts,
                                       rews,
                                       dones,
                                       next_obss,
                                       next_h_states,
                                       discounting,
                                       infos,
                                       lengths,
                                       update_info)

            if hasattr(self.model, c.VALUE_RMS):
                update_info[f"{c.VALUE_RMS}/{c.MEAN}"] = self.model.value_rms.mean.numpy()
                update_info[f"{c.VALUE_RMS}/{c.VARIANCE}"] = self.model.value_rms.var.numpy()
            return True, update_info
        return False, update_info
