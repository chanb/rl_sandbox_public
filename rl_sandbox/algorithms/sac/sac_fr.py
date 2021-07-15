import timeit
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class SACFunctionalRegularization(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ SAC  Algorithm: https://arxiv.org/abs/1801.01290
        NOTE: We combine the Q1 and Q2 loss and update both Q functions at the same time.

        Functional Regularization on Q-value function: https://arxiv.org/abs/2106.02613
        """
        # The functional regularization constant
        self._kappa = algo_params.get(c.KAPPA, c.DEFAULT_FR_PARAMS[c.KAPPA])
        super().__init__(model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks)


    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths):
        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)
            min_q_next, _, _, _ = self.model.q_vals(next_obss, next_h_states, next_acts)
            min_q_next = min_q_next.detach()

            if hasattr(self.model, c.VALUE_RMS):
                min_q_next = self.model.value_rms.unnormalize(min_q_next.cpu()).to(self.device)

            v_next = (min_q_next - self.model.alpha.detach() * next_lprobs)

            target = rews + (self._gamma ** discounting) * (1 - dones) * v_next

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu()
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target).to(self.device)

            _, targ_q1_val, targ_q2_val, _ = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)

        q1_loss = ((q1_val - target) ** 2).sum()
        q2_loss = ((q2_val - target) ** 2).sum()
        q1_reg = ((q1_val - targ_q1_val) ** 2).sum() * self._kappa
        q2_reg = ((q2_val - targ_q2_val) ** 2).sum() * self._kappa

        return q1_loss, q2_loss, q1_reg, q2_reg

    def update_qs(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.qs_opt.zero_grad()
        total_q1_loss = 0.
        total_q2_loss = 0.
        total_q1_reg = 0.
        total_q2_reg = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            q1_loss, q2_loss, q1_reg, q2_reg = self._compute_qs_loss(obss[opt_idxes],
                                                                     h_states[opt_idxes],
                                                                     acts[opt_idxes],
                                                                     rews[opt_idxes],
                                                                     dones[opt_idxes],
                                                                     next_obss[opt_idxes],
                                                                     discounting[opt_idxes],
                                                                     lengths[opt_idxes])
            q1_loss /= self._batch_size
            q2_loss /= self._batch_size
            q1_reg /= self._batch_size
            q2_reg /= self._batch_size
            qs_loss = q1_loss + q2_loss + q1_reg + q2_reg
            total_q1_loss += q1_loss.detach().cpu()
            total_q2_loss += q2_loss.detach().cpu()
            total_q1_reg += q2_reg.detach().cpu()
            total_q2_reg += q2_reg.detach().cpu()
            qs_loss.backward()

        nn.utils.clip_grad_norm_(self.model.qs_parameters,
                                self._max_grad_norm)
        self.qs_opt.step()
        update_info[c.Q_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.Q1_LOSS].append(total_q1_loss.numpy())
        update_info[c.Q2_LOSS].append(total_q2_loss.numpy())
        update_info[c.Q1_REG].append(total_q1_reg.numpy())
        update_info[c.Q2_REG].append(total_q2_reg.numpy())

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
