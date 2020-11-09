import timeit
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class SACDAC(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

    def update_policy(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.policy_opt.zero_grad()
        total_pi_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            non_absorbing_idx = torch.where(obss[opt_idxes, -1, -1] == 0)[0]
            pi_loss = self._compute_pi_loss(obss[opt_idxes][non_absorbing_idx],
                                            h_states[opt_idxes][non_absorbing_idx],
                                            acts[opt_idxes][non_absorbing_idx],
                                            lengths[opt_idxes][non_absorbing_idx])
            pi_loss /= len(non_absorbing_idx)
            total_pi_loss += pi_loss.detach().cpu()
            pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.policy_parameters,
                                self._max_grad_norm)
        self.policy_opt.step()
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.PI_LOSS].append(total_pi_loss.numpy())

    def update_alpha(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.alpha_opt.zero_grad()
        total_alpha_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            non_absorbing_idx = torch.where(obss[opt_idxes, -1, -1] == 0)[0]
            alpha_loss = self._compute_alpha_loss(obss[opt_idxes][non_absorbing_idx],
                                                  h_states[opt_idxes][non_absorbing_idx],
                                                  lengths[opt_idxes][non_absorbing_idx])
            alpha_loss /= len(non_absorbing_idx)
            total_alpha_loss += alpha_loss.detach().cpu()
            alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.model.log_alpha,
                                self._max_grad_norm)
        self.alpha_opt.step()
        update_info[c.ALPHA_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.ALPHA_LOSS].append(total_alpha_loss.numpy())

    def update(self, reward_function, next_obs, next_h_state):
        self.step += 1

        update_info = {}

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
            update_info[c.DISCRIMINATOR_REWARD] = []

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths = self.buffer.sample_with_next_obs(
                    self._batch_size * self._num_prefetch, next_obs, next_h_state)                
                
                obss = self.train_preprocessing(obss)
                next_obss = self.train_preprocessing(next_obss)

                if hasattr(self.model, c.OBS_RMS):
                    self.model.obs_rms.update(obss)

                # NOTE: This computes the new reward using the provided reward function (discriminator)
                with torch.no_grad():
                    idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
                    last_obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]
                    rews = reward_function(last_obss, acts).detach()
                    update_info[c.DISCRIMINATOR_REWARD].append(rews.cpu().numpy())
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
