import numpy as np
import timeit
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class SACPER(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths, tree_idxes, is_weights):
        rews, dones, discounting, is_weights = rews.to(self.device), dones.to(self.device), discounting.to(self.device), is_weights.to(self.device)
        min_q, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)
            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)
            min_q_targ, _, _, _ = self._target_model.q_vals(next_obss, targ_next_h_states, next_acts)
            min_q_targ = min_q_targ.detach()

            if hasattr(self.model, c.VALUE_RMS):
                min_q_targ = self.model.value_rms.unnormalize(min_q_targ.cpu()).to(self.device)

            v_next = (min_q_targ - self.model.alpha.detach() * next_lprobs)

            target = rews + (self._gamma ** discounting) * (1 - dones) * v_next

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu()
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target).to(self.device)

            # Take the mean of TD errors
            new_priorities = ((min_q.detach() - target) ** 2 * is_weights).cpu().numpy()
            tree_idxes = tree_idxes.numpy()
            idx_arr, unique_idxes = np.unique(tree_idxes, return_index=True)
            one_hot = (tree_idxes.reshape(-1, 1) == idx_arr.reshape(1, -1))
            mean_new_priorities = one_hot.T @ new_priorities / one_hot.sum(axis=0).reshape(-1, 1)
            self.buffer.update_priorities(tree_idxes[unique_idxes], mean_new_priorities.reshape(-1))
        
        q1_loss = ((q1_val - target) ** 2 * is_weights).sum()
        q2_loss = ((q2_val - target) ** 2 * is_weights).sum()

        return q1_loss, q2_loss

    def update_qs(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info):
        tic = timeit.default_timer()
        self.qs_opt.zero_grad()
        total_q1_loss = 0.
        total_q2_loss = 0.

        tree_idxes = infos[c.TREE_IDX]
        is_weights = infos[c.IS_WEIGHT].unsqueeze(-1)

        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * self._num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * self._num_samples_per_accum)
            q1_loss, q2_loss = self._compute_qs_loss(obss[opt_idxes],
                                                     h_states[opt_idxes],
                                                     acts[opt_idxes],
                                                     rews[opt_idxes],
                                                     dones[opt_idxes],
                                                     next_obss[opt_idxes],
                                                     discounting[opt_idxes],
                                                     lengths[opt_idxes],
                                                     tree_idxes[opt_idxes],
                                                     is_weights[opt_idxes])
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
