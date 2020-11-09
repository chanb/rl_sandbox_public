import timeit
import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.algorithms.utils import aug_data
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

class UpdateSACIntentions(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)
        self._num_tasks = algo_params.get(c.NUM_TASKS, 1)
        self._action_dim = algo_params[c.ACTION_DIM]

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)
        rews = rews.reshape(task_batch_size, 1) * self._reward_scaling
        dones = dones.repeat(1, self._num_tasks).reshape(task_batch_size, 1)
        discounting = discounting.repeat(1, self._num_tasks).reshape(task_batch_size, 1)

        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)

            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)
            next_acts = next_acts.reshape(task_batch_size, self._action_dim)
            next_lprobs = next_lprobs.reshape(task_batch_size, 1)

            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)

            min_q_targ, _, _, _ = self._target_model.q_vals(
                aug_data(data=next_obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                aug_data(data=targ_next_h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                next_acts)
            min_q_targ = torch.gather(min_q_targ, dim=1, index=tasks)
            min_q_targ = min_q_targ.detach()

            v_next = (min_q_targ - alpha * next_lprobs)

            if hasattr(self.model, c.VALUE_RMS):
                v_next = v_next.reshape(batch_size, self._num_tasks).cpu()
                v_next = self.model.value_rms.unnormalize(v_next)
                v_next = v_next.to(self.device).reshape(task_batch_size, 1)

            target = rews + (self._gamma ** discounting) * (1 - dones) * v_next

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu().reshape(batch_size, self._num_tasks)
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target)
                target = target.to(self.device).reshape(task_batch_size, 1)

        q1_loss = ((q1_val.reshape(task_batch_size, 1) - target) ** 2).sum()
        q2_loss = ((q2_val.reshape(task_batch_size, 1) - target) ** 2).sum()

        return q1_loss, q2_loss

    def _compute_pi_loss(self, obss, h_states, acts, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)

        acts, lprobs = self.model.act_lprob(obss, h_states)
        acts = acts.reshape(task_batch_size, self._action_dim)
        lprobs = lprobs.reshape(task_batch_size, 1)

        min_q, _, _, _ = self.model.q_vals(
            aug_data(data=obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            aug_data(data=h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            acts,
            lengths=lengths.repeat(1, self._num_tasks).reshape(task_batch_size))
        min_q = torch.gather(min_q, dim=1, index=tasks)

        with torch.no_grad():
            alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)
        pi_loss = (alpha * lprobs - min_q).sum()
        
        return pi_loss

    def _compute_alpha_loss(self, obss, h_states, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks
        with torch.no_grad():
            _, lprobs = self.model.act_lprob(obss, h_states, lengths=lengths)
            lprobs = lprobs.reshape(task_batch_size, 1)
        
        alpha = self.model.alpha.repeat(batch_size).reshape(task_batch_size, 1)
        alpha_loss = (-alpha * (lprobs + self._target_entropy).detach()).sum()

        return alpha_loss
