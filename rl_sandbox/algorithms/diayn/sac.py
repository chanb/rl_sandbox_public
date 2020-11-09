import timeit
import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask


class SACDIAYN(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths):
        dones, discounting = dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            # We use pseudo-reward rather than the environment reward
            idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
            last_obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]
            tasks = last_obss[:, -self.diayn_task_dim:]
            rews = self.diayn_discriminator.lprob(last_obss[:, :-self.diayn_task_dim], tasks) - self.diayn_prior.lprob(tasks)

            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)
            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)
            min_q_targ, _, _, _ = self._target_model.q_vals(next_obss, targ_next_h_states, next_acts)
            min_q_targ = min_q_targ.detach()
            v_next = (min_q_targ - self.model.alpha.detach() * next_lprobs)

            if hasattr(self.model, c.VALUE_RMS):
                v_next = self.model.value_rms.unnormalize(v_next.cpu()).to(self.device)

            target = rews + (self._gamma ** discounting) * (1 - dones) * v_next

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu()
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target).to(self.device)

        q1_loss = ((q1_val - target) ** 2).sum()
        q2_loss = ((q2_val - target) ** 2).sum()

        return q1_loss, q2_loss