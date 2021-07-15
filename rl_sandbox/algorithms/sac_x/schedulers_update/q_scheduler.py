import timeit
import torch

import rl_sandbox.constants as c


class UpdateQScheduler:
    def __init__(self, model, algo_params):
        self.model = model
        self._num_tasks = algo_params.get(c.NUM_TASKS, 1)
        self._action_dim = algo_params[c.ACTION_DIM]

        self._scheduler_period = algo_params[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD]
        self._scheduler_tau = algo_params[c.SCHEDULER_TAU]
        self.main_intention = algo_params.get(c.MAIN_INTENTION, 0)
        
        self._gamma = algo_params[c.GAMMA]
        self._rewards = []
        self._discounting = []

    def state_dict(self):
        return self.model.state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _compute_returns(self):
        episode_length = len(self._rewards)
        returns = torch.zeros(episode_length + 1)
        for step in range(episode_length - 1, -1, -1):
            returns[step] = self._rewards[step] + \
                (self._gamma ** self._discounting[step]) * returns[step + 1]

        # Only take the returns for every scheduler's action
        return returns[:-1][::self._scheduler_period]

    def update_scheduler(self, obs, act, update_info):
        traj = obs + [act.item()]

        print(f"Scheduler Trajectory: {traj} - Q([], a), for all a: {self.model.compute_qs([])}")

        tic = timeit.default_timer()
        update_info[c.Q_UPDATE_TIME] = []
        rets = self._compute_returns()
        for step in range(len(traj)):
            old_q_value = self.model.compute_qsa(traj[:step], traj[step])
            new_q_value = old_q_value * (1 - self._scheduler_tau) + rets[step] * self._scheduler_tau
            self.model.update_qsa(traj[:step], traj[step], new_q_value)
        update_info[c.Q_UPDATE_TIME].append(timeit.default_timer() - tic)

    def update(self, obs, act, reward, done, info):
        self._rewards.append(reward[self.main_intention].item())
        self._discounting.append(info[c.DISCOUNTING][0].item())

        update_info = dict()
        if done:
            obs = info[c.HIGH_LEVEL_OBSERVATION]
            act = info[c.HIGH_LEVEL_ACTION]
            self.update_scheduler(obs, act, update_info)
            self._rewards.clear()
            self._discounting.clear()
            return True, update_info
        return False, update_info
