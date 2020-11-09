import timeit

import rl_sandbox.constants as c


class SACX:
    def __init__(self, update_scheduler, update_intentions, algo_params):
        self.update_scheduler = update_scheduler
        self.update_intentions = update_intentions
        self.algo_params = algo_params
        self.buffer = update_intentions.buffer
        self.step = 0

    def state_dict(self):
        state_dict = {
            c.SCHEDULER: self.update_scheduler.state_dict(),
            c.INTENTIONS: self.update_intentions.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.update_scheduler.load_state_dict(state_dict[c.SCHEDULER])
        self.update_intentions.load_state_dict(state_dict[c.INTENTIONS])

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1

        update_info = {}

        # Intention Learning
        # NOTE: intention learning stores the current transition to the buffer
        tic = timeit.default_timer()
        updated_intentions, intentions_info = self.update_intentions.update(curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state)
        toc = timeit.default_timer()
        if updated_intentions:
            update_info[c.INTENTIONS_UPDATE_TIME] = toc - tic
            update_info.update(intentions_info)

        # Scheduler Learning
        tic = timeit.default_timer()
        updated_scheduler, scheduler_info = self.update_scheduler.update(curr_obs, act, rew, done, info)
        toc = timeit.default_timer()
        if updated_scheduler:
            update_info[c.SCHEDULER_UPDATE_TIME] = toc - tic
            update_info.update(scheduler_info)

        return updated_intentions, update_info

