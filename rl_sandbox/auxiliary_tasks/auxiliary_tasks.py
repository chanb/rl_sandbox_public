class AuxiliaryTask:
    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        pass

    def compute_loss(self, next_obs, next_h_state):
        return 0., dict()

    def zero_grad(self):
        pass

    def step(self):
        pass
        
        
class AuxiliaryTasks(AuxiliaryTask):
    def __init__(self, aux_tasks):
        super().__init__()
        self._aux_tasks = aux_tasks

    def load_state_dict(self, state_dict):
        for task_name, task_state_dict in state_dict.items():
            assert task_name in self._aux_tasks
            self._aux_tasks[task_name].load_state_dict(task_state_dict)

    def state_dict(self):
        state_dict = dict()
        for task_name, task in self._aux_tasks.items():
            state_dict[task_name] = task.state_dict()
        return state_dict

    def compute_loss(self, next_obs, next_h_state):
        update_info = dict()

        total_loss = 0
        for task_name, task in self._aux_tasks.items():
            loss = task.compute_loss(next_obs, next_h_state)
            update_info[task_name] = loss.detach().cpu()

            total_loss += loss

        return total_loss, update_info

    def zero_grad(self):
        for task in self._aux_tasks.values():
            task.opt.zero_grad()

    def step(self):
        for task in self._aux_tasks.values():
            task.opt.step()
