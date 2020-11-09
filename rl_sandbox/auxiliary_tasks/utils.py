import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTasks
from rl_sandbox.auxiliary_tasks.koopman import Koopman, KoopmanDynamics
from rl_sandbox.model_architectures.utils import make_model, make_optimizer


def make_auxiliary_tasks(tasks, model, buffer, cfg):
    aux_tasks = dict()
    if tasks is not None:
        for task_name, task_setting in tasks.items():
            assert task_name not in aux_tasks
            if task_name == c.KOOPMAN:
                task_setting[c.MODEL_SETTING][c.KWARGS][c.LAYERS_DIM] = model.encoder.layers_dim
                decoder = make_model(task_setting[c.MODEL_SETTING]).to(task_setting[c.DEVICE])
                dynamics = KoopmanDynamics(z_dim=task_setting[c.Z_DIM],
                                           u_dim=task_setting[c.U_DIM],
                                           device=task_setting[c.DEVICE])
                aux_opt = make_optimizer(list(decoder.parameters()) + list(dynamics.parameters()), task_setting[c.OPTIMIZER_SETTING])

                aux_tasks[c.KOOPMAN] = Koopman(rec_dim=task_setting[c.REC_DIM],
                                               batch_size=task_setting[c.BATCH_SIZE],
                                               decoder=decoder,
                                               encoder=model.encoder,
                                               dynamics=dynamics,
                                               opt=aux_opt,
                                               buffer=buffer,
                                               algo_params=cfg,
                                               reduction=task_setting[c.REDUCTION],
                                               loss_coef=task_setting[c.LOSS_COEF],
                                               device=task_setting[c.DEVICE])
            else:
                raise NotImplementedError
    return AuxiliaryTasks(aux_tasks)
