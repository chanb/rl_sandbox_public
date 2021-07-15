import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.algorithms.sac_x.intentions_update.sac_intentions import UpdateSACIntentions
from rl_sandbox.algorithms.sac_x.schedulers_update.q_scheduler import UpdateQScheduler
from rl_sandbox.algorithms.sac_x.sac_x import SACX
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed


def train_sacx_sac(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.BUFFER_SETTING].get(c.LOAD_BUFFER, False))
    intentions = make_model(experiment_config[c.INTENTIONS_SETTING])

    policy_opt = make_optimizer(intentions.policy_parameters, experiment_config[c.OPTIMIZER_SETTING][c.INTENTIONS])
    qs_opt = make_optimizer(intentions.qs_parameters, experiment_config[c.OPTIMIZER_SETTING][c.QS])
    alpha_opt = make_optimizer([intentions.log_alpha], experiment_config[c.OPTIMIZER_SETTING][c.ALPHA])

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     intentions,
                                     buffer,
                                     experiment_config)

    update_intentions = UpdateSACIntentions(model=intentions,
                                            policy_opt=policy_opt,
                                            qs_opt=qs_opt,
                                            alpha_opt=alpha_opt,
                                            learn_alpha=experiment_config[c.LEARN_ALPHA],
                                            buffer=buffer,
                                            algo_params=experiment_config,
                                            aux_tasks=aux_tasks)

    scheduler = make_model(experiment_config[c.SCHEDULER_SETTING][c.TRAIN])
    update_scheduler = UpdateQScheduler(model=scheduler,
                                        algo_params=experiment_config)

    learning_algorithm = SACX(update_scheduler=update_scheduler,
                              update_intentions=update_intentions,
                              algo_params=experiment_config)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = SACXAgent(scheduler=scheduler,
                      intentions=intentions,
                      learning_algorithm=learning_algorithm,
                      scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD],
                      preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = SACXAgent(scheduler=make_model(experiment_config[c.SCHEDULER_SETTING][c.EVALUATION]),
                                     intentions=intentions,
                                     learning_algorithm=None,
                                     scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.EVALUATION][c.SCHEDULER_PERIOD],
                                     preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.SACX, cfg=experiment_config)

    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          buffer_preprocess=buffer_preprocessing,
          auxiliary_reward=experiment_config[c.AUXILIARY_REWARDS].reward,
          experiment_settings=experiment_config,
          summary_writer=summary_writer,
          save_path=save_path)
