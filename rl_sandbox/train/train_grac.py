import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.cem.cem import CEMQ
from rl_sandbox.algorithms.grac.grac import GRAC
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed

def train_grac(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    evaluation_env = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)

    # NOTE: The original implementation actually uses CEM as well for interaction to gather data, but we won't here.
    # experiment_config[c.MODEL_SETTING][c.KWARGS][c.CEM] = CEMQ(cov_noise_init=experiment_config[c.COV_NOISE_INIT],
    #                                                            cov_noise_end=experiment_config[c.COV_NOISE_END],
    #                                                            cov_noise_tau=experiment_config[c.COV_NOISE_TAU],
    #                                                            action_dim=experiment_config[c.ACTION_DIM],
    #                                                            batch_size=1,
    #                                                            num_iters=experiment_config[c.NUM_ITERS],
    #                                                            pop_size=experiment_config[c.POP_SIZE],
    #                                                            elite_size=experiment_config[c.ELITE_SIZE],
    #                                                            device=experiment_config[c.DEVICE],
    #                                                            min_action=experiment_config[c.MIN_ACTION],
    #                                                            max_action=experiment_config[c.MAX_ACTION])

    model = make_model(experiment_config[c.MODEL_SETTING])
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed)

    # policy_opt = make_optimizer(model.policy_parameters, experiment_config[c.OPTIMIZER_SETTING])

    # NOTE: The original implementation also adaptively changes the learning rate, but not in this implementation...
    policy_opt = make_optimizer(model.policy_parameters, {
        c.OPTIMIZER: torch.optim.Adam,
        c.KWARGS: {
            c.LR: 2e-4,
        },
    })
    qs_opt = make_optimizer(model.qs_parameters, experiment_config[c.OPTIMIZER_SETTING])

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     model,
                                     buffer,
                                     experiment_config)

    learning_algorithm = GRAC(model=model,
                              policy_opt=policy_opt,
                              qs_opt=qs_opt,
                              buffer=buffer,
                              algo_params=experiment_config,
                              aux_tasks=aux_tasks)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = ACAgent(model=model,
                    learning_algorithm=learning_algorithm,
                    preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_agent = ACAgent(model=model,
                               learning_algorithm=None,
                               preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.SAC, cfg=experiment_config)
    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          buffer_preprocess=buffer_preprocessing,
          experiment_settings=experiment_config,
          summary_writer=summary_writer,
          save_path=save_path)
