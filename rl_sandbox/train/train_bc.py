import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.bc.bc import BC
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.fake_env import FakeEnv
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed

def train_bc(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    evaluation_env = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
    model = make_model(experiment_config[c.MODEL_SETTING])
    expert_buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed)
    expert_buffer.load(experiment_config[c.EXPERT_BUFFER])
    optimizer = make_optimizer(model.parameters(), experiment_config[c.OPTIMIZER_SETTING][c.POLICY])

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     model,
                                     expert_buffer,
                                     experiment_config)

    learning_algorithm = BC(model=model,
                            optimizer=optimizer,
                            buffer=expert_buffer,
                            algo_params=experiment_config,
                            aux_tasks=aux_tasks)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = ACAgent(model=model,
                    learning_algorithm=learning_algorithm,
                    preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    # NOTE: Apparently deterministic policy works better.
    agent.compute_action = agent.deterministic_action

    evaluation_agent = ACAgent(model=model,
                               learning_algorithm=None,
                               preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.BC, cfg=experiment_config)
    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          buffer_preprocess=buffer_preprocessing,
          experiment_settings=experiment_config,
          summary_writer=summary_writer,
          save_path=save_path)
