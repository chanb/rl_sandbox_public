import _pickle as pickle
import numpy as np
import os
import timeit
import torch

from collections import namedtuple
from functools import partial
from pprint import pprint

import rl_sandbox.constants as c

from rl_sandbox.envs.fake_env import FakeEnv
from rl_sandbox.utils import DummySummaryWriter, EpochSummary

def buffer_warmup(agent,
                  env,
                  buffer,
                  buffer_preprocess,
                  transition_preprocess,
                  experiment_settings,
                  render=False):
    clip_action = experiment_settings.get(
        c.CLIP_ACTION, c.DEFAULT_TRAIN_PARAMS[c.CLIP_ACTION])
    min_action = experiment_settings.get(c.MIN_ACTION, None)
    max_action = experiment_settings.get(c.MAX_ACTION, None)
    num_steps = experiment_settings.get(c.NUM_STEPS, 0)
    num_episodes = experiment_settings.get(c.NUM_EPISODES, 0)

    buffer_preprocess.reset()
    curr_obs = env.reset()
    curr_obs = buffer_preprocess(curr_obs)
    curr_h_state = agent.reset()
    curr_step = 0
    curr_episode = 0
    while True:
        if hasattr(env, c.RENDER) and render:
            env.render()
        action, next_h_state, act_info = agent.compute_action(
            obs=curr_obs, hidden_state=curr_h_state)

        env_action = action
        if clip_action:
            env_action = np.clip(
                action, a_min=min_action, a_max=max_action)

        next_obs, reward, done, env_info = env.step(env_action)
        next_obs = buffer_preprocess(next_obs)

        info = dict()
        info[c.DISCOUNTING] = env_info.get(c.DISCOUNTING, 1)
        info.update(act_info)
        
        buffer.push(**transition_preprocess(curr_obs,
                                            curr_h_state,
                                            action,
                                            reward,
                                            done,
                                            info,
                                            next_obs=next_obs,
                                            next_h_state=next_h_state))
        curr_obs = next_obs
        curr_h_state = next_h_state
        curr_step += 1

        if curr_step >= num_steps:
            break

        if done:
            buffer_preprocess.reset()
            curr_obs = env.reset()
            curr_obs = buffer_preprocess(curr_obs)
            curr_h_state = agent.reset()
            curr_episode += 1

            if curr_episode >= num_episodes:
                break

def train(agent,
          evaluation_agent,
          train_env,
          evaluation_env,
          buffer_preprocess,
          experiment_settings,
          auxiliary_reward=lambda reward, **kwargs: np.array([reward]),
          summary_writer=DummySummaryWriter(),
          save_path=None):
    # Training Setting
    clip_action = experiment_settings.get(
        c.CLIP_ACTION, c.DEFAULT_TRAIN_PARAMS[c.CLIP_ACTION])
    min_action = experiment_settings.get(c.MIN_ACTION, None)
    max_action = experiment_settings.get(c.MAX_ACTION, None)
    max_total_steps = experiment_settings.get(
        c.MAX_TOTAL_STEPS, c.DEFAULT_TRAIN_PARAMS[c.MAX_TOTAL_STEPS])

    # Progress Tracking
    curr_episode = experiment_settings.get(
        c.CURR_EPISODE, c.DEFAULT_TRAIN_PARAMS[c.CURR_EPISODE])
    num_updates = experiment_settings.get(
        c.NUM_UPDATES, c.DEFAULT_TRAIN_PARAMS[c.NUM_UPDATES])
    returns = experiment_settings.get(
        c.RETURNS, c.DEFAULT_TRAIN_PARAMS[c.RETURNS])
    cum_episode_lengths = experiment_settings.get(
        c.CUM_EPISODE_LENGTHS, c.DEFAULT_TRAIN_PARAMS[c.CUM_EPISODE_LENGTHS])

    # Logging
    print_interval = experiment_settings.get(
        c.PRINT_INTERVAL, c.DEFAULT_TRAIN_PARAMS[c.PRINT_INTERVAL])
    save_interval = experiment_settings.get(
        c.SAVE_INTERVAL, c.DEFAULT_TRAIN_PARAMS[c.SAVE_INTERVAL])
    log_interval = experiment_settings.get(
        c.LOG_INTERVAL, c.DEFAULT_TRAIN_PARAMS[c.LOG_INTERVAL])
    train_render = experiment_settings.get(
        c.TRAIN_RENDER, c.DEFAULT_TRAIN_PARAMS[c.TRAIN_RENDER])

    # Evaluation
    evaluation_frequency = experiment_settings.get(
        c.EVALUATION_FREQUENCY, c.DEFAULT_TRAIN_PARAMS[c.EVALUATION_FREQUENCY])
    evaluation_returns = experiment_settings.get(
        c.EVALUATION_RETURNS, c.DEFAULT_TRAIN_PARAMS[c.EVALUATION_RETURNS])
    num_evaluation_episodes = experiment_settings.get(
        c.NUM_EVALUATION_EPISODES, c.DEFAULT_TRAIN_PARAMS[c.NUM_EVALUATION_EPISODES])
    evaluation_render = experiment_settings.get(
        c.EVALUATION_RENDER, c.DEFAULT_TRAIN_PARAMS[c.EVALUATION_RENDER])

    assert save_path is None or os.path.isdir(save_path)

    num_tasks = experiment_settings.get(c.NUM_TASKS, 1)
    eps_per_task = int(num_evaluation_episodes / num_tasks)
    multitask_returns = np.zeros([num_tasks, eps_per_task])

    eval = partial(evaluate_policy,
                   agent=evaluation_agent,
                   env=evaluation_env,
                   buffer_preprocess=buffer_preprocess,
                   num_episodes=num_evaluation_episodes,
                   clip_action=clip_action,
                   min_action=min_action,
                   max_action=max_action,
                   render=evaluation_render,
                   auxiliary_reward=auxiliary_reward,)

    # TODO: Just a placeholder. We ideally want an agent sampler.
    exploration_strategy = experiment_settings.get(c.EXPLORATION_STRATEGY, None)

    done = False
    if isinstance(train_env, FakeEnv):
        auxiliary_reward = lambda reward, **kwargs: np.array([reward])

    try:
        returns.append(0)
        cum_episode_lengths.append(cum_episode_lengths[-1])
        curr_h_state = agent.reset()
        curr_obs = train_env.reset()
        buffer_preprocess.reset()
        curr_obs = buffer_preprocess(curr_obs)
        tic = timeit.default_timer()

        epoch_summary = EpochSummary()
        epoch_summary.new_epoch()
        for timestep_i in range(cum_episode_lengths[-1], max_total_steps):
            if hasattr(train_env, c.RENDER) and train_render:
                train_env.render()

            action, next_h_state, act_info = agent.compute_action(
                obs=curr_obs, hidden_state=curr_h_state)

            if timestep_i < experiment_settings.get(c.EXPLORATION_STEPS, 0) and exploration_strategy is not None:
                action, _, act_info = exploration_strategy.compute_action(
                    obs=curr_obs, hidden_state=curr_h_state)

            if timestep_i % print_interval == 0:
                pprint(f"Action: {action}")
                pprint(act_info)

            env_action = action
            if clip_action:
                env_action = np.clip(action,
                                     a_min=min_action,
                                     a_max=max_action)

            next_obs, reward, done, env_info = train_env.step(env_action)
            next_obs = buffer_preprocess(next_obs)

            reward = np.atleast_1d(auxiliary_reward(observation=curr_obs,
                                                    action=env_action,
                                                    reward=reward,
                                                    done=done,
                                                    next_observation=next_obs,
                                                    info=env_info))

            info = dict()
            info[c.DISCOUNTING] = env_info.get(c.DISCOUNTING, np.array([1]))
            info.update(act_info)
            updated, update_info = agent.update(curr_obs,
                                                curr_h_state,
                                                action,
                                                reward,
                                                done,
                                                info,
                                                next_obs,
                                                next_h_state)

            curr_obs = next_obs
            curr_h_state = next_h_state

            returns[-1] += reward
            cum_episode_lengths[-1] += 1

            if updated:
                num_updates += 1
                for update_key, update_value in update_info.items():
                    update_value_mean = update_value
                    if isinstance(update_value, (list, tuple, np.ndarray)):
                        if len(update_value) == 0:
                            continue
                        update_value_mean = np.mean(update_value)
                    epoch_summary.log(f"{c.UPDATE_INFO}/{update_key}", update_value, track_min_max=False)

                    # Tensorboard is slow sometimes, use this log interval to gate amount of information
                    if num_updates % log_interval == 0:
                        summary_writer.add_scalar(
                            f"{c.UPDATE_INFO}/{update_key}", update_value_mean, num_updates)
            else:
                for update_key, update_value in update_info.items():
                    epoch_summary.log(f"{c.UPDATE_INFO}/{update_key}", update_value, track_min_max=False)

            if done:
                curr_h_state = agent.reset()
                curr_obs = train_env.reset()
                buffer_preprocess.reset()
                curr_obs = buffer_preprocess(curr_obs)

                # Logging
                episode_length = cum_episode_lengths[-1] if curr_episode == 0 else cum_episode_lengths[-1] - \
                    cum_episode_lengths[-2]
                for task_i, task_i_ret in enumerate(returns[-1]):
                    summary_writer.add_scalar(
                        f"{c.INTERACTION_INFO}/task_{task_i}/{c.RETURNS}", task_i_ret, timestep_i)
                summary_writer.add_scalar(
                    f"{c.INTERACTION_INFO}/{c.EPISODE_LENGTHS}", episode_length, curr_episode)

                epoch_summary.log(f"{c.INTERACTION_INFO}/{c.RETURNS}", returns[-1], axis=0)
                epoch_summary.log(f"{c.INTERACTION_INFO}/{c.EPISODE_LENGTHS}", episode_length)

                returns.append(0)
                cum_episode_lengths.append(cum_episode_lengths[-1])
                curr_episode += 1

            curr_timestep = timestep_i + 1
            if evaluation_frequency > 0 and curr_timestep % evaluation_frequency == 0:
                evaluation_returns.append(eval())
                for task_i, task_i_ret in enumerate(evaluation_returns[-1]):
                    rets_slice = slice(task_i * eps_per_task, task_i * eps_per_task + eps_per_task)
                    task_i_ret = task_i_ret[rets_slice]

                    summary_writer.add_scalar(
                        f"{c.EVALUATION_INFO}/task_{task_i}/{c.AVERAGE_RETURNS}", np.mean(task_i_ret), timestep_i)
                    multitask_returns[task_i] = task_i_ret

                epoch_summary.log(f"{c.EVALUATION_INFO}/{c.AVERAGE_RETURNS}", multitask_returns, axis=(0, 2))

            if curr_timestep % print_interval == 0:
                epoch_summary.print_summary()
                epoch_summary.new_epoch()

            if save_path is not None and curr_timestep % save_interval == 0:
                curr_save_path = f"{save_path}/{timestep_i}.pt"
                print(f"Saving model to {curr_save_path}")
                torch.save(agent.learning_algorithm.state_dict(), curr_save_path)
                pickle.dump({c.RETURNS: returns if done else returns[:-1],
                             c.CUM_EPISODE_LENGTHS: cum_episode_lengths if done else cum_episode_lengths[:-1],
                             c.EVALUATION_RETURNS: evaluation_returns,},
                            open(f'{save_path}/{c.TRAIN_FILE}', 'wb'))
                if hasattr(agent, c.LEARNING_ALGORITHM) and hasattr(agent.learning_algorithm, c.BUFFER):
                    if save_path is not None:
                        agent.learning_algorithm.buffer.save(f"{save_path}/{c.TERMINATION_BUFFER_FILE}")
    finally:
        if save_path is not None:
            torch.save(agent.learning_algorithm.state_dict(),
                       f"{save_path}/{c.TERMINATION_STATE_DICT_FILE}")
            if not done:
                returns = returns[:-1]
                cum_episode_lengths = cum_episode_lengths[:-1]
            pickle.dump(
                {c.RETURNS: returns, c.CUM_EPISODE_LENGTHS: cum_episode_lengths,
                    c.EVALUATION_RETURNS: evaluation_returns},
                open(f'{save_path}/{c.TERMINATION_TRAIN_FILE}', 'wb')
            )
        if hasattr(agent, c.LEARNING_ALGORITHM) and hasattr(agent.learning_algorithm, c.BUFFER):
            if save_path is not None:
                agent.learning_algorithm.buffer.save(f"{save_path}/{c.TERMINATION_BUFFER_FILE}")
            agent.learning_algorithm.buffer.close()
    toc = timeit.default_timer()
    print(f"Training took: {toc - tic}s")


def evaluate_policy(agent,
                    env,
                    buffer_preprocess,
                    num_episodes,
                    clip_action,
                    min_action,
                    max_action,
                    render,
                    auxiliary_reward=lambda reward, **kwargs: np.array([reward]),
                    verbose=False):
    eval_returns = []
    for _ in range(num_episodes):
        eval_returns.append(0)
        curr_obs = env.reset()
        buffer_preprocess.reset()
        curr_obs = buffer_preprocess(curr_obs)
        h_state = agent.reset()
        done = False
        while not done:
            if hasattr(env, c.RENDER) and render:
                env.render()
            action, h_state, act_info = agent.deterministic_action(
                obs=curr_obs, hidden_state=h_state)

            if clip_action:
                action = np.clip(action, a_min=min_action, a_max=max_action)

            next_obs, reward, done, env_info = env.step(action)
            next_obs = buffer_preprocess(next_obs)

            eval_returns[-1] += np.atleast_1d(auxiliary_reward(observation=curr_obs,
                                                               action=action,
                                                               reward=reward,
                                                               done=done,
                                                               next_observation=next_obs,
                                                               info=env_info))
            curr_obs = next_obs
        
        if verbose:
            print(eval_returns[-1])

    return np.array(eval_returns).T
