import numpy as np


def go_right_fast(next_observation, **kwargs):
    return int(next_observation[1] > 0.03)

def go_left_fast(next_observation, **kwargs):
    return int(next_observation[1] < -0.03)

class MountainCarContinuousAuxiliaryReward:
    def __init__(self, aux_rewards=[go_left_fast, go_right_fast]):
        self._aux_rewards = aux_rewards

    @property
    def num_auxiliary_rewards(self):
        return len(self._aux_rewards)

    def reward(self,
               observation,
               action,
               reward,
               done,
               next_observation,
               info):
        observation = observation.reshape(-1)
        next_observation = next_observation.reshape(-1)
        reward_vector = [reward]
        for task_reward in self._aux_rewards:
            reward_vector.append(task_reward(observation=observation,
                                             action=action,
                                             reward=reward,
                                             next_observation=next_observation,
                                             done=done,
                                             info=info))
        return np.array(reward_vector, dtype=np.float32)
