import _pickle as pickle
import gzip
import numpy as np
import os

from rl_sandbox.buffers.ram_buffer import NumPyBuffer
from rl_sandbox.constants import OBSERVATIONS


obs_dim = (1,)
h_state_dim = (1,)
action_dim = (1,)
reward_dim = (1,)
checkpoint_interval = 3

def generate_transition():
    obs = np.random.uniform(size=obs_dim).astype(np.float32)
    h_state = np.random.uniform(size=h_state_dim).astype(np.float32)
    action = np.random.uniform(size=action_dim).astype(np.float32)
    reward = np.random.uniform(size=reward_dim).astype(np.float32)
    done = np.random.randint(0, 2, size=(1,))
    info = dict()
    return (obs, h_state, action, reward, done, info)

for circular in range(2):
    print(f"CIRCULAR: {circular} ====================")
    buffer = NumPyBuffer(
        memory_size=5,
        obs_dim=obs_dim,
        h_state_dim=h_state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        circular=circular,
        rng=np.random.RandomState(1),
        checkpoint_interval=checkpoint_interval,
        checkpoint_path='./',
    )

    for ii in range(1, 13):
        transition = generate_transition()

        print(f"Iteration {ii} -----------")
        if not circular and ii % 6 == 0:
            print("REMOVE -----")
            print(np.concatenate((buffer.observations, np.expand_dims(buffer._checkpoint_idxes, axis=1)), axis=1))
            buffer.remove(4)
            print(np.concatenate((buffer.observations, np.expand_dims(buffer._checkpoint_idxes, axis=1)), axis=1))
            print("REMOVE_END -----")

        buffer.push(*transition)
        print(np.concatenate((buffer.observations, np.expand_dims(buffer._checkpoint_idxes, axis=1)), axis=1))

        if (not circular and ii in (3, 7, 10)) or (circular and ii % 6 == 0):
            ckpt_file = str(ii // checkpoint_interval - 1) + '.pkl'

            with gzip.open(ckpt_file, 'rb') as f:
                obss = pickle.load(f)[OBSERVATIONS]
                print(obss)
