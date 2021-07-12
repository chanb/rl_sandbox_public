import _pickle as pickle
import gzip
import numpy as np
import os

import rl_sandbox.constants as c

from rl_sandbox.buffers.ram_buffer import NumPyBuffer


class DiskNumPyBuffer(NumPyBuffer):
    def __init__(self,
                 memory_size,
                 obs_dim,
                 h_state_dim,
                 action_dim,
                 reward_dim,
                 infos=dict(),
                 disk_dir="./",
                 burn_in_window=0,
                 padding_first=False,
                 checkpoint_interval=0,
                 checkpoint_path=None,
                 rng=np.random,
                 dtype=np.float32):
        self.rng = rng
        self._memory_size = memory_size
        self._dtype = dtype
        if os.path.isdir(disk_dir):
            mode = "r+"
        else:
            mode = "w+"
            os.makedirs(f"{disk_dir}", exist_ok=True)
        self.observations = np.memmap(filename=f"{disk_dir}/observations.npy", mode=mode, shape=(memory_size, *obs_dim), dtype=dtype)
        print(self.observations.shape)
        self.hidden_states = np.memmap(filename=f"{disk_dir}/hidden_states.npy", mode=mode, shape=(memory_size, *h_state_dim), dtype=dtype)
        self.actions = np.memmap(filename=f"{disk_dir}/actions.npy", mode=mode, shape=(memory_size, *action_dim), dtype=dtype)
        self.rewards = np.memmap(filename=f"{disk_dir}/rewards.npy", mode=mode, shape=(memory_size, *reward_dim), dtype=dtype)
        self.dones = np.memmap(filename=f"{disk_dir}/dones.npy", mode=mode, shape=(memory_size, 1), dtype=np.bool)
        self.infos = dict()
        for info_name, (info_shape, info_dtype) in infos.items():
            self.infos[info_name] = np.memmap(filename=f"{disk_dir}/{info_name}.npy", mode=mode, shape=(memory_size, *info_shape), dtype=info_dtype)

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_idxes = np.ones(shape=memory_size, dtype=np.bool)
        if checkpoint_path is not None and memory_size >= checkpoint_interval > 0:
            self._checkpoint_path = checkpoint_path
            os.makedirs(checkpoint_path, exist_ok=True)
            self.checkpoint = self._checkpoint
            self._checkpoint_count = 0
        else:
            self.checkpoint = lambda: None

        self._pointer = 0
        self._count = 0
        
        # This keeps track of the past X observations and hidden states for RNN
        self.burn_in_window = burn_in_window
        if burn_in_window > 0:
            self.padding_first = padding_first
            self.historic_observations = np.memmap(filename=f"{disk_dir}/historic_observations.npy", mode=mode, shape=(burn_in_window, *obs_dim), dtype=dtype)
            self.historic_hidden_states = np.memmap(filename=f"{disk_dir}/historic_hidden_states.npy", mode=mode, shape=(burn_in_window, *h_state_dim), dtype=dtype)
            self.historic_dones = np.memmap(filename=f"{disk_dir}/historic_dones.npy", mode=mode, shape=(burn_in_window, 1), dtype=np.bool) + 1

    def save(self, save_path, end_with_done=True):
        pointer = self._pointer
        count = self._count

        if end_with_done:
            done_idxes = np.where(self.dones == 1)[0]
            if len(done_idxes) == 0:
                print("No completed episodes. Nothing to save.")
                return

            wraparound_idxes = done_idxes[done_idxes < self._pointer]
            if len(wraparound_idxes) > 0:
                pointer = (wraparound_idxes[-1] + 1) % self._memory_size
                count -= (self._pointer - pointer)
            else:
                pointer = (done_idxes[-1] + 1) % self._memory_size
                count -= (self._pointer + self._memory_size - pointer)

        with gzip.open(save_path, "wb") as f:
            pickle.dump({
                c.MEMORY_SIZE: self._memory_size,
                c.POINTER: pointer,
                c.COUNT: count,
                c.DTYPE: self._dtype,
                c.RNG: self.rng,
            }, f, protocol=-1)

    def load(self, load_path):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        self._memory_size = data[c.MEMORY_SIZE]
        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]
        self._dtype = data[c.DTYPE]
        self.rng = data[c.RNG]