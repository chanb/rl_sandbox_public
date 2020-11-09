import numpy as np
import os
import torch

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
                 history_length=0,
                 checkpoint_interval=0,
                 checkpoint_path=None,
                 rng=np.random,
                 dtype=np.float32):
        self.rng = rng
        self._memory_size = memory_size
        self._dtype = dtype
        os.makedirs(f"{disk_dir}", exist_ok=True)
        self.observations = np.memmap(filename=f"{disk_dir}/observations.npy", mode="w+", shape=(memory_size, *obs_dim), dtype=dtype)
        self.hidden_states = np.memmap(filename=f"{disk_dir}/hidden_states.npy", mode="w+", shape=(memory_size, *h_state_dim), dtype=dtype)
        self.actions = np.memmap(filename=f"{disk_dir}/actions.npy", mode="w+", shape=(memory_size, *action_dim), dtype=dtype)
        self.rewards = np.memmap(filename=f"{disk_dir}/rewards.npy", mode="w+", shape=(memory_size, *reward_dim), dtype=dtype)
        self.dones = np.memmap(filename=f"{disk_dir}/dones.npy", mode="w+", shape=(memory_size, 1), dtype=np.bool)
        self.infos = dict()
        for info_name, (info_shape, info_dtype) in infos.items():
            self.infos[info_name] = np.memmap(filename=f"{disk_dir}/{info_name}.npy", mode="w+", shape=(memory_size, *info_shape), dtype=info_dtype)

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
        self.history_length = history_length
        self.history_frame = np.zeros(shape=(history_length, *obs_dim), dtype=dtype)

    def save(self, save_path, end_with_done=True):
        pass
