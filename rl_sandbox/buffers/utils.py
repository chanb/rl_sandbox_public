import numpy as np

import rl_sandbox.constants as c

from rl_sandbox.buffers.disk_buffer import DiskNumPyBuffer
from rl_sandbox.buffers.ram_buffer import NumPyBuffer, NextStateNumPyBuffer


class SumTree:
    def __init__(self, capacity, rng=np.random):
        """ Sum tree data structure for prioritized experience replay.
        Reference: https://github.com/rlcode/per/blob/master/SumTree.py

        Args:
        - capacity: The amount of data the tree can store.
        """
        self._capacity = capacity
        self.tree_levels = int(np.ceil(np.log2(capacity + 1)) + 1)
        self.tree = np.zeros(shape=(2 ** self.tree_levels - 1, ))
        self.shift = len(self.tree) - 2 ** (self.tree_levels - 1)
        self.rng = rng
        self._count = 0
        self._pointer = 0

    def _propagate(self, tree_idxes, deltas):
        for _ in range(1, self.tree_levels):
            tree_idxes = (tree_idxes - 1) // 2
            np.add.at(self.tree, tree_idxes, deltas)

    def _retrieve(self, random_values):
        random_values = self.total_value * random_values
        tree_idxes = np.zeros(len(random_values), dtype=np.int64)
        for _ in range(self.tree_levels - 1):
            tree_idxes = 2 * tree_idxes + 1
            left_values = self.tree[tree_idxes]
            where_right = np.where(random_values > left_values)[0]
            tree_idxes[where_right] += 1
            random_values[where_right] -= left_values[where_right]
        return tree_idxes

    @property
    def total_value(self):
        return self.tree[0]

    def add(self, sum_value):
        tree_idx = self._pointer + self.shift

        self.update(np.array([tree_idx]), np.array([sum_value]))

        self._pointer = (self._pointer + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def update(self, idxes, sum_values):
        deltas = sum_values - self.tree[idxes]
        self.tree[idxes] = sum_values
        self._propagate(idxes, deltas)

    def sample(self, batch_size):
        random_values = self.rng.uniform(size=batch_size)
        tree_idx = self._retrieve(random_values)
        return tree_idx, self.tree[tree_idx]


def make_buffer(buffer_cfg, seed=None, load_buffer=False):
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    buffer_cfg[c.KWARGS][c.RNG] = np.random.RandomState(seed)

    if buffer_cfg[c.STORAGE_TYPE] == c.DISK:
        buffer = DiskNumPyBuffer(**buffer_cfg[c.KWARGS])
    elif buffer_cfg[c.STORAGE_TYPE] == c.RAM:
        if buffer_cfg.get(c.STORE_NEXT_OBSERVATION, False):
            buffer = NextStateNumPyBuffer(**buffer_cfg[c.KWARGS])
        else:
            buffer = NumPyBuffer(**buffer_cfg[c.KWARGS])
    else:
        raise NotImplementedError

    for wrapper_config in buffer_cfg[c.BUFFER_WRAPPERS]:
        buffer = wrapper_config[c.WRAPPER](buffer, **wrapper_config[c.KWARGS])

    if load_buffer:
        buffer.load(load_buffer)

    return buffer
