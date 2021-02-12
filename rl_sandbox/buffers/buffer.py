from collections import namedtuple

import rl_sandbox.constants as c


class NoSampleError(Exception):
    pass


class LengthMismatchError(Exception):
    pass


class CheckpointIndexError(Exception):
    pass


class Buffer:
    @property
    def memory_size(self):
        raise NotImplementedError

    @property
    def is_full(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def push(self, obs, h_state, act, rew, done, info, **kwargs):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def sample(self, batch_size, idxes=None):
        raise NotImplementedError

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None):
        raise NotImplementedError

    def sample_consecutive(self, batch_size, end_with_done=False):
        raise NotImplementedError

    def save(self, save_path, **kwargs):
        raise NotImplementedError

    def load(self, load_path):
        raise NotImplementedError

    def close(self):
        pass
