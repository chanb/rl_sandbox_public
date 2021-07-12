from rl_sandbox.buffers.buffer import Buffer


class BufferWrapper(Buffer):
    def __init__(self, buffer):
        self.buffer = buffer

    def sample(self, batch_size, idxes=None):
        return self.buffer.sample(batch_size, idxes)

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None):
        return self.buffer.sample_with_next_obs(batch_size, next_obs, next_h_state, idxes)

    def sample_consecutive(self, batch_size, end_with_done=False):
        return self.buffer.sample_consecutive(batch_size, end_with_done)
    
    def sample_init_obs(self, batch_size):
        return self.buffer.sample_init_obs(batch_size)

    @property
    def memory_size(self):
        return self.buffer.memory_size

    @property
    def is_full(self):
        return self.buffer.is_full

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, h_state, act, rew, done, info, **kwargs):
        self.buffer.push(obs, h_state, act, rew, done, info, **kwargs)

    def clear(self):
        return self.buffer.clear()

    def save(self, save_path, **kwargs):
        return self.buffer.save(save_path, **kwargs)

    def load(self, load_path):
        return self.buffer.load(load_path)

    def transfer_data(self, load_path):
        return self.buffer.transfer_data(load_path)

    def close(self):
        return self.buffer.close()

    def __getattr__(self, attr):
        return getattr(self.buffer, attr)
