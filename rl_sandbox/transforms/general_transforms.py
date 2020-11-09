import numpy as np
import torch


class Transform:
    def __call__(self, obs):
        raise NotImplementedError

    def reset(self):
        pass


class Identity(Transform):
    def __call__(self, obs):
        return obs


class Compose(Transform):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, obs):
        for transform in self._transforms:
            obs = transform(obs)
        return obs

    def reset(self):
        for transform in self._transforms:
            transform.reset()


class ComposeMultimodal(Transform):
    def __init__(self, modal_dims, modal_transforms, is_torch=False):
        self._modal_dims = modal_dims
        self._modal_transforms = modal_transforms
        self._expected_obs_dim = np.sum(self._modal_dims)
        self._is_torch = is_torch

    def __call__(self, obs):
        assert obs.shape[-1] == self._expected_obs_dim
        res_obs = []
        last_dim = 0
        for (modal_dim, modal_transform) in zip(self._modal_dims, self._modal_transforms):
            res_obs.append(modal_transform(obs[..., last_dim:last_dim + modal_dim]))
            last_dim += modal_dim

        return torch.cat(res_obs, dim=-1) if self._is_torch else np.concatenate(res_obs, axis=-1)

    def reset(self):
        for modal_transform in self._modal_transforms:
            modal_transform.reset()


class AsType(Transform):
    def __init__(self, dtype=np.float32):
        self._dtype = dtype

    def __call__(self, obs):
        return obs.astype(self._dtype)


class TimeLimit(Transform):
    def __init__(self, max_timesteps):
        self._max_timesteps = max_timesteps
        self.reset()

    def __call__(self, obs):
        obs = np.concatenate((
            obs.reshape(-1),
            [(self._max_timesteps - self._curr_timestep) / self._max_timesteps]
        ), axis=0).astype(np.float32)
        self._curr_timestep += 1
        return obs

    def reset(self):
        self._curr_timestep = 0


class Reshape(Transform):
    def __init__(self, shape=-1):
        self._shape = shape

    def __call__(self, obs):
        return obs.reshape(self._shape)


class Transpose(Transform):
    def __init__(self, transpose):
        self._transpose = transpose

    def __call__(self, obs):
        return obs.transpose(self._transpose)


class FrameStack(Transform):
    def __init__(self, frame_dim):
        """ stack observation along axis 0. Assumes observation has 1 less dimension
        """
        assert len(frame_dim) > 1
        self._frame_dim = frame_dim
        self._frames = np.zeros(shape=frame_dim, dtype=np.float32)

    def __call__(self, obs):
        self._frames = np.concatenate((self._frames[1:], [obs]))
        return self._frames

    def reset(self):
        self._frames.fill(0)


class Normalize(Transform):
    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    def __call__(self, obs):
        return (obs - self._mean) / self._var


class Take(Transform):
    def __init__(self, axis, take_idx):
        self._axis = axis
        self._take_idx = take_idx

    def __call__(self, obs):
        return np.take(obs, self._take_idx, axis=self._axis)

class DictToNumPy(Transform):
    def __init__(self, keys):
        self._keys = keys

    def __call__(self, obs):
        return np.stack([obs[key] for key in self._keys])
