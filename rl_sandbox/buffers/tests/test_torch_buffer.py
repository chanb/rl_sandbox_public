import numpy as np
import torch
import unittest

from rl_sandbox.buffers.ram_buffer import (
    NoSampleError,
    NumPyBuffer,)
import rl_sandbox.constants as c


class TestTorchBuffer(unittest.TestCase):
    def setUp(self):
        self.obs_dim = (84, 84)
        self.h_state_dim = (1,)
        self.action_dim = (6,)
        self.reward_dim = (1,)
        self.info_attr = {
            c.VALUE: (self.reward_dim, np.float32),
            c.MEAN: (self.action_dim, np.float32)
        }

    def _compare_values(self, list_1, list_2):
        assert len(list_1) == len(list_2)
        for val_1, val_2 in zip(list_1, list_2):
            assert type(val_1) == type(val_2), f"{type(val_1)}, {type(val_2)}"
            if isinstance(val_1, np.ndarray):
                if not np.allclose(val_1, val_2):
                    return False
            elif isinstance(val_1, torch.Tensor):
                if not torch.allclose(val_1, val_2):
                    return False
            elif isinstance(val_1, dict):
                assert val_1.keys() == val_2.keys()
                for key in val_1:
                    if not self._compare_values(val_1[key], val_2[key]):
                        return False
            else:
                if val_1 != val_2:
                    return False
        return True

    def _generate_transition(self):
        obs = np.random.uniform(size=self.obs_dim).astype(np.float32)
        h_state = np.random.uniform(size=self.h_state_dim).astype(np.float32)
        action = np.random.uniform(size=self.action_dim).astype(np.float32)
        reward = np.random.uniform(size=self.reward_dim).astype(np.float32)
        done = np.random.randint(0, 2, size=(1,))
        info = {
            c.MEAN: np.random.uniform(size=self.action_dim).astype(np.float32),
            c.VALUE: np.random.uniform(size=self.reward_dim).astype(np.float32)
        }

        return (obs, h_state, action, reward, done, info)

    def test_sample_larger_than_buffer(self):
        buffer = TorchBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        t1 = self._generate_transition()
        t2 = self._generate_transition()
        
        empty = torch.tensor([])
        obss, h_states, acts, rews, dones, infos = buffer.sample(1)
        assert self._compare_values(obss, empty)
        assert self._compare_values(h_states, empty)
        assert self._compare_values(acts, empty)
        assert self._compare_values(rews, empty)
        assert self._compare_values(dones, empty)
        assert self._compare_values([infos], [{c.MEAN: empty, c.VALUE: empty}])

        buffer.push(*t1)
        expected_res = [torch.from_numpy(np.expand_dims(val, axis=0)) for val in t1[:-1]]
        expected_info = {key: torch.from_numpy(np.expand_dims(val, axis=0)) for key, val in t1[-1].items()}

        obss, h_states, acts, rews, dones, infos = buffer.sample(3)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        buffer.push(*t2)
        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t1[:-1], t2[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t1[-1][key], t2[-1][key]))) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos = buffer.sample(3)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample(self):
        buffer = TorchBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        t1 = self._generate_transition()
        t2 = self._generate_transition()
        t3 = self._generate_transition()

        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)

        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t1[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t1[-1][key], t3[-1][key]))) for key in t1[-1].keys()}

        obss, h_states, acts, rews, dones, infos = buffer.sample(2)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample_with_next_obs(self):
        buffer = TorchBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        t1 = self._generate_transition()
        t2 = self._generate_transition()
        t3 = self._generate_transition()
        next_t = self._generate_transition()

        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)

        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t1[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t1[-1][key], t3[-1][key]))) for key in t1[-1].keys()}

        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos = buffer.sample_with_next_obs(2, next_obs=next_t[0], next_h_state=next_t[1])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t2[:2], next_t[:2])]
        assert self._compare_values((next_obss, next_h_states), expected_res)

    def test_sample_consecutive_without_done(self):
        buffer = TorchBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        t1 = self._generate_transition()
        t2 = self._generate_transition()
        t3 = self._generate_transition()

        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)

        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t2[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t2[-1][key], t3[-1][key]))) for key in t2[-1].keys()}
        obss, h_states, acts, rews, dones, infos = buffer.sample_consecutive(2, end_with_done=False)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [torch.from_numpy(np.stack((val_1, val_2, val_3))) for val_1, val_2, val_3 in zip(t1[:-1], t2[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t1[-1][key], t2[-1][key], t3[-1][key]))) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos = buffer.sample_consecutive(3, end_with_done=False)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        obss, h_states, acts, rews, dones, infos = buffer.sample_consecutive(4, end_with_done=False)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample_consecutive_with_done(self):
        buffer = TorchBuffer(
            memory_size=4,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        t1 = list(self._generate_transition())
        t1[4] = np.array([0])
        t2 = list(self._generate_transition())
        t2[4] = np.array([1])
        t3 = list(self._generate_transition())
        t3[4] = np.array([1])
        t4 = list(self._generate_transition())
        t4[4] = np.array([0])

        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)
        buffer.push(*t4)

        expected_res = [torch.from_numpy(np.stack((val_1, val_2))) for val_1, val_2 in zip(t2[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t2[-1][key], t3[-1][key]))) for key in t2[-1].keys()}
        obss, h_states, acts, rews, dones, infos = buffer.sample_consecutive(2, end_with_done=True)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [torch.from_numpy(np.stack((val_1, val_2, val_3))) for val_1, val_2, val_3 in zip(t1[:-1], t2[:-1], t3[:-1])]
        expected_info = {key: torch.from_numpy(np.stack((t1[-1][key], t2[-1][key], t3[-1][key]))) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos = buffer.sample_consecutive(3, end_with_done=True)
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        try:
            _ = buffer.sample_consecutive(4, end_with_done=True)
            assert 0
        except NoSampleError:
            pass

        buffer = NumPyBuffer(
            memory_size=1,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)
        buffer.push(*t4)

        try:
            _ = buffer.sample_consecutive(2, end_with_done=True)
            assert 0
        except NoSampleError:
            pass


if __name__ == "__main__":
    unittest.main()
