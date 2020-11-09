import numpy as np
import torch
import unittest

from rl_sandbox.buffers.ram_buffer import (
    NoSampleError,
    NumPyBuffer,)
import rl_sandbox.constants as c


class TestNumPyBuffer(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
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
            assert type(val_1) == type(val_2)
            if isinstance(val_1, np.ndarray):
                if not np.allclose(val_1, val_2):
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

    def _generate_transition(self, random_done=True):
        obs = np.random.uniform(size=self.obs_dim).astype(np.float32)
        h_state = np.random.uniform(size=self.h_state_dim).astype(np.float32)
        action = np.random.uniform(size=self.action_dim).astype(np.float32)
        reward = np.random.uniform(size=self.reward_dim).astype(np.float32)
        if random_done:
            done = np.random.randint(0, 2, size=(1,))
        else:
            done = 1
        info = {
            c.MEAN: np.random.uniform(size=self.action_dim).astype(np.float32),
            c.VALUE: np.random.uniform(size=self.reward_dim).astype(np.float32)
        }

        return (obs, h_state, action, reward, done, info)

    def test_push_empty(self):
        buffer = NumPyBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )

        transition = self._generate_transition()

        assert buffer.push(*transition)
        assert len(buffer) == 1
        assert self._compare_values(buffer[0], transition)

    def test_push_full(self):
        buffer = NumPyBuffer(
            memory_size=1,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )

        t1 = self._generate_transition()
        buffer.push(*t1)

        t2 = self._generate_transition()

        assert buffer.push(*t2)
        assert len(buffer) == 1
        assert self._compare_values(buffer[0], t2)

    def test_clear_empty(self):
        buffer = NumPyBuffer(
            memory_size=1,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.pointer == 0

    def test_remove(self):
        buffer = NumPyBuffer(
            memory_size=4,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )
        t1 = self._generate_transition()
        t2 = self._generate_transition()
        t3 = self._generate_transition()
        t4 = self._generate_transition()
        buffer.push(*t1)
        buffer.push(*t2)
        buffer.push(*t3)
        buffer.push(*t4)

        buffer.clear()
        assert len(buffer) == 0
        assert buffer.pointer == 0

    def test_sample_empty(self):
        buffer = NumPyBuffer(
            memory_size=3,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            rng=np.random.RandomState(1)
        )
        
        try:
            obss, h_states, acts, rews, dones, infos, lengths, idxes = buffer.sample(1)
            assert 0, "sample: Should raise NoSampleError"
        except NoSampleError:
            pass

        try:
            obss, h_states, acts, rews, dones, infos, next_obss, next_h_states, lengths, idxes = buffer.sample_with_next_obs(1, None, None)
            assert 0, "sample_with_next_obs: Should raise NoSampleError"
        except NoSampleError:
            pass

        try:
            obss, h_states = buffer.sample_init_obs(1)
            assert 0, "sample_init_obs: Should raise NoSampleError"
        except NoSampleError:
            pass

    def test_sample_larger_than_buffer(self):
        buffer = NumPyBuffer(
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

        buffer.push(*t1)
        expected_res = [np.stack((val, val, val)) for val in t1[:-1]]
        expected_info = {key: np.stack((val, val, val)) for key, val in t1[-1].items()}

        obss, h_states, acts, rews, dones, infos, lengths, idxes = buffer.sample(3)
        assert np.all(idxes == [0, 0, 0])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        buffer.push(*t2)
        expected_res = [np.stack((val_2, val_2, val_1)) for val_1, val_2 in zip(t1[:-1], t2[:-1])]
        expected_info = {key: np.stack((t2[-1][key], t2[-1][key], t1[-1][key])) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos, lengths, idxes = buffer.sample(3)
        assert np.all(idxes == [1, 1, 0])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample(self):
        buffer = NumPyBuffer(
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

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t2[:-1], t1[:-1])]
        expected_info = {key: np.stack((t2[-1][key], t1[-1][key])) for key in t1[-1].keys()}

        obss, h_states, acts, rews, dones, infos, lengths, idxes = buffer.sample(2)
        assert np.all(idxes == [1, 0])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample_with_next_obs(self):
        buffer = NumPyBuffer(
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

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t2[:-1], t1[:-1])]
        expected_info = {key: np.stack((t2[-1][key], t1[-1][key])) for key in t1[-1].keys()}

        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, idxes = buffer.sample_with_next_obs(2, next_obs=next_t[0], next_h_state=next_t[1])
        assert np.all(idxes == [1, 0])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t3[:2], t2[:2])]
        assert self._compare_values((next_obss, next_h_states), expected_res)

    def test_sample_init_obs(self):
        buffer = NumPyBuffer(
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

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t2[:2], t2[:2])]

        obss, h_states = buffer.sample_init_obs(2)
        assert self._compare_values((obss, h_states), expected_res)

    def test_sample_consecutive_without_done(self):
        buffer = NumPyBuffer(
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

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t2[:-1], t3[:-1])]
        expected_info = {key: np.stack((t2[-1][key], t3[-1][key])) for key in t2[-1].keys()}
        obss, h_states, acts, rews, dones, infos, lengths, idx = buffer.sample_consecutive(2, end_with_done=False)
        assert idx == 3
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(t1[:-1], t2[:-1], t3[:-1])]
        expected_info = {key: np.stack((t1[-1][key], t2[-1][key], t3[-1][key])) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos, lengths, idx = buffer.sample_consecutive(3, end_with_done=False)
        assert idx == 3
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        obss, h_states, acts, rews, dones, infos, lengths, idx = buffer.sample_consecutive(4, end_with_done=False)
        assert idx == 3
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

    def test_sample_consecutive_with_done(self):
        buffer = NumPyBuffer(
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

        expected_res = [np.stack((val_1, val_2)) for val_1, val_2 in zip(t2[:-1], t3[:-1])]
        expected_info = {key: np.stack((t2[-1][key], t3[-1][key])) for key in t2[-1].keys()}
        obss, h_states, acts, rews, dones, infos, lengths, idx = buffer.sample_consecutive(2, end_with_done=True)
        assert idx == 3
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(t1[:-1], t2[:-1], t3[:-1])]
        expected_info = {key: np.stack((t1[-1][key], t2[-1][key], t3[-1][key])) for key in t1[-1].keys()}
        obss, h_states, acts, rews, dones, infos, lengths, idx = buffer.sample_consecutive(3, end_with_done=True)
        assert idx == 3
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

    def test_push_wraparound(self):
        buffer = NumPyBuffer(
            memory_size=4,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )

        ts = [self._generate_transition() for _ in range(6)]
        for ti in ts:
            buffer.push(*ti)

        assert len(buffer) == 4
        assert buffer.pointer == 2
        assert self._compare_values(buffer.observations[3], ts[3][0])
        assert self._compare_values(buffer.observations[2], ts[2][0])
        assert self._compare_values(buffer.observations[1], ts[5][0])
        assert self._compare_values(buffer.observations[0], ts[4][0])

    def test_sample_with_next_obs_wraparound(self):
        buffer = NumPyBuffer(
            memory_size=4,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )

        ts = [self._generate_transition() for _ in range(4)]
        for ti in ts:
            buffer.push(*ti)
        next_t = self._generate_transition()

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(ts[0][:-1], ts[0][:-1], ts[3][:-1])]
        expected_info = {key: np.stack((ts[0][-1][key], ts[0][-1][key], ts[3][-1][key])) for key in ts[0][-1].keys()}

        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, idxes = buffer.sample_with_next_obs(3, next_obs=next_t[0], next_h_state=next_t[1])
        assert np.all(idxes == [0, 0, 3])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(ts[1][:2], ts[1][:2], next_t[:2])]
        assert self._compare_values((next_obss, next_h_states), expected_res)

        t4 = next_t
        buffer.push(*t4)
        t5 = self._generate_transition()
        buffer.push(*t5)
        t6 = self._generate_transition()
        buffer.push(*t6)
        next_t = self._generate_transition()

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(t6[:-1], t4[:-1], t6[:-1])]
        expected_info = {key: np.stack((t6[-1][key], t4[-1][key], t6[-1][key])) for key in ts[0][-1].keys()}

        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, idxes = buffer.sample_with_next_obs(3, next_obs=next_t[0], next_h_state=next_t[1])
        assert np.all(idxes == [2, 0, 2])
        assert self._compare_values((obss, h_states, acts, rews, dones), expected_res)
        assert self._compare_values([infos], [expected_info])

        expected_res = [np.stack((val_1, val_2, val_3)) for val_1, val_2, val_3 in zip(next_t[:2], t5[:2], next_t[:2])]
        assert self._compare_values((next_obss, next_h_states), expected_res)

    def test_sample_init_obs_wraparound(self):
        buffer = NumPyBuffer(
            memory_size=4,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr
        )

        ts = [self._generate_transition(random_done=False) for _ in range(6)]
        for ti in ts:
            buffer.push(*ti)

        expected_res = [np.stack((val_1, val_2, val_3, val_4)) for val_1, val_2, val_3, val_4 in zip(ts[4][:2], ts[4][:2], ts[3][:2], ts[4][:2])]

        obss, h_states = buffer.sample_init_obs(4)
        assert self._compare_values((obss, h_states), expected_res)

    def test_burn_in_window(self):
        buffer = NumPyBuffer(
            memory_size=10,
            obs_dim=self.obs_dim,
            h_state_dim=self.h_state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            infos=self.info_attr,
            burn_in_window=3,
            padding_first=True,
        )

        ts = [self._generate_transition() for _ in range(4)]
        for ti in ts:
            buffer.push(*ti)

        idxes = np.arange(4)
        buffer._get_burn_in_window(idxes)
        print('=' * 50)
        ts = [self._generate_transition() for _ in range(8)]
        for ti in ts:
            buffer.push(*ti)

        idxes = np.random.randint(5, size=4)
        buffer._get_burn_in_window(idxes)

        next_t = self._generate_transition()
        print('-' * 50)
        obss, h_states, _, _, _, next_obss, next_h_states, _, lengths, _ = buffer.sample_with_next_obs(5, next_t[0], next_t[1])
        print(obss)
        print(h_states)
        print(lengths)


if __name__ == "__main__":
    unittest.main()
