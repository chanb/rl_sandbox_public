import _pickle as pickle
import numpy as np
import os
import timeit
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import rl_sandbox.constants as c



class DummySummaryWriter():
    def add_scalar(self, arg_1, arg_2, arg_3):
        pass

    def add_scalars(self, arg_1, arg_2, arg_3):
        pass


def make_summary_writer(save_path, algo, cfg):
    summary_writer = DummySummaryWriter()
    cfg[c.ALGO] = algo
    if save_path is not None:
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        save_path = f"{save_path}/{time_tag}"
        os.makedirs(save_path, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
        pickle.dump(
            cfg,
            open(f'{save_path}/{algo}_experiment_setting.pkl', 'wb'))

    return summary_writer, save_path

def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, c.MAX_INT)

    np.random.seed(seed)
    torch.manual_seed(seed)


class EpochSummary:
    def __init__(self, default_key_length=10, padding=11):
        self._key_length = default_key_length
        self._padding = padding
        self._summary = dict()
        self._epoch = 0
        self._init_tic = timeit.default_timer()

    def log(self, key, value, track_std=True, track_min_max=True, axis=None):
        self._key_length = max(self._key_length, len(key))
        self._summary.setdefault(key, {
            c.LOG_SETTING: {
                c.STANDARD_DEVIATION: track_std,
                c.MIN_MAX: track_min_max,
                c.AXIS: axis,
            },
            c.CONTENT: []
        })
        self._summary[key][c.CONTENT].append(value)

    def new_epoch(self):
        self._epoch += 1
        self._summary.clear()
        self._curr_tic = timeit.default_timer()

    def print_summary(self):
        toc = timeit.default_timer()
        key_length = self._key_length + self._padding
        print("=" * 100)
        print(f"Epoch: {self._epoch}")
        print(f"Epoch Time Spent: {toc - self._curr_tic}")
        print(f"Total Time Spent: {toc - self._init_tic}")
        print("=" * 100)
        print('|'.join(str(x).ljust(key_length) for x in ("Key", "Content")))
        print("-" * 100)
        for key in sorted(self._summary):
            val = self._summary[key][c.CONTENT]
            setting = self._summary[key][c.LOG_SETTING]
            try:
                print('|'.join(str(x).ljust(key_length) for x in (f"{key} - AVG", np.mean(val, axis=setting[c.AXIS]))))
                if setting[c.STANDARD_DEVIATION]:
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - STD DEV", np.std(val, axis=setting[c.AXIS]))))
                if setting[c.MIN_MAX]:
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - MIN", np.min(val, axis=setting[c.AXIS]))))
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - MAX", np.max(val, axis=setting[c.AXIS]))))
            except:
                print(val)
                print(key)
                assert 0
        print("=" * 100)
