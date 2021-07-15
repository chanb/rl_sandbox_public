"""
Example usage:
python plot_evaluations.py \
    --root_dir=./results/hopper/1/ \
    --variants=bc,dac,sac-x \
    --file_name=train.pkl \
    --task_i=0 \
    --title=test \
    --recycling=3,3,3 \
    --multitasks=0,0,1 \
    --baselines=1,0,0

"""

import _pickle as pickle
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def str2list(v):
    return [item for item in v.split(',')] if v else []

def str2intlist(v):
    return [int(item) for item in v.split(',')] if v else []

def get_returns(root_dir, variants, multitasks, file_name, task_i, recycling):
    print("Root Directory: {}".format(root_dir))
    all_eval_rets = {}
    for variant in os.listdir(root_dir):
        if variant not in variants:
            continue

        multitask = multitasks[variants.index(variant)]
        curr_variant_eval_rets = []

        for full_dir, _, file_names in os.walk(os.path.join(root_dir, variant)):
            for curr_file in file_names:
                if curr_file != file_name:
                    continue
                full_path = os.path.join(full_dir, curr_file)
                print("Loading file: {}".format(full_path))
                data = pickle.load(open(full_path, "rb"))
                curr_variant_eval_ret = np.array(data['evaluation_returns'])
                if multitask and len(recycling) > 1:
                    recycling = np.concatenate(([0], np.cumsum(recycling)))
                    task_idxes = np.arange(curr_variant_eval_ret.shape[-1]) % np.sum(recycling)
                    task_idxes = np.where(np.logical_and(task_idxes >= recycling[task_i], task_idxes < recycling[task_i + 1]))[0]
                    curr_variant_eval_rets.append(curr_variant_eval_ret[..., task_i, task_idxes])
                else:
                    task_to_use = min(curr_variant_eval_ret.shape[-1] - 1, task_i)
                    if task_to_use != task_i:
                        print("Using task {} since {} is out of bound".format(task_to_use, task_i))
                    curr_variant_eval_rets.append(curr_variant_eval_ret[..., task_to_use])

        if len(curr_variant_eval_rets) == 0:
            continue

        all_eval_rets[variant] = (np.mean(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.std(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.min(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.max(curr_variant_eval_rets, axis=0)[..., 0])
        print("Variant {} has {} runs".format(variant, len(curr_variant_eval_rets)))
    return all_eval_rets

def main(root_dir, variants, multitasks, baselines, title, task_i, recycling, file_name="train.pkl", x_tick=5000):
    all_eval_rets = get_returns(root_dir, variants, multitasks, file_name, task_i, recycling)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    for idx, (variant, (ret_mean, ret_std, ret_min, ret_max)) in enumerate(all_eval_rets.items()):
        is_baseline = baselines[variants.index(variant)]
        idx %= len(CB_color_cycle)
        if is_baseline:
            ax.axhline(np.max(ret_mean), label=variant, color=CB_color_cycle[idx], linewidth=1)
        else:
            ax.plot(np.arange(1, len(ret_mean) + 1) * x_tick, ret_mean, label=variant, color=CB_color_cycle[idx], linewidth=1)
            ax.fill_between(np.arange(1, len(ret_mean) + 1) * x_tick, ret_mean - ret_std, ret_mean + ret_std, alpha=0.3, color=CB_color_cycle[idx])

    plt.grid(b=True, which='major')
    plt.title(f"{title} - Over 5 seeds")
    plt.xlabel("Number of steps")
    plt.ylabel("Average returns")
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=str, help="Directory consisting results")
    parser.add_argument("--variants", required=True, type=str2list, help="A list of algorithms to plot")
    parser.add_argument("--multitasks", required=True, type=str2intlist, help="A list of booleans indicating whether algorithm is trained with multitask. 0 means false, 1 means true.")
    parser.add_argument("--baselines", required=True, type=str2intlist, help="A list of booleans indicating whether algorithm is a baseline. 0 means false, 1 means true.")
    parser.add_argument("--file_name", default="termination_train.pkl", type=str, help="Filename to be matched")
    parser.add_argument("--title", required=True, type=str, help="Title of the plot")
    parser.add_argument("--task_i", default=0, type=int, help="Return for specific task")
    parser.add_argument("--recycling", default=[1], type=str2intlist, help="Fetch returns based on the recycling schedule")
    args = parser.parse_args()

    assert np.all(np.array(args.recycling) > 0)
    assert len(args.variants) == len(args.multitasks) == len(args.baselines)
    assert np.all(np.logical_or(np.array(args.multitasks) == 0, np.array(args.multitasks) == 1))
    assert np.all(np.logical_or(np.array(args.baselines) == 0, np.array(args.baselines) == 1))
    main(args.root_dir, args.variants, args.multitasks, args.baselines, args.title, args.task_i, args.recycling, args.file_name)
