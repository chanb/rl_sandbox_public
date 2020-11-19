import _pickle as pickle
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def str2tuple(v):
    return tuple([item for item in v.split(',')] if v else [])

def get_returns(root_dir, variants, file_name):
    print("Root Directory: {}".format(root_dir))
    all_eval_rets = {}
    for variant in os.listdir(root_dir):
        if variant not in variants:
            continue

        curr_variant_eval_rets = []

        for full_dir, _, file_names in os.walk(os.path.join(root_dir, variant)):
            for curr_file in file_names:
                if curr_file != file_name:
                    continue
                full_path = os.path.join(full_dir, curr_file)
                print("Loading file: {}".format(full_path))
                data = pickle.load(open(full_path, "rb"))
                curr_variant_eval_rets.append(data['evaluation_returns'])

        if len(curr_variant_eval_rets) == 0:
            continue

        all_eval_rets[variant] = (np.mean(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.std(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.min(curr_variant_eval_rets, axis=0)[..., 0],
                                  np.max(curr_variant_eval_rets, axis=0)[..., 0])
        print("Variant {} has {} runs".format(variant, len(curr_variant_eval_rets)))
    return all_eval_rets

def main(root_dir, variants, title, file_name="train.pkl", x_tick=5000):
    all_eval_rets = get_returns(root_dir, variants, file_name)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    for idx, (variant, (ret_mean, ret_std, ret_min, ret_max)) in enumerate(all_eval_rets.items()):
        idx %= len(CB_color_cycle)
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
    parser.add_argument("--variants", required=True, type=str2tuple, help="A list of algorithms to plot")
    parser.add_argument("--file_name", default="termination_train.pkl", type=str, help="Filename to be matched")
    parser.add_argument("--title", required=True, type=str, help="Title of the plot")
    args = parser.parse_args()

    main(args.root_dir, args.variants, args.title, args.file_name)
