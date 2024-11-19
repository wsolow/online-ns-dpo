import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .colors import ColorRevolver

D_KEYS0 = {
    "sw_dpo": "SW-DPO",
    "nsdpo": "NS-DPO",
    "vanilla_dpo": "DPO",
}
D_KEYS1 = {
    "g": "SW-DPO",
    "ns_dpo": "NS-DPO",
}
D_ALGS = {
    "nsdpo": 0,
    "sw_dpo": 0,
    "vanilla_dpo": 0,
}
L_ALGS = ["nsdpo", "sw_dpo", "vanilla_dpo"]

KEYS_REMOVE = ["0.9847"]

def remove_keys(keys):
    targets = list()
    for k in keys:    
        for kr in KEYS_REMOVE:
            if kr in k:
                targets.append(k)
                break
    for t in targets:
        del keys[t]
    return keys    

def parse_s1(s1):
    for k in D_KEYS0:
        s1 = s1.replace(k, D_KEYS0[k])
    return s1

def parse_s2(s2):
    if s2[0] == "g":
        s2 = f"$\gamma = {s2[1:]}$"
    elif s2[0] == "w":
        s2 = f"$w = {s2[1:]}$"
    return s2

def polish_title(title, names):
    res = list()
    for target in L_ALGS:
        for k in names:
            if target in k:
                res.append(target)
                break
    if len(res) == 1:
        title += f", {D_KEYS0[res[0]]}"
    return title

def parse_keys(keys):
    res = dict()
    for k in keys:
        s1 = parse_s1(k)
        s1 = s1.split("_")
        s2 = ""
        add_s2 = False
        if len(s1) > 1:
            s2 = parse_s2(s1[1])
            add_s2 = True
        res[k] = s1[0]
        if add_s2:
            res[k] += f" ({s2})"
    return res

def draw_results_from_df(
    path_df,
    path_fig,
    target_x="size_data",
    target_y="regret",
    xlabel="number of datapoints",
    ylabel="Cumulative Regret",
    title="Synthetic Experiments",
    figsize=(12, 6),
    fontsize_axes=30,
    fontsize_title=30,
    fontsize_ticks=20,
    fontsize_legs=20,
    linewidth=4.0,
):
    df = pd.read_csv(path_df, index_col=0)
    if target_y not in df.columns:
        return
    df_groupby = df.groupby(["config_name", target_x])
    df_avg = df_groupby.mean().reset_index()
    df_std = df_groupby.std().reset_index()

    fig = plt.figure(figsize=figsize)
    title = polish_title(title, df_avg.config_name.value_counts().keys())
    # plt.title(title, fontsize=fontsize_title)
    ax = plt.gca()

    steps = {k: df_avg.loc[df_avg.config_name==k][target_x] for k in df_avg.config_name.value_counts().keys()}
    means = {k: df_avg.loc[df_avg.config_name==k][target_y] for k in df_avg.config_name.value_counts().keys()}
    stds = {k: df_std.loc[df_std.config_name==k][target_y] for k in df_std.config_name.value_counts().keys()}

    allvalues = np.stack([means[k] for k in means])

    if target_y == "expected_regret":
        ylim_upper = 0.6
        ymin = -0.3
    elif target_y == "expected_obj":
        ylim_upper = 0.8
        ymin = -0.05
    elif target_y == "reward_accuracy":
        ylim_upper = 0.9
        ymin = 0.7
    else:
        ylim_upper = allvalues.max() * 1.05
        ymin = min(allvalues.min(), 0.)

    plt.ylim(ymin, ylim_upper)
    plt.xlabel(xlabel, fontsize=fontsize_axes)
    plt.ylabel(ylabel, fontsize=fontsize_axes)

    xmin = 10000
    xmax = -10000

    names = parse_keys(df_avg.config_name.value_counts().keys())
    names = remove_keys(names)
    cr = ColorRevolver()
    crg = ColorRevolver(colorset="G")
    crb = ColorRevolver(colorset="B")
    crr = ColorRevolver(colorset="R")
    for k in names:
        if "nsdpo" in k:
            color = crb.get_color()
        elif "sw_dpo" in k:
            color = crg.get_color()
        elif "vanilla_dpo" in k:
            color = crr.get_color()
        else:
            color = cr.get_color()
        plt.plot(
            steps[k],
            means[k],
            label=names[k],
            linewidth=linewidth,
            color=color,
        )
        plt.fill_between(
            steps[k],
            means[k]+stds[k],
            means[k]-stds[k],
            alpha=0.2,
            color=color,
        )
        if steps[k].min() < xmin:
            xmin = steps[k].min()
        if steps[k].max() > xmax:
            xmax = steps[k].max()

    plt.xlim(xmin, xmax)
    plt.grid(alpha=0.2)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    # legends = plt.legend(loc="upper right", fontsize=fontsize_legs)
    legends = plt.legend(loc="upper right", fontsize=fontsize_legs, ncols=2)
    for line in legends.get_lines():
        line.set_linewidth(linewidth * 2)
    plt.savefig(path_fig, bbox_inches='tight')

def draw_rewdiffs(
    df,
    path_fig,
    figsize=(5, 5),
    xlabel="timestep",
    ylabel="correct preference ratio"
):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.ylim(0., 1.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.bar(df["timestep"], df["rewdiff"])

    plt.savefig(path_fig, bbox_inches='tight')

if __name__ == "__main__":
    import argparse

    def parse_args():
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--project", type=str)
        parser.add_argument("--metric", type=str)
        parser.add_argument("--path_save", type=str, default="plots_new")

        return parser.parse_args()

    args = parse_args()
    path = f"./logs/{args.project}/"
    path_save = f"./{args.path_save}/"
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok=True)

    name_df = "eval_project"

    if args.metric == "expected_regret":
        draw_results_from_df(
            path + f"/{name_df}.csv",
            # path + f"/{name_df}_expected_regret_new.png",
            path_save + f"/{args.project}_expected_regret_new.png",
            target_x="steps",
            target_y="expected_regret",
            xlabel="Training Steps",
            ylabel="Expected Regret"
        )
    elif args.metric == "expected_obj":
        draw_results_from_df(
            path + f"/{name_df}.csv",
            # path + f"/{name_df}_expected_RLHFobjgap_new.png",
            path_save + f"/{args.project}_expected_RLHFobjgap_new.png",
            target_x="steps",
            target_y="expected_obj",
            xlabel="Training Steps",
            ylabel="RLHF Objective Gap"
        )
    elif args.metric == "racc":
        draw_results_from_df(
            path + f"/{name_df}.csv",
            # path + f"/{name_df}_expected_RLHFobjgap_new.png",
            path_save + f"/{args.project}_racc_new.png",
            target_x="steps",
            target_y="reward_accuracy",
            xlabel="Training Steps",
            ylabel="Reward Accuracy"
        )
    elif args.metric == "average_regret":
        draw_results_from_df(
            path + f"/{name_df}.csv",
            path + f"/{name_df}_regret_avg_new.png",
            target_y="regret_avg",
        )