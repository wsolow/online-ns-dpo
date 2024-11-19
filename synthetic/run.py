import os
import argparse
import numpy as np
import wandb
import pandas as pd
from omegaconf import OmegaConf

from envs.linear_bandit import run_linear_bandit
from envs.offline_bandit import run_offline_bandit

from utils.io_utils import create_log_dirs, timeit
from utils.visualize import draw_results_from_df

def agg_dfs(paths, name_read="eval_df", name_save="eval_agg"):
    # aggregate .csv results
    df_agg = list()
    for path in paths:
        path_df = path + "/" + f"{name_read}.csv"
        df = pd.read_csv(path_df, index_col=0)
        df_agg.append(df)
    df_agg = pd.concat(df_agg)
    df_agg.to_csv(path + f"/{name_save}.csv")
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_regret_avg.png",
        target_y="regret_avg",
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_regret_pref.png",
        target_y="regret_pref",
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_KLdivs.png",
        target_y="KL_divs",
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_train_loss.png",
        target_x="steps",
        target_y="train_loss",
        xlabel="train steps",
        ylabel="train_loss"
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_valid_loss.png",
        target_x="steps",
        target_y="valid_loss",
        xlabel="train steps",
        ylabel="valid_loss"
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_expected_regret.png",
        target_x="steps",
        target_y="expected_regret",
        xlabel="train steps",
        ylabel="expected regret"
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_expected_RLHFobj.png",
        target_x="steps",
        target_y="expected_obj",
        xlabel="train steps",
        ylabel="expected RLHF objective"
    )
    draw_results_from_df(
        path + f"/{name_save}.csv",
        path + f"/{name_save}_racc.png",
        target_x="steps",
        target_y="reward_accuracy",
        xlabel="train steps",
        ylabel="reward accuracy"
    )

def run_multiple_seeds(args, config_name, config):
    log_dirs = create_log_dirs(args, config_name, config)
    for seed in range(config.seed_start, config.seed_start + config.num_seeds):
        log_dir = log_dirs[seed]
        if config.env_bandit == "LinearBandit":
            run_linear_bandit(seed, args, config_name, config, log_dir)
        elif config.env_bandit == "OfflineBandit":
            run_offline_bandit(seed, args, config_name, config, log_dir)

    # aggregate .csv results
    agg_dfs(
        log_dirs.values(), 
        name_read="eval_df",
        name_save="../eval_group"
    )

def parse_args():
    
    def str2bool(v):
        return v.lower() == "true"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_config", type=str, default="default")
    parser.add_argument("--default_config", type=str, default="default")
    parser.add_argument("--project", type=str, default=None)

    return parser.parse_args()

def prepare_config(args):
    config_default = OmegaConf.load(f"./configs/{args.default_config}.yaml")
    configs = list()
    config_names = list()
    if args.project is not None:
        cands_config = os.listdir(f"./configs/{args.project}")

        for cand in cands_config:
            if cand[-5:] == ".yaml":
                config = OmegaConf.load(f"./configs/{args.project}/{cand}")
                config = OmegaConf.merge(config_default, config)
                configs.append(config)
                config_names.append(cand[:-5])
    else:
        config = OmegaConf.load(f"./configs/{args.name_config}.yaml")
        config = OmegaConf.merge(config_default, config)
        configs.append(config)
        config_names.append(args.name_config)
    return configs, config_names

def get_project_paths(args, config_names):
    res = list()
    for config_name in config_names:
        base_path = f"./logs/{args.project}/{config_name}/"
        subpaths = os.listdir(base_path)
        for subpath in subpaths:
            cand_path = base_path + subpath
            if os.path.isdir(cand_path):
                res.append(cand_path)
    
    return res

if __name__ == "__main__":
    args = parse_args()
    configs, config_names = prepare_config(args)

    np.set_printoptions(precision=3, suppress=True)
    for idx_config in range(len(configs)):
        config = configs[idx_config]
        config_name = config_names[idx_config]
        run_multiple_seeds(args, config_name, config)

    if args.project is not None:
        paths_agg = get_project_paths(args, config_names)
        agg_dfs(
            paths_agg, 
            name_read="eval_group",
            name_save="../../eval_project"
        )
    