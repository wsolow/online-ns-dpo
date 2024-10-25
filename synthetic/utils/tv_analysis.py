import numpy as np
import argparse
import os
from utils import sigmoid
from omegaconf import OmegaConf


def calculate_feature_bound(config):
    """
        * assumes linear reward setting in time-varying duelling bandits.
    """
    return config.action_num

def calculate_cs(config):
    """
        * assumes linear reward setting in time-varying duelling bandits.
    """
    fbound = calculate_feature_bound(config)
    term = 2 * fbound * config.max_param_norm

    return sigmoid(term) * sigmoid(-term)

def calculate_gamma(config):
    bt = config.tv.bt
    fdim = 2 * config.state_dim

    return (1 - (bt / (fdim * config.num_data)) ** (2 / 3))

def calculate_betatd(config):
    """
        Calculate the beta_t(delta), which is going to be used for calculating the bonus.
        * assumes linear reward setting in time-varying duelling bandits.
    """
    cs = calculate_cs(config)
    fbound = calculate_feature_bound(config)
    fdim = 2 * config.state_dim
    gamma = calculate_gamma(config)

    t1 = np.sqrt(config.l2_coef) * cs * config.max_param_norm
    t2 = np.sqrt(
        -2 * np.log(config.delta) + fdim * np.log(
                1 + (
                    (4 * (fbound ** 2) * (1 - (gamma ** (2 * config.num_data))) ) / (config.l2_coef * fdim * (1 - gamma**2))
            )
        )
    )

    return t1 + t2

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

if __name__ == "__main__":
    args = parse_args()
    configs, config_names = prepare_config(args)

    config = configs[0]
    fbound = calculate_feature_bound(config)
    cs = calculate_cs(config)
    gamma = calculate_gamma(config)
    betatd = calculate_betatd(config)
    bonus_coef = 2 * betatd / cs

    print(f"""
        analysis of {config_names[0]}:
            fbound : {fbound}
            cs : {cs}
            gamma : {gamma}
            betatd : {betatd}
            bonus_coef : {bonus_coef}
    """)
    