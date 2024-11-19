import numpy as np
import wandb
import pandas as pd

from omegaconf import OmegaConf

from envs.dataset import create_offline_data, report_rewdiffs
from envs.linear_bandit import LinearBanditEnv
from envs.reward import (
    set_reward_params_tv, 
    calc_pseudo_regret
)

from algos.sigmoidloss import SigmoidLossOptimization
from algos.dpo import DirectPreferenceOptimization
from algos.action_selection import (
    init_cov_matrix_tv,
    update_cov_tv,
    select_action_pair,
    init_pref_data
)

def run_offline_bandit(seed, args, config_name, config, log_dir):
    np.random.seed(seed)
    path_evals = log_dir + "/" + "evaluation.txt"
    path_df = log_dir + "/" + "eval_df.csv"

    with open(log_dir + "/" + config_name + ".yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)

    print(f"Logging to {log_dir}")
    print("Seed:" + str(seed))
    
    if config.wandb.use:
        print("USING WANDB")
        wandb.login(
            key=config.wandb.key
        )
        if args.project is not None:
            wandb_project = args.project
        else:
            wandb_project = config.wandb.project
        wandb_group = config_name
        wandb_name = wandb_group + f"_{seed}"
        wandb.init(
            entity=config.wandb.entity,
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
            config=config,
            dir=log_dir,
        )

    state_dim = config.state_dim
    action_num = config.action_num
    feature_dim = 2 * config.state_dim
        
    env = LinearBanditEnv(config)
    feature_func = env.get_feature_func()

    opt_params = set_reward_params_tv(
        config,
    )

    if config.algo == "sigmoidloss":
        POAgent = SigmoidLossOptimization
    elif config.algo == "dpo":
        POAgent = DirectPreferenceOptimization
    else:
        raise NotImplementedError
    
    ref_agent = POAgent(
        config,
        feature_dim,
        feature_func,
        param=None,
    )

    print('reset state dist', config.reset_state_dist)

    # Generate datasets
    train_data, valid_data, train_optactions, valid_optactions, opt_agents, train_rewdiffs, valid_rewdiffs = create_offline_data(
        config,
        env,
        feature_dim,
        feature_func,
        POAgent,
        opt_params,
        ref_agent=ref_agent
    )

    report_rewdiffs(log_dir, config_name, seed, train_data, train_rewdiffs)

    print(f"opt_params: {opt_params.shape} | train_data: {train_data.shape} | valid_data: {valid_data.shape}")

    cols_eval = [
        "config_name", "seed", "size_data", "size_batch", 
        "steps", "train_loss", "valid_loss", 
        "expected_regret", "expected_obj", "reward_accuracy", "grad_norm", "param_norm"
    ]
    evals = list()

    agent = POAgent(
        config,
        feature_dim,
        feature_func,
        param=None,
        ref_agent=ref_agent,
    )

    values = agent.train_offline(
        train_data=train_data,
        valid_data=valid_data,
        opt_agents=opt_agents,
    )

    # aggregate values (pandas)
    for i in range(len(values["steps"])):
        evals.append(
            [
                config_name, 
                seed, 
                train_data.shape[0],
                config.size_batch,
                values["steps"][i],
                values["train_loss"][i],
                values["valid_loss"][i],
                values["expected_regret"][i],
                values["expected_obj"][i],
                values["racc"][i],
                values["grad_norm"][i],
                values["param_norm"][i],
            ]
        )

    eval_df = pd.DataFrame(
        evals,
        columns=cols_eval
    )
    eval_df.to_csv(path_df)

    if config.wandb.use:
        wandb.finish()