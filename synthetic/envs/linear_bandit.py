import numpy as np
import wandb
import pandas as pd
import sys

from omegaconf import OmegaConf
from envs.reward import (
    set_reward_params_tv, 
    calc_pseudo_regret
)
from envs.preference import get_preference, apply_preference

from algos.sigmoidloss import SigmoidLossOptimization
from algos.dpo import DirectPreferenceOptimization
from algos.action_selection import (
    init_cov_matrix_tv,
    update_cov_tv,
    select_action_pair,
    init_pref_data
)

class LinearBanditEnv:
    def __init__(
        self, 
        config,
    ):
        self.state_dim = config.state_dim
        self.action_num = config.action_num
        self.action_space = [action_idx for action_idx in range(self.action_num)]
        self.reset_state_dist = config.reset_state_dist
        self.feature_func = self.get_feature_func()
        
        self.reset()

    def get_feature_func(self):
        def feature_func(
            states: np.ndarray,
            actions: np.ndarray
        ):
            assert states.ndim == 2
            assert actions.ndim == 1

            actions = actions[:, None].repeat(repeats=self.state_dim, axis=1)
            feature1 = ((actions + 1) * np.cos(states * np.pi))
            feature2 = ((1.0/(actions + 1)) * np.sin(states * np.pi))

            output = np.concatenate(
                [
                    feature1,
                    feature2
                ],
                axis=1
            )


            return output

        return feature_func

    def reset(self):
        rdist = self.reset_state_dist
        if rdist == "uniform":
            self.cur_state = np.random.uniform(0, 1, self.state_dim)[None, :]
        elif rdist == "normal":
            self.cur_state = np.random.normal(size=self.state_dim)[None, :]
        else:
            raise NotImplementedError()
        return self.cur_state

    @property
    def state(self):
        return self.cur_state

def run_linear_bandit(seed, args, config_name, config, log_dir):
    np.random.seed(seed)
    path_evals = log_dir + "/" + "evaluation.txt"
    path_df = log_dir + "/" + "eval_df.csv"

    with open(log_dir + "/" + config_name + ".yaml", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)

    print(f"Logging to {log_dir}")
    print("Seed:" + str(seed))
    
    mode = "loglinear"
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
            # group=config.wandb.group,
            # name=config.wandb.group + "_" + str(seed),
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
    pref_data = init_pref_data(config, env)
    # Append 0 to end
    pref_data = np.concatenate((pref_data,[[0]]),axis=1)

    # inv_cov = init_cov_matrix(config, feature_dim)
    cov = init_cov_matrix_tv(config, feature_dim)

    cols_eval = ["config_name", "seed", "size_data", "regret_avg", "regret_pref", "reward_opt", "reward_pref", "reward_npref"]
    evals = list()

    total_regret_avg = 0.
    total_regret_pref = 0.
    total_opt = 0.
    total_pref = 0.
    total_npref = 0.

    for t in range(config.num_steps):
        opt_param = opt_params[t]

        # Create the optimal agent at this time step        
        opt_agent= POAgent(
            config,
            feature_dim,
            feature_func,
            param=opt_param,
            ref_agent=ref_agent,
        )
        
        for i in range(config.train_per_step):
            
            # calculate parameter
            agent = POAgent(
                config,
                feature_dim,
                feature_func,
                param=None,
                ref_agent=ref_agent,
            )

            agent.train(
                train_data=pref_data,
                true_param=opt_param
            )

            # sample a state
            new_state = env.reset()

            # choose action pairs
            inv_cov = np.linalg.inv(cov)
            new_action_pair = select_action_pair(
                config,
                new_state,
                agent,
                inv_cov,
                config.action_selection
            )


            # get preference feedback
            # CURRENTLY FROM THE
            preference = get_preference(
                config,
                feature_func,
                agent,
                new_state,
                new_action_pair,
            )

            new_action_pair = apply_preference(
                preference,
                new_action_pair
            )

            # calculate regret
            regret_avg, regret_pref, rewards = calc_pseudo_regret(
                config,
                opt_agent,
                pref_data[-1:],
            )

            # update training data, covariance matrix
            pref_data = np.concatenate(
                [
                    pref_data,
                    np.concatenate(
                        [
                            new_state,
                            new_action_pair,
                            [[t]]
                        ],
                        axis=1
                    )
                ],
                axis=0
            )
            cov = update_cov_tv(
                config,
                feature_func,
                cov,
                new_state,
                new_action_pair,
                config.gamma
            )

            # print intermediate results, report to wandb
            size_data = len(pref_data)
            if (size_data % config.freq_report == 0) or (size_data == config.num_steps):

                total_regret_avg += regret_avg
                total_regret_pref += regret_pref
                total_opt += rewards["optimal"].sum()
                total_pref += rewards["pref"].sum()
                total_npref += rewards["npref"].sum()

                # pandas
                evals.append(
                    [
                        config_name, 
                        seed, 
                        size_data, 
                        total_regret_avg,
                        total_regret_pref, 
                        total_opt, 
                        total_pref, 
                        total_npref
                    ]
                )

                # print
                s_evals = f"[{size_data:>8d} points] regret_avg: {total_regret_avg:.4f} | regret_pref: {total_regret_pref:.4f} | reward_optimal: {total_opt:.4f} | reward_pref: {total_pref:.4f} | reward_npref: {total_npref:.4f}"
                print(s_evals)
                with open(path_evals, "a") as fp_evals:
                    fp_evals.write(s_evals + "\n")

                # wandb
                if config.wandb.use:
                    d_wandb = {
                        "evals/regret_avg": total_regret_avg,
                        "evals/regret_pref": total_regret_pref,
                        "evals/reward_opt": total_opt,
                        "evals/reward_pref": total_pref,
                        "evals/reward_npref": total_npref,
                    }
                    wandb.log(d_wandb, step=size_data)

    eval_df = pd.DataFrame(
        evals,
        columns=cols_eval
    )
    eval_df.to_csv(path_df)

    if config.wandb.use:
        wandb.finish()