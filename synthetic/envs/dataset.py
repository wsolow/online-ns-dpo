import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pickle
import os

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

from utils.visualize import draw_rewdiffs

def create_offline_data(
    config, 
    env, 
    feature_dim,
    feature_func,
    POAgent,
    rparams,
    ref_agent=None
):

    # check the existence of the dataset
    path_dataset = f"./datasets/{config.name_dataset}.pkl"
    if os.path.exists(path_dataset):
        print(f"Dataset loaded: {path_dataset}")
        with open(path_dataset, 'rb') as fp:
            d_load = pickle.load(fp)

        train_data = d_load["train_data"]
        valid_data = d_load["valid_data"]
        train_optactions = d_load["train_optactions"]
        valid_optactions = d_load["valid_optactions"]
        rparams = d_load["rparams"]
        train_rewdiffs = d_load["train_rewdiffs"]
        valid_rewdiffs = d_load["valid_rewdiffs"]

        # defining optagents with rparams given
        opt_agents = list()
        for i in range(rparams.shape[0]):
            opt_agents.append(
                POAgent(
                    config,
                    feature_dim,
                    feature_func,
                    param=rparams[i],
                    ref_agent=ref_agent,
                )
            )
    else:
        
        opt_agents = list()
        if config.algo == "sigmoidloss":
            mode_reward = "loglinear"
        elif config.algo == "dpo":
            mode_reward = "loglinear"

        if config.tv.use: # time-varying (non-stationary)
            train_states = list()
            train_actions = list()
            train_tsteps = list()
            train_optactions = list()
            train_rewdiffs = list()
            valid_states = list()
            valid_actions = list()
            valid_tsteps = list()
            valid_optactions = list()
            valid_rewdiffs = list()

            for i in range(config.num_steps):
                opt_param = rparams[i]
                
                # sample states
                states_i = list()

                num_data = config.train_per_step + config.valid_per_step
                for j in range(num_data):
                    states_i.append(env.reset())

                states_i = np.concatenate(
                    states_i,
                    axis=0
                )

                actions_i = select_action_pair(
                    config, 
                    states_i,
                    method="random"
                )

                opt_agent= POAgent(
                    config,
                    feature_dim,
                    feature_func,
                    param=opt_param,
                    ref_agent=ref_agent,
                )

                preference = get_preference(
                    config,
                    feature_func,
                    opt_agent,
                    states_i,
                    actions_i,
                )
                
                actions_i = apply_preference(
                    preference,
                    actions_i
                )
                optactions_i = opt_agent.ret_action_prob(states_i).argmax(axis=-1)

                # check reward difference
                dataset_i = np.concatenate([states_i, actions_i], axis = 1)
                if config.algo == "dpo":
                    _, rewdiffs_i = opt_agent.calc_log_ratio_diff(dataset_i)
                elif config.algo == "sigmoidloss":
                    _, rewdiffs_i = opt_agent.calc_rew_diff(dataset_i)

                train_states.append(states_i[:config.train_per_step, :])
                train_actions.append(actions_i[:config.train_per_step, :])
                train_tsteps.append(np.ones((config.train_per_step, 1)) * i)
                train_optactions.append(optactions_i[:config.train_per_step])
                valid_states.append(states_i[config.train_per_step:, :])
                valid_actions.append(actions_i[config.train_per_step:, :])
                valid_tsteps.append(np.ones((config.valid_per_step, 1)) * i)
                valid_optactions.append(optactions_i[config.train_per_step:])
                opt_agents.append(opt_agent)

                train_rewdiffs.append(rewdiffs_i[:config.train_per_step])
                valid_rewdiffs.append(rewdiffs_i[config.train_per_step:])
            
            train_states = np.concatenate(
                train_states,
                axis=0
            )
            train_actions = np.concatenate(
                train_actions,
                axis=0
            )
            train_tsteps = np.concatenate(
                train_tsteps,
                axis=0
            )
            train_optactions = np.concatenate(
                train_optactions,
                axis=0
            )
            valid_states = np.concatenate(
                valid_states,
                axis=0
            )
            valid_actions = np.concatenate(
                valid_actions,
                axis=0
            )
            valid_tsteps = np.concatenate(
                valid_tsteps,
                axis=0
            )
            valid_optactions = np.concatenate(
                valid_optactions,
                axis=0
            )

            train_rewdiffs = np.concatenate(train_rewdiffs, axis=0)
            valid_rewdiffs = np.concatenate(valid_rewdiffs, axis=0)

        else: # stationary setting
            num_data = config.num_steps * (
                config.train_per_step + config.valid_per_step
            )
            num_train = config.num_steps * config.train_per_step
            opt_param = rparams[0]

            # sample states
            states = list()
            for i in range(num_data):
                states.append(env.reset())

            states = np.concatenate(
                states,
                axis=0
            )

            actions = select_action_pair(
                config, 
                states,
                method="random"
            )

            # apply preference
            # opt_agent= SigmoidLossOptimization(
            opt_agent= POAgent(
                config,
                feature_dim,
                feature_func,
                param=opt_param,
                ref_agent=ref_agent,
            )

            preference = get_preference(
                config,
                feature_func,
                opt_agent,
                states,
                actions,
            )
            actions = apply_preference(
                preference,
                actions
            )
            optactions = opt_agent.ret_action_prob(states).argmax(axis=-1)

            # check reward difference
            dataset = np.concatenate([states, actions], axis = 1)
            if config.algo == "dpo":
                _, rewdiffs = opt_agent.calc_log_ratio_diff(dataset)
            elif config.algo == "sigmoidloss":
                _, rewdiffs = opt_agent.calc_rew_diff(dataset)

            train_states = states[:num_train, :]
            train_actions = actions[:num_train, :]
            train_tsteps = np.ones((train_states.shape[0], 1))
            train_optactions = optactions[:num_train]
            valid_states = states[num_train:, :]
            valid_actions = actions[num_train:, :]
            valid_tsteps = np.ones((valid_states.shape[0], 1))
            valid_optactions = optactions[num_train:]
            opt_agents.append(opt_agent)
            train_rewdiffs = rewdiffs[:num_train]
            valid_rewdiffs = rewdiffs[num_train:]

        train_data = np.concatenate(
            [train_states, train_actions, train_tsteps],
            axis=1
        )

        valid_data = np.concatenate(
            [valid_states, valid_actions, valid_tsteps],
            axis=1
        )

        with open(path_dataset, "wb") as fp:
            pickle.dump({
                "train_data": train_data,
                "valid_data": valid_data,
                "train_optactions": train_optactions,
                "valid_optactions": valid_optactions,
                "rparams": rparams,
                "train_rewdiffs": train_rewdiffs,
                "valid_rewdiffs": valid_rewdiffs,
            }, fp)
            print(f"Dataset created and saved: {path_dataset}")

    return train_data, valid_data, train_optactions, valid_optactions, opt_agents, train_rewdiffs, valid_rewdiffs

def report_rewdiffs(
    log_dir,
    config_name,
    seed,
    dataset,
    rewdiffs
):
    df = pd.DataFrame()
    df["timestep"] = dataset[: , -1]
    df["rewdiff"] = (rewdiffs > 0) 
    df = df.groupby(["timestep"]).mean().reset_index()

    df["config_name"] = config_name
    df["seed"] = seed
    
    path_csv = log_dir + "/rewdiff.csv"
    path_fig = log_dir + "/rewdiff.png"
    df.to_csv(path_csv)
    draw_rewdiffs(df, path_fig)