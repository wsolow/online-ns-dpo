import numpy as np
from algos.action_selection import get_actions_features
from utils.utils import split_state_actions
from utils.utils import IDX_PREF, IDX_NPREF
from algos.dpo import DirectPreferenceOptimization
from algos.sigmoidloss import SigmoidLossOptimization

def set_reward_params(
    feature_dim: int, 
    v1: float=1.,
    v2: float=2.
):
    assert (feature_dim > 0) and (feature_dim % 2 == 0)

    rparams = np.array([v1, v2], np.float32)
    if feature_dim > 2:
        rparams = np.repeat(
            rparams,
            feature_dim // 2
        )

    assert feature_dim == rparams.shape[0]
    return rparams

def get_num_params(config):
    return config.num_steps

def faury_rparams(config):
    """
    Faury transition from one point to another
    """
    num_data = get_num_params(config)
    rparams = np.array([1., 0.] * config.state_dim, np.float32)
    rparams_end = np.array([0., 1.] * config.state_dim, np.float32)
    drift_coef = config.drift_coef
    t1 = int(num_data / 3)
    t2 = int(2 * num_data / 3)
    itv = t2 - t1

    res = list()
    for i in range(num_data):
        if i < t1:
            res.append(rparams)
        elif i >= t1 and i < t2:
            res.append(
                np.array(
                    [
                        np.cos(((i - t1) / itv) * (drift_coef * np.pi / 2)),
                        np.sin(((i - t1) / itv) * (drift_coef * np.pi / 2))
                    ] * config.state_dim
                )
            )
        else:
            res.append(
                np.array(
                    [
                        np.cos((t2 / itv) * (drift_coef * np.pi / 2)),
                        np.sin((t2 / itv) * (drift_coef * np.pi / 2))
                    ] * config.state_dim
                )
            )
        
    return np.array(res) / np.sqrt(config.state_dim)

def linear_rparams(config):
    """
        linear transition from one point to another.
    """
    num_data = get_num_params(config)
    rparams = set_reward_params(
        config.state_dim * 2,
        config.reward_v1,
        config.reward_v2,
    )
    rparams_end = set_reward_params(
        config.state_dim * 2,
        config.tv.v1_target,
        config.tv.v2_target,
    )

    res = list()
    for i in range(num_data):
        ratio = (num_data - i) / num_data
        res.append(
            ratio * rparams + (1 - ratio) * rparams_end
        )
        
    return np.array(res)

def set_reward_params_tv(config):
    """
    Set the reward parameters for non-stationarity
    """
    feature_dim = config.state_dim * 2
    v1 = config.reward_v1
    v2 = config.reward_v2
    num_data = get_num_params(config)

    assert (feature_dim > 0) and (feature_dim % 2 == 0)

    rparams = np.array([v1, v2], np.float32)
    if feature_dim > 2:
        rparams = np.repeat(
            rparams,
            feature_dim // 2
        )

    assert feature_dim == rparams.shape[0]

    rparams = np.tile(
        rparams[None, :],
        (num_data, 1)
    )

    # apply recipes for non-stationarity
    if config.tv.use:
        if config.tv.type == "linear":
            rparams = linear_rparams(config)
        if config.tv.type == "faury":
            rparams = faury_rparams(config)

    return rparams


def calc_pseudo_regret(
    config,
    opt_agent,
    pref_data
):

    if isinstance(opt_agent, DirectPreferenceOptimization):
        func_rew = opt_agent.calc_implicit_reward
    elif isinstance(opt_agent, SigmoidLossOptimization):
        func_rew = opt_agent.calc_reward
    else:
        raise NotImplementedError

    rewards = dict()
    states, actions = split_state_actions(config.state_dim, pref_data)
    opt_policy = opt_agent.ret_policy()

    opt_actions = np.argmax(opt_policy(states), axis=1)
    rewards["optimal"] = func_rew(
        states,
        opt_actions
    )

    rewards["pref"] = func_rew(
        states,
        actions[:, IDX_PREF]
    )
    rewards["npref"] = func_rew(
        states,
        actions[:, IDX_NPREF]
    )

    regret_avg = (
        2 * rewards["optimal"].sum() - rewards["pref"].sum() - rewards["npref"].sum()
    ) / 2

    regret_avg = rewards["optimal"].sum() - rewards["pref"].sum()

    regret_pref = np.maximum(0,rewards["pref"].sum() - rewards["npref"].sum())
    
    regret_pref = 1 if rewards["pref"].sum() - rewards["npref"] < 0 else 0
    return regret_avg, regret_pref, rewards
