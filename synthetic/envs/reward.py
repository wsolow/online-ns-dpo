import numpy as np
from algos.action_selection import get_actions_features
from utils.utils import split_state_actions
from utils.utils import IDX_PREF, IDX_NPREF

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
    if config.env_bandit == "OfflineBandit":
        num_data = config.odata.num_steps
    else:
        num_data = config.num_data
    return num_data

def faury_rparams(config):
    # assert config.state_dim == 1
    num_data = get_num_params(config)
    rparams = np.array([1., 0.] * config.state_dim, np.float32)
    rparams_end = np.array([0., 1.] * config.state_dim, np.float32)
    drift_coef = config.odata.drift_coef

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

def set_reward_params_tv(
    config
):
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
    pref_data,
    mode="linear"
):
    if mode == "loglinear":
        func_rew = opt_agent.calc_implicit_reward
    elif mode == "linear":
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

    regret_pref = rewards["optimal"].sum() - rewards["pref"].sum()
    
    return regret_avg, regret_pref, rewards

def calc_KL_divergence(
    config,
    opt_agent,
    agent,
    pref_data,
    state_only=False,
):
    rewards = dict()
    if state_only:
        states = pref_data
    else:
        states, actions = split_state_actions(config.state_dim, pref_data)

    opt_policy = opt_agent.ret_policy()
    policy = agent.ret_policy()

    probs_opt = opt_policy(states)
    probs = policy(states)

    term1 = (probs * np.log(probs)).sum(axis=-1)
    term2 = (probs * np.log(probs_opt)).sum(axis=-1)
    KL_divs = config.reg_coef * (
        term1 - term2
    )
    
    if state_only:
        res = KL_divs.mean()
    else:
        res = KL_divs.sum()

    return res

def calc_expected_regret(
    config,
    agent,
    opt_agent,
    states,
    mode="linear"
):
    rewards = dict()
    opt_policy = opt_agent.ret_policy()
    agent_policy = agent.ret_policy()

    if mode == "loglinear":
        opt_probs = opt_policy(states)
        opt_rewards = opt_agent.get_rewards(
            opt_agent.action_num,
            opt_agent.feature_func,
            states
        )
        rewards["optimal"] = (opt_probs * opt_rewards).sum(axis=-1)

        agent_probs = agent_policy(states)
        rewards["agent"] = (agent_probs * opt_rewards).sum(axis=-1)
        
    elif mode == "linear":
        func_rew = opt_agent.calc_reward
        opt_actions = np.argmax(opt_policy(states), axis=1)
        rewards["optimal"] = func_rew(
            states,
            opt_actions
        )

        agent_actions = np.argmax(agent_policy(states), axis=1)
        rewards["agent"] = func_rew(
            states,
            agent_actions
        )
    else:
        raise NotImplementedError

    expected_regret = rewards["optimal"].mean() - rewards["agent"].mean()
    
    return expected_regret

def calc_reward_accuracy(
    config,
    agent,
    data,
    mode="linear"
):
    if mode == "loglinear":
        func_rew = agent.calc_implicit_reward
    elif mode == "linear":
        func_rew = agent.calc_reward
    else:
        raise NotImplementedError

    rewards = dict()
    states, actions = split_state_actions(config.state_dim, data)
    policy = agent.ret_policy()

    rewards["pref"] = func_rew(
        states,
        actions[:, IDX_PREF]
    )
    rewards["npref"] = func_rew(
        states,
        actions[:, IDX_NPREF]
    )

    rdiff = rewards["pref"] - rewards["npref"]
    racc = (rdiff > 0).mean()
    
    return racc