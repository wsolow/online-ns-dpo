import numpy as np
from utils.utils import IDX_PREF, IDX_NPREF

def init_cov_matrix(
    config,
    feature_dim,
):
    return np.linalg.inv(
        np.eye(feature_dim) * (config.cov_init_coef / (config.reg_coef)**2)
    )

def init_cov_matrix_tv(
    config,
    feature_dim,
):
    return np.eye(feature_dim) * (config.cov_init_coef)

def get_actions_features(
    feature_func,
    states,
    actions
):
    pref_actions = actions[:, IDX_PREF]
    npref_actions = actions[:, IDX_NPREF]

    pref_feat = feature_func(states, pref_actions)
    npref_feat = feature_func(states, npref_actions)

    return pref_actions, npref_actions, pref_feat, npref_feat

def calc_feat_diff(
    feature_func,
    states,
    actions
):
    pref_actions, npref_actions, pref_feat, npref_feat = get_actions_features(
        feature_func,
        states,
        actions
    )

    return pref_feat - npref_feat

def calc_feat_sum(
    feature_func,
    states,
    actions
):
    pref_actions, npref_actions, pref_feat, npref_feat = get_actions_features(
        feature_func,
        states,
        actions
    )

    return pref_feat + npref_feat

def fast_update_cov(
    feature_func,
    inv_cov: np.ndarray,
    new_states: np.ndarray,
    new_actions: np.ndarray,
):
    feat_diffs = calc_feat_diff(feature_func, new_states, new_actions)

    for i, phi in enumerate(feat_diffs):
        u, v = phi[:, None], phi[:, None]
        inv_cov -= (inv_cov @ u) @ (v.T @ inv_cov) / (1 + v.T @ inv_cov @ u)

    return inv_cov

def update_cov_tv(
    config,
    feature_func,
    cov: np.ndarray,
    new_states: np.ndarray,
    new_actions: np.ndarray,
    gamma
):
    feat_diffs = calc_feat_diff(feature_func, new_states, new_actions)
    assert feat_diffs.shape[0] == 1

    cov = (1 - gamma) * (config.cov_init_coef) * np.eye(cov.shape[0]) + gamma * cov
    for i, phi in enumerate(feat_diffs):
        u, v = phi[:, None], phi[:, None]
        cov += u @ v.T

    return cov

def calc_bonus(
    feature_func,
    states: np.ndarray,
    actions: np.ndarray,
    inv_cov: np.ndarray
):
    feat_diffs = calc_feat_diff(feature_func, states, actions)
    return np.sqrt(
        np.matmul(
            np.matmul(
                feat_diffs,
                inv_cov
            ),
            feat_diffs.T
        ).diagonal()
    )

def comb_pairs(
    cands
):
    """
    give all the possible pairs of given candidate values.
    assumes cands.ndim == 1
    """
    assert cands.ndim == 1, "[comb_pairs] got cands.ndim > 1"
    pairs = np.concatenate(
        [
            np.repeat(cands, cands.shape[0])[:, None],
            np.tile(cands, cands.shape[0])[:, None]
        ],
        axis = 1
    )

    return pairs

def um_find_pair(
    feature_func,
    state,
    actions,
    inv_cov
):
    """
    Given a state and action pairs to consider, find the argmax pair of the bonus term.
    """

    states = np.tile(state, (actions.shape[0], 1))
    bonuses = calc_bonus(
        feature_func,
        states,
        actions,
        inv_cov
    )
    argmax_bonus = np.argmax(bonuses)

    return actions[argmax_bonus]

def filter_actions(
    config,
    agent,
    states,
    inv_cov
):
    """
    Filter actions based on the criterion below:
    r(x, a) - r(x, a') + bonus(x, a, a') >= 0 forall a' in possible actions
    """
    feature_func = agent.feature_func
    actions = np.arange(config.action_num)
    res = list()
    for i in range(states.shape[0]):
        res_i = list()
        state = np.repeat(states[i][None, :], config.action_num-1, axis=0)
        for j in actions:
            actions_cand = np.concatenate(
                [
                    np.ones(config.action_num)[:, None] * j,
                    actions[:, None]
                ],
                axis=1
            )
            actions_cand = np.concatenate(
                [actions_cand[:j], actions_cand[j+1:]],
                axis=0
            )
            bonuses_cand = calc_bonus(
                feature_func,
                state,
                actions_cand,
                inv_cov
            )
            feat_diff, reward_diffs = agent.calc_log_ratio_diff(
                np.concatenate(
                    [state, actions_cand],
                    axis=1
                )
            )
            values = reward_diffs + config.bonus_coef * bonuses_cand
            # print(f"reward_diffs: max {reward_diffs.max()} | min {reward_diffs.min()}")
            # print(f"bonuses_cand: max {bonuses_cand.max()} | min {bonuses_cand.min()}")
            # print(f"values: max {values.max()} | min {values.min()}")
            if (values > 0).all():
                res_i.append(j)
        if len(res_i) < 2:
            print(f"less than 1 action remained. using all actions")
            res.append(actions.copy())
        else:
            print(f"remaining actions: {len(res_i)}")
            res.append(np.array(res_i))
    return res

def ucb_find_pair(
    config,
    agent,
    state,
    actions,
    inv_cov
):
    """
    Given a state and action pairs to consider, find the argmax pair of the UCB term.
    """
    if config.env_bandit == "LogLinearBandit":
        rew_func = agent.calc_log_ratio_diff
    elif config.env_bandit == "LinearBandit":
        rew_func = agent.calc_rew_diff
    else:
        raise NotImplementedError

    feature_func = agent.feature_func

    states = np.tile(state, (actions.shape[0], 1))
    bonuses = calc_bonus(
        feature_func,
        states,
        actions,
        inv_cov
    )
    # implicit reward differences
    feat_diff, reward_diffs = rew_func(
        np.concatenate(
            [states, actions],
            axis=1
        )
    )

    ucbs = reward_diffs + config.bonus_coef * bonuses
    argmax_bonus = np.argmax(ucbs)
    
    return actions[argmax_bonus]

def diucb_find_pair(
    config,
    agent,
    state,
    actions,
    inv_cov
):
    """
    Given a state and action pairs to consider, find the argmax pair of the Di et al.- style UCB term.
    """
    feature_func = agent.feature_func

    states = np.tile(state, (actions.shape[0], 1))
    bonuses = calc_bonus(
        feature_func,
        states,
        actions,
        inv_cov
    )
    # implicit reward differences
    feat_diff, reward_sums = agent.calc_rew_sum(
        np.concatenate(
            [states, actions],
            axis=1
        )
    )

    ucbs = reward_sums + config.bonus_coef * bonuses
    argmax_bonus = np.argmax(ucbs)
    
    return actions[argmax_bonus]

def select_action_pair(
    config,
    states,
    # feature_func=None,
    agent=None,
    inv_cov=None,
    method="random"
):
    
    actions = np.repeat(
        np.arange(config.action_num)[None, :],
        states.shape[0],
        axis=0
    )
    if method == "random":
        probs = np.random.random(actions.shape).argsort(axis=1)[:, ::-1]
        actions = actions[
            np.tile(
                np.arange(actions.shape[0])[:, None],
                (1, 2)
            ),
            probs[:, :2]
        ]
        selected_pairs = actions

    elif method == "um":
        feature_func = agent.feature_func
        selected_pairs = list()

        for i in range(states.shape[0]):
            state_i = states[i][None, :]
            actions_i = actions[i].copy()

            # obtain possible action pairs of A_t(s_t)
            cand_action_pairs = comb_pairs(actions_i)

            # find an action pair which maximises the bonus term
            pair_um = um_find_pair(
                feature_func,
                state_i,
                cand_action_pairs,
                inv_cov
            )

            selected_pairs.append(pair_um)

        selected_pairs = np.array(selected_pairs)

    elif method == "ucb":
        feature_func = agent.feature_func
        selected_pairs = list()

        # construct subset of original A_t(s_t)
        if config.filter_actions:
            filtered_actions = filter_actions(
                config,
                agent,
                states,
                inv_cov
            )

        for i in range(states.shape[0]):
            state_i = states[i][None, :]
            
            if config.filter_actions:
                # obtain possible action pairs of A_t(s_t)
                actions_i = filtered_actions[i]
            else:
                actions_i = actions[i].copy()

            cand_action_pairs = comb_pairs(actions_i)

            # find an action pair which maximises the bonus term
            pair_ucb = ucb_find_pair(
                config,
                agent,
                state_i,
                cand_action_pairs,
                inv_cov
            )

            selected_pairs.append(pair_ucb)

        selected_pairs = np.array(selected_pairs)
    elif method == "diucb":
        feature_func = agent.feature_func
        selected_pairs = list()

        # construct subset of original A_t(s_t)
        if config.filter_actions:
            filtered_actions = filter_actions(
                config,
                agent,
                states,
                inv_cov
            )

        for i in range(states.shape[0]):
            state_i = states[i][None, :]
            
            if config.filter_actions:
                # obtain possible action pairs of A_t(s_t)
                actions_i = filtered_actions[i]
            else:
                actions_i = actions[i].copy()

            cand_action_pairs = comb_pairs(actions_i)

            # find an action pair which maximises the bonus term
            pair_ucb = diucb_find_pair(
                config,
                agent,
                state_i,
                cand_action_pairs,
                inv_cov
            )

            selected_pairs.append(pair_ucb)

        selected_pairs = np.array(selected_pairs)

    else:
        raise NotImplementedError

    return selected_pairs

def init_pref_data(config, env):
    state = env.reset()
    actions = select_action_pair(
        config, 
        state,
        method="random"
    )

    return np.concatenate(
        [state, actions],
        axis=1
    )