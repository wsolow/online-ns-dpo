import numpy as np

from utils.utils import sigmoid, IDX_NPREF
from algos.action_selection import calc_feat_diff

def get_preference(
    config,
    feature_func,
    opt_agent,
    states,
    actions,
    mode="linear"
):
    """
    sample preference label.
    """

    if mode == "linear":
        func_diff = opt_agent.calc_rew_diff
    elif mode == "loglinear":
        func_diff = opt_agent.calc_log_ratio_diff
    else:
        raise NotImplementedError

    feat_diff, reward_diff = func_diff(
        np.concatenate(
            [
                states,
                actions,
            ],
            axis=1
        )
    )

    if config.env_bandit == "OfflineBandit" and not config.odata.sample_prefs:
        preferences = (reward_diff > 0) * 1.
    else:
        preferences = np.random.binomial(
            1,
            sigmoid(reward_diff)
        )
    return preferences

def apply_preference(
    preferences,
    action_pairs
):
    """
    apply preference label to the given action pairs.
    flip the action pair when the given preference is IDX_NPREF.
    """
    
    flip = preferences == IDX_NPREF
    action_pairs[flip, :] = action_pairs[flip, ::-1]

    return action_pairs
