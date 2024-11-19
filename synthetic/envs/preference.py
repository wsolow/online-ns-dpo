import numpy as np

from utils.utils import sigmoid, IDX_NPREF
from algos.action_selection import calc_feat_diff
from algos.dpo import DirectPreferenceOptimization
from algos.sigmoidloss import SigmoidLossOptimization

def get_preference(
    config,
    feature_func,
    opt_agent,
    states,
    actions
):
    """
    sample preference label.
    """

    if isinstance(opt_agent, DirectPreferenceOptimization):
        func_diff = opt_agent.calc_log_ratio_diff
    elif isinstance(opt_agent, SigmoidLossOptimization):
        func_diff = opt_agent.calc_rew_diff
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
    if config.env_bandit == "OfflineBandit" and not config.sample_prefs:
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
