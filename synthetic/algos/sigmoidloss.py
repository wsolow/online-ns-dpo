import numpy as np

from .agent import Agent
from .action_selection import get_actions_features, calc_feat_diff, calc_feat_sum
from utils.utils import sigmoid, split_state_actions

class SigmoidLossOptimization(Agent):
    def __init__(
        self, 
        config,
        feature_dim,
        feature_func,
        param=None,
        ref_agent=None,
    ):
        super().__init__(
            config, 
            feature_dim, 
            feature_func, 
            param, 
            ref_agent
        )

        self.cs = config.cs
        self.ks = config.ks
        self.gammas = np.array([])

    def calc_reward(
        self,
        states,
        actions
    ) -> np.ndarray:
        """
        Calculates reward corresponding to the given states and actions.
        NOTE: for each state, only **a single action** is given, instead of an action pair.
        """

        features = self.feature_func(
            states,
            actions
        )

        return features @ self.param

    def get_rewards(
        self,
        action_num,
        feature_func,
        states: np.ndarray
    ):
        num_states = states.shape[0]
        actions = np.tile(np.arange(action_num), num_states)
        states = np.repeat(states, action_num, axis=0)

        feature_mat = feature_func(states, actions)
        rewards = self.calc_reward(states, actions).reshape(num_states, action_num)

        return rewards

    def calc_rew_diff(
        self,
        dataset: np.ndarray
    ) -> np.ndarray:

        states, actions = split_state_actions(self.state_dim, dataset)

        feat_diff = calc_feat_diff(
            self.feature_func,
            states,
            actions
        )

        return feat_diff, (feat_diff @ self.param)

    def calc_rew_sum(
        self,
        dataset: np.ndarray
    ) -> np.ndarray:

        states, actions = split_state_actions(self.state_dim, dataset)

        feat_sum = calc_feat_sum(
            self.feature_func,
            states,
            actions
        )

        return feat_sum, (feat_sum @ self.param)

    def update_step(
        self,
        dataset: np.ndarray
    ) -> float:
        if self.config.env_bandit == "OfflineBandit":
            gammas = self.set_gammas(dataset=dataset)
        else:
            gammas = self.set_gammas(num_items=dataset.shape[0])
        feat_diff, rew_diff = self.calc_rew_diff(dataset)

        coef = sigmoid(rew_diff)[:, None]
        grad = (
            -gammas * (1 - coef) * feat_diff
        ).sum(axis=0) + self.l2_coef * self.cs * self.param

        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad

        return np.sqrt(np.sum(np.square(grad)))
    
    def evaluate_loss(self, dataset: np.ndarray) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if self.config.env_bandit == "OfflineBandit":
            gammas = self.set_gammas(dataset=dataset)
        else:
            gammas = self.set_gammas(num_items=dataset.shape[0])

        feat_diff, rew_diff = self.calc_rew_diff(dataset)
        loss = (
            -gammas * np.log(sigmoid(rew_diff))
        ).sum() + (self.l2_coef * self.cs / 2) * np.dot(self.param, self.param)

        return loss


