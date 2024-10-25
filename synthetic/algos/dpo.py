import numpy as np

from .agent import Agent
from .action_selection import get_actions_features, calc_feat_diff
from utils.utils import sigmoid, split_state_actions

class DirectPreferenceOptimization(Agent):
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

        self.gamma = config.gamma
        if type(config.gamma2) == float:
            self.gamma2 = config.gamma2
        else:
            self.gamma2 = None

    def set_gamma2s(
        self,
        dataset=None,
        num_items=None,
    ):
        if dataset is not None:
            exponents = (self.config.odata.num_steps - 1) - dataset[:, -1]
        elif num_items is not None:
            exponents = np.arange(num_items) + 1 - num_items

        gamma2s = self.gamma2 ** exponents

        return gamma2s[:, None]

    def calc_implicit_reward(
        self,
        states,
        actions
    ) -> np.ndarray:
        """
        Calculates implicit reward corresponding to the given states and actions.
        NOTE: for each state, only **a single action** is given, instead of an action pair.
        """

        features = self.feature_func(
            states,
            actions
        )

        param_diff = self.param - self.ref_agent.param
        rewards = self.reg_coef * (features @ param_diff)

        return rewards

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
        rewards = self.calc_implicit_reward(states, actions).reshape(num_states, action_num)

        return rewards

    def calc_log_ratio_diff(
        self,
        dataset: np.ndarray
    ) -> np.ndarray:

        states, actions = split_state_actions(self.state_dim, dataset)

        feat_diff = calc_feat_diff(
            self.feature_func,
            states,
            actions
        )

        param_diff = self.param - self.ref_agent.param

        log_ratio_diff = self.reg_coef * (feat_diff @ param_diff)

        return feat_diff, log_ratio_diff

    def update_step(
        self,
        dataset: np.ndarray
    ) -> float:

        if self.config.tv.use:
            gammas = self.set_gammas(dataset=dataset)
            if self.config.use_sw:
                coefs_sw = self.apply_window(dataset=dataset)
            dataset = dataset[:, :-1]
            
            if self.gamma2 is not None:
                gamma2s = self.set_gamma2s(dataset=dataset)

        feat_diff, log_ratio_diff = self.calc_log_ratio_diff(dataset)

        coef = sigmoid(-log_ratio_diff)[:, None]
        neg_cur_data_grad = self.reg_coef * coef * feat_diff
        if self.gamma2 is not None:
            neg_cur_data_grad *= gamma2s
        
        if self.config.tv.use:
            grad = -(gammas * neg_cur_data_grad)
            if self.config.use_sw:
                grad *= coefs_sw
            if self.gamma2 is not None:
                grad *= gamma2s
            grad = grad.mean(axis=0)
        else:
            grad = -neg_cur_data_grad.mean(axis=0)

        sum_sq_grad = np.sum(np.square(grad))
        self.hist_grad_squared_norm += sum_sq_grad
        if self.is_adaptive and self.hist_grad_squared_norm > 0.:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad

        return np.sqrt(np.sum(np.square(grad)))
    
    def evaluate_loss(
        self, 
        dataset: np.ndarray
    ) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """

        if self.config.tv.use:
            gammas = self.set_gammas(dataset=dataset)
            dataset = dataset[:, :-1]

        feat_diff, log_ratio_diff = self.calc_log_ratio_diff(dataset)

        if self.config.tv.use:
            loss = -(gammas * np.log(sigmoid(log_ratio_diff))).mean()
        else:
            loss = -np.log(sigmoid(log_ratio_diff)).mean()

        return loss


