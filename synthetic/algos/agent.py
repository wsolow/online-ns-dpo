import copy
import numpy as np
from utils.utils import sigmoid, softmax, IDX_PREF, IDX_NPREF
from envs.reward import calc_expected_regret, calc_KL_divergence, calc_reward_accuracy

class Agent:
    def __init__(
        self, 
        config,
        feature_dim,
        feature_func,
        param=None,
        ref_agent=None,
    ):
        self.config = config
        self.state_dim = config.state_dim
        self.action_num = config.action_num
        self.step_size = config.step_size
        self.num_iters = config.num_iters
        self.reg_coef = config.reg_coef
        self.l2_coef = config.l2_coef
        self.wandb_use = config.wandb.use
        self.freq_eval = config.freq_eval
        self.max_param_norm = config.max_param_norm

        self.is_adaptive = config.is_adaptive
        self.ada_coef = config.ada_coef
        self.hist_grad_squared_norm = 0.0

        self.ref_agent = ref_agent
        self.feature_dim = feature_dim
        self.feature_func = feature_func

        if param is not None:
            self.param = param
        if ref_agent is not None:
            self.ref_policy = ref_agent.ret_policy()
            if param is None:
                self.param = np.random.uniform(0, 1, self.feature_dim)
        else: # default reference policy: uniform
            if param is None:
                self.param = np.zeros(self.feature_dim)

    def policy(
        self, 
        action_num, 
        feature_func, 
        param, 
        states: np.ndarray
    ):
        num_states = states.shape[0]
        actions = np.tile(np.arange(action_num), num_states)
        states = np.repeat(states, action_num, axis=0)

        feature_mat = feature_func(states, actions)
        logits = np.matmul(feature_mat, param).reshape(num_states, action_num)
        prob = softmax(logits)

        return prob

    def ret_action_prob(
        self, 
        states: np.ndarray
    ): 
        return self.policy(
            self.action_num,
            self.feature_func,
            self.param,
            states
        )

    def ret_policy(self):
        action_num = self.action_num
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param

        def policy_out(states: np.ndarray):
            return self.policy(
                action_num,
                feature_func,
                param,
                states
            )
            
        return policy_out

    def project_param(self):
        norm_param = np.linalg.norm(self.param)
        if norm_param > self.max_param_norm:
            self.param = self.param / norm_param * self.max_param_norm

    def train(
        self,
        train_data,
        true_param: np.ndarray,
    ):
        if true_param is not None:
            assert true_param.shape == self.param.shape, 'text here'
            true_param_norm = np.linalg.norm(true_param)

        for step in range(self.num_iters):
            grad_norm = self.update_step(train_data)
            if (step == 0) or ((step+1) % self.freq_eval == 0):
                train_loss = self.evaluate_loss(train_data)
                param_norm = np.linalg.norm(self.param)
                cosine_sim = 0

                logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, "
                            f"grad_norm :{grad_norm:.4f}, param_norm: {param_norm:.4f}, "
                             )
                if true_param is not None:
                    cosine_sim = np.dot(self.param, true_param)/(true_param_norm * param_norm)
                    logging_str += f"cos_sim: {cosine_sim:.4f}."
                
                print(logging_str)

        if self.max_param_norm is not None and self.max_param_norm > 0:
            self.project_param()
        
        return train_loss
    
    def set_gammas(
        self,
        dataset=None,
        num_items=None,
    ):
        if dataset is not None:
            exponents = (self.config.odata.num_steps - 1) - dataset[:, -1]
        elif num_items is not None:
            exponents = (num_items - 1) - np.arange(num_items)

        gammas = self.gamma ** exponents

        return gammas[:, None]

    def apply_window(
        self,
        dataset=None,
        num_items=None,
    ):
        if dataset is not None:
            in_window = (self.config.odata.num_steps - dataset[:, -1]) <= self.config.window_size
        elif num_items is not None:
            in_window = (num_items - np.arange(num_items)) <= self.config.window_size

        coefs_window = 1.0 * in_window

        return coefs_window[:, None]

    def train_offline(
        self,
        train_data,
        valid_data,
        opt_agents
    ):
        indices_train = np.arange(train_data.shape[0])
        values = {
            "steps": list(),
            "train_loss": list(),
            "valid_loss": list(),
            "grad_norm": list(),
            "param_norm": list(),
            "expected_regret": list(),
            "expected_obj": list(),
            "racc": list()
        }

        valid_last_states = valid_data[-self.config.odata.valid_per_step:, :self.state_dim]
        valid_last_actions = valid_data[-self.config.odata.valid_per_step:, self.state_dim:]

        if self.config.algo == "sigmoidloss":
            mode_reward = "linear"
        elif self.config.algo == "dpo":
            mode_reward = "loglinear"

        for step in range(self.num_iters):
            # select batch from the training data
            indices_batch = np.random.choice(indices_train, self.config.odata.size_batch)
            train_batch = train_data[indices_batch]
            grad_norm = self.update_step(train_batch)

            if (step == 0) or ((step+1) % self.freq_eval == 0):
                train_loss = self.evaluate_loss(train_data)
                valid_loss = self.evaluate_loss(valid_data)
                param_norm = np.linalg.norm(self.param)

                # calculate expected reward, for datapoints in the last timestep
                # use valid_last_states
                expected_regret = calc_expected_regret(
                    self.config,
                    self,
                    opt_agents[-1],
                    valid_last_states,
                    mode=mode_reward
                )

                expected_obj = calc_KL_divergence(
                    self.config,
                    opt_agents[-1],
                    self,
                    valid_last_states,
                    state_only=True
                )

                racc = calc_reward_accuracy(
                    self.config,
                    self,
                    valid_data[-self.config.odata.valid_per_step:],
                    mode=mode_reward
                )

                values["steps"].append(step)
                values["train_loss"].append(train_loss)
                values["valid_loss"].append(valid_loss)
                values["grad_norm"].append(grad_norm)
                values["param_norm"].append(param_norm)
                values["expected_regret"].append(expected_regret)
                values["expected_obj"].append(expected_obj)
                values["racc"].append(racc)

                logging_str = (f"Iteration: {step: d}, train_loss: {train_loss: .4f}, valid_loss: {valid_loss: .4f}, "
                            f"grad_norm :{grad_norm:.4f}, param_norm: {param_norm:.4f}, expected_regret: {expected_regret:.4f} "
                             f"expected_RLHFobj: {expected_obj:.4f}, reward_accuracy: {racc:.4f}")
                
                print(logging_str)

        if self.max_param_norm is not None and self.max_param_norm > 0:
            self.project_param()
        
        return values

    @property
    def get_param(self) -> np.ndarray:
        return self.param
    
    def update_step(
        self, 
        dataset:np.ndarray
    ) -> float:
        pass

    def evaluate_loss(
        self, 
        dataset:np.ndarray
    ) -> float:
        pass
        
    def get_rewards(
        self,
        action_num,
        feature_func,
        states: np.ndarray
    ):
        pass


        
        
