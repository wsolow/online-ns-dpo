import numpy as np

IDX_PREF = 1
IDX_NPREF = 0

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    assert x.ndim <= 2, "x has to be either 1 or 2-dimensional"
    expx = np.exp(x - x.max(axis=-1, keepdims=True))
    return expx / expx.sum(axis=-1, keepdims=True)

def ret_uniform_policy(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy(state: np.ndarray = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy

def split_state_actions(state_dim, dataset):
    """
    split the dataset into arrays of state and action pairs.
    """
    states = dataset[:, :state_dim]
    actions = dataset[:, state_dim:]

    return states, actions

if __name__ == "__main__":
    a = np.random.random((10))
    sa = softmax(a)
    print(sa, sa.sum())

    b = np.random.random((19, 5))
    sb = softmax(b)
    print(sb, sb.sum(axis=-1))
