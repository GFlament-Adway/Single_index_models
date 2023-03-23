import numpy as np
import pandas as pd
from generate_poisson import Poisson_process
import math


def first_test(x):
    return 0.1 * x ** 2


def sigmoid(x, threshold=0.5):
    p = np.exp(x) / (1 + np.exp(x))
    return np.where(p > threshold, 1, 0)


def alpha_func(x, theta=2):
    #return 15/np.exp(theta*x - 2)
    return np.exp(np.dot(x, theta))


def duration_data_Nielsen_2022(n_ind=3, max_t=100, theta=2, n_params=1):
    init_state = [3 + k / 10 for k in range(10)]
    X = [[[np.random.choice([np.random.choice(init_state) for _ in range(max_t)]) for _ in range(n_params)] for _ in range(n_ind)]]
    for t in range(1, max_t):
        X += [[[X[t - 1][k][j] + np.random.normal(0, 0.07) for j in range(n_params)]for k in range(n_ind)]]
    X = np.array(X)
    obs = [Poisson_process(np.array(alpha_func(X[:, i], theta=theta)), 2*np.array(alpha_func(X[:, i], theta=theta)))
    for i in range(len(X[0, :]))]
    return obs, X


def get_data(n=1000, true_beta=[3, 1], link_func=first_test, Y_type="c"):
    """
    :param n: Number of observations
    :return:
    """
    X_1 = np.random.normal(0, 1, n)
    X_2 = np.random.normal(0, 1, n)
    betas = true_beta
    X = pd.DataFrame.from_dict({"X_1": X_1, "X_2": X_2})
    if Y_type == "c":
        Y = pd.DataFrame.from_dict({"Y": link_func(betas[0] * X_1 + betas[1] * X_2) + np.random.normal(0, 1, n)})

    elif Y_type == "binary":
        Y = pd.DataFrame.from_dict({"Y": link_func(betas[0] * X_1 + betas[1] * X_2)})

    return X, Y, betas
