import numpy as np
import pandas as pd


def f(x):
    return 0.1 * x**2 

def get_data(n=1000, true_beta = [3,1]):
    """

    :param n: Number of observations
    :return:
    """
    X_1 = np.random.normal(0, 1, n)
    X_2 = np.random.normal(0, 1, n)
    betas = true_beta
    c = 0
    X = pd.DataFrame.from_dict({"X_1": X_1, "X_2": X_2})
    Y = pd.DataFrame.from_dict({"Y": c + f(betas[0] * X_1 + betas[1] * X_2) + np.random.normal(0, 1, n)})
    return X, Y, betas, c
