import numpy as np
import pandas as pd


def f(x):
    return 0.1 * x**2 

def sigmoid(x, threshold = 0.5):
    p = np.exp(x)/(1 + np.exp(x))
    return p
    #return np.where(p > threshold, 1, 0)

def get_data(n=1000, true_beta = [3,1], link_func = f, Y_type="c"):
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

