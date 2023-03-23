# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:18:04 2023
@author: guillaume.flament_ad
"""

import numpy as np
from scipy import stats
import pandas as pd


def generate_kernel_matrix(x):
    matrix = np.subtract.outer(x.to_numpy(), x.to_numpy())
    return np.array(matrix)


def kernel(u, mu=0, bandwidth=1):
    """
    :param u:
    :param mu:
    ...
    Leave one out grâce à -diag.
    """
    kernel_mat = np.maximum(stats.norm.pdf(u, loc=mu, scale=bandwidth), 10e-50)
    try:
        diag = np.diag(kernel_mat.diagonal())
    except ValueError:
        diag = 0
    return kernel_mat - diag


def NW_hazard(X, Y, betas, kernel_b=1):
    """
    :param X: dataframe that contains the features.
    :param Y: dataframe that contains Y values.
    :param kernel_b: Float that contains the bandwidth of the kernel.
    :return: Nadaraya Watson estimate.
    """

    #X = pd.DataFrame(np.array([X[int(Y[i]) - 1, i, :] for i in range(X.shape[1])]))
    df_theta_x_ = pd.DataFrame(np.matmul(X, betas))  # Compute $X^T \beta$

    mat = kernel(generate_kernel_matrix(df_theta_x_.iloc[:, 0]), mu=0, bandwidth=kernel_b)
    denum = mat.sum(axis=1)
    num = np.matmul(Y, mat)
    return num / denum


def NW(X, Y, betas, kernel_b=1):
    """
    :param X: dataframe that contains the features.
    :param Y: dataframe that contains Y values.
    :param kernel_b: Float that contains the bandwidth of the kernel.
    :return: Nadaraya Watson estimate.
    """

    df_theta_x_ = pd.DataFrame.from_dict({"X_theta": X.dot(betas)})  # Compute $X^T \beta$
    mat = kernel(generate_kernel_matrix(df_theta_x_.iloc[:, 0]), mu=0, bandwidth=kernel_b)
    denum = mat.sum(axis=1)
    print(mat.shape, Y.shape)
    num = np.matmul(mat, Y.iloc[:, 0])
    return num / denum


class KernelReg():
    def __init__(self, X, Y, betas, init_bandwidth):
        self.y = Y
        self.x = X
        self.betas = betas
        self.bandwidth = init_bandwidth

    def error(self, x, y):
        return (x - y) ** 2

    def NWE(self, x_new=None, X=None, Y=None, bandwidth=None):

        if X is None:
            X = self.x

        if Y is None:
            Y = self.y
        if bandwidth is None:
            bandwidth = self.bandwidth

        if x_new is None:
            df_theta_x_ = pd.DataFrame.from_dict({"X_theta": X.dot(self.betas)})  # Compute $X^T \beta$
            mat = kernel(generate_kernel_matrix(df_theta_x_.iloc[:, 0]), mu=0, bandwidth=bandwidth)
            denum = mat.sum(axis=1)
            num = np.matmul(mat, Y.iloc[:, 0])
            return num / denum

        else:
            df_theta_x_ = pd.DataFrame.from_dict({"X_theta": X.dot(betas)})  # Compute $X^T \beta$
            mat = kernel(df_theta_x_.iloc[:, 0] - x_new, mu=0, bandwidth=bandwidth)
            denum = mat.sum()
            num = np.matmul(mat, Y.iloc[:, 0])
            return num / denum

    def cv(self, bandwidth):
        loo__ = [[k for k in range(len(self.x)) if k != j] for j in range(len(self.x))]
        loo_x = [self.x.iloc[loo__[k], :] for k in range(len(loo__))]
        loo_y = [self.y.iloc[loo__[k], :] for k in range(len(loo__))]
        err = 0
        for k in range(len(loo_x)):
            # Least square cv.
            err += (self.y["Y"][k] - self.NWE(self.x.iloc[k].dot(self.betas), loo_x[k], loo_y[k],
                                              bandwidth=bandwidth)) ** 2
            # Other likelihood CV.
            # err += likelihood.
        return err

    def fit(self):
        self.bandwidth = fmin(KR.cv, self.bandwidth, maxiter=1000)[0]

    def pred(self, x_new=None):
        if x_new is not None:
            return [self.NWE(x_new=x) for x in x_new]
        return self.NWE()


if __name__ == "__main__":
    from gen_data import get_data, first_test
    from scipy.optimize import fmin
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric import kernel_regression
    import time

    time_start_own_package = time.time()
    init_betas = [3, 1]
    n_ind = 1000
    X, Y, betas = get_data(n=n_ind, true_beta=init_betas, link_func=first_test, Y_type="c")

    X_low = X.dot(init_betas).quantile(0.0025)
    X_up = X.dot(init_betas).quantile(0.975)

    integral_interval = np.linspace(X_low, X_up, num=1000)

    KR = KernelReg(X, Y, init_betas, 1.06 * np.std(X.dot(betas).to_numpy()) * n_ind ** (- 1. / (4 + len(X.columns))))
    KR.fit()
    time_end_own_package = time.time()
    print("init bw : ", 16 * np.std(X.dot(betas).to_numpy()) * n_ind ** (- 1. / (4 + len(X.columns))))
    print("bandwidth found by own package : ", KR.bandwidth)
    print("Time to compute for our package :", time_end_own_package - time_start_own_package)
    fitted_values = KR.pred(x_new=integral_interval)

    time_start_statsmod = time.time()
    mod = kernel_regression.KernelReg(Y["Y"], X.dot(betas).to_numpy(), var_type="c", reg_type="lc")
    fitted_mod = mod.fit(integral_interval)
    time_end_statsmod = time.time()

    print("Bandwidth found by Statsmodel package : ", mod.bw)
    print("Time to compute for Statsmodel package : ", time_end_statsmod - time_start_statsmod)

    plt.figure()

    plt.scatter(X.dot(betas), Y["Y"], label="Observed", s=0.1)
    plt.plot(integral_interval, fitted_values, label="Own package", linestyle=(0, (1, 1)))
    plt.plot(integral_interval, fitted_mod[0], label="Statsmodel package", linestyle=(0, (1, 3)))
    plt.legend()
    plt.plot()
    plt.show()
