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
    kernel_mat = np.maximum(stats.norm.pdf(u, loc=mu, scale=1), 10e-50)
    diag = np.diag(kernel_mat.diagonal())
    return kernel_mat - diag


def NW(X, Y, betas, kernel_b = 1):
    """
    :param X: dataframe that contains the features.
    :param Y: dataframe that contains Y values.
    :param kernel_b: Float that contains the bandwidth of the kernel.
    :return: Nadaraya Watson estimate.
    """

    df_theta_x_ = pd.DataFrame.from_dict({"X_theta": X.dot(betas)}) #Compute $X^T \beta$
    mat = kernel(generate_kernel_matrix(df_theta_x_.iloc[:,0]), mu=0, bandwidth=kernel_b)
    denum = mat.sum(axis = 1)
    num = np.matmul(kernel(mat), Y.iloc[:,0])
    return num/denum

"""
class KernelReg():
    def __init__(self, X, Y, init_bandwidth):
        self.Y = Y
        self.X = X
        self.bandwidth = init_bandwidth
    
    def cv(self):
        loo__ = [[k for k in range(len(X)) if k != j] for j in range(len(X))]
        loo_x = self.x.iloc[loo__,:]
        loo_y = self.y
        hat_y = NW(loo_x, loo_y, self.bandwidth)
        return self.bandwidth
    
    def pred(self, x_new):
        pass
"""