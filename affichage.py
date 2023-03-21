# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:27:30 2023

@author: guillaume.flament_ad
"""

from estimate_PP import Parametric_Model
from gen_data import duration_data_Nielsen_2022, alpha
import statsmodels.api as sm
import pandas as pd
import numpy as np
from generate_poisson import KM_est
from KernelReg import KernelReg
import matplotlib.pyplot as plt


def affiche_poisson_process__(n_obs, n_times, theta=0.5):
    obs, X = duration_data_Nielsen_2022(n_obs, n_times, theta=theta)


    PM = Parametric_Model(X, obs, -10)
    print(PM.loss__(-10), PM.loss__(-5), PM.loss__(1), PM.loss__(2), PM.loss__(5))

    sf = KM_est(obs)
    sf.fit()

    plt.figure()
    plt.title("Survival times for : " + r"$\alpha(x) = \exp({theta}*x - 2)/15 $".format(theta=theta))
    for j in range(n_obs//10):
        plt.plot([np.exp(-(np.sum([np.exp(X.iloc[:, j].to_numpy()[i] * -theta) for i in range(t)]))) for t in
                  range(len(sf.surv_func))], alpha=0.1, color="red")
    plt.plot(sf.surv_func)
    plt.show()


if __name__ == "__main__":
    np.random.seed(12345)
    affiche_poisson_process__(100, 50, 2)
