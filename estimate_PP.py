import numpy as np
from gen_data import duration_data_Nielsen_2022
import scipy
import pandas as pd
from KernelReg import NW_hazard

class Parametric_Model:
    def __init__(self, X, obs, init_theta, bw = 1, param=True):
        """
        :param X: Covariates.
        :param Y: Duration times.
        :param Delta: indicator of censorship.
        """
        self.X = X
        self.obs = obs
        self.n_times = X.shape[0]
        self.n_obs = X.shape[1]
        self.max_time = X.shape[0]
        self.theta = init_theta
        self.param_estimator = param
        self.bw = bw

    def intensity__(self, X):
        return np.exp(np.dot(X, self.theta))

    def np_intensity__(self, X):
        df_jump_time = pd.DataFrame([self.obs[i].jump_time for i in range(self.n_obs)]).T
        NP_intensity_est = pd.DataFrame()
        for t in range(self.max_time):
            df_X = np.array(X[t, :,:])
            NP_intensity_est = pd.concat([NP_intensity_est, NW_hazard(df_X, df_jump_time, self.theta, self.bw)])
        return NP_intensity_est.to_numpy()

    def loss__(self, theta):
        if type(theta) == np.ndarray:
            theta = theta[0]
        like = 0
        n_ind = len(self.obs)
        if self.param_estimator==True:
            intensity_mat = self.intensity__(X)
        else:
            intensity_mat = self.np_intensity__(X)
        # Z = pd.DataFrame([self.PP_processes[i].Z for i in range(self.n_obs)])
        survival_mat = np.exp(-np.cumsum(intensity_mat, axis=0))
        for i in range(n_ind):
            like += survival_mat[min(int(self.obs[i].jump_time), self.max_time - 1), i] * intensity_mat[
                min(int(self.obs[i].jump_time), self.max_time - 1), i]
        return -like

    def fit(self):
        optimized = scipy.optimize.minimize(self.loss__, x0=self.theta)
        self.theta = optimized["x"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    n_obs = 500
    n_times = 30
    true_theta = [1, -0.4]
    n_rep = 100
    n_starting_point = 10
    estimated_thetas = []
    starting_time = time.time()

    for _ in range(n_rep):
        min_like = 10e999
        obs, X = duration_data_Nielsen_2022(n_ind=n_obs, max_t=n_times, theta=true_theta, n_params = len(true_theta))
        print("Mean event time : ", np.mean([obs[i].jump_time for i in range(n_obs)]), "Censorship rate :",
              1 - np.mean([obs[i].delta for i in range(n_obs)]))
        for _ in range(n_starting_point):
            init_theta = np.random.normal(true_theta, 0.2)
            PM = Parametric_Model(X, obs, init_theta, param=False)
            PM.fit()

            if PM.loss__(PM.theta) < min_like:
                best_theta = PM.theta

        estimated_thetas += [PM.theta]

        print("Current mean estimated theta : ", np.mean(np.array(estimated_thetas), axis=0))
    print("Time : ", time.time() - starting_time)
    df_param = pd.DataFrame(estimated_thetas)

    plt.figure()
    df_param.boxplot()
    plt.show()