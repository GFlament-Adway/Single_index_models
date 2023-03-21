import numpy as np
from gen_data import duration_data_Nielsen_2022
import scipy
import pandas as pd


class Parametric_Model:
    def __init__(self, X, obs, init_theta):
        """
        :param X: Covariates.
        :param Y: Duration times.
        :param Delta: indicator of censorship.
        """
        self.X = X
        self.obs = obs
        self.max_time = X.shape[0]
        self.theta = init_theta

    def intensity__(self, t, x_new):
        return np.exp(np.dot(x_new[np.max([int(t) - 1, 0])], self.theta) - 2)/15  # First time indexed by 0.

    def survival__(self, max_time, x):
        return np.exp(-np.sum([self.intensity__(t, x) for t in range(int(max_time))]))

    def loss__(self, theta):
        if type(theta) == np.ndarray:
            theta = theta[0]
        log_like = 0
        n_ind = len(self.obs)
        intensity_mat = np.exp(-np.dot(theta, X))
        # Z = pd.DataFrame([self.PP_processes[i].Z for i in range(self.n_obs)])
        survival_mat = np.exp(-np.cumsum(intensity_mat, axis=0))
        for i in range(n_ind):
            log_like += survival_mat[min(int(self.obs[i].jump_time) - 1, self.max_time - 1), i] * intensity_mat[
                min(int(self.obs[i].jump_time) - 1, self.max_time - 1), i]
        return -log_like

    def fit(self):
        optimized = scipy.optimize.minimize(self.loss__, x0=self.theta)
        print(optimized)


if __name__ == "__main__":
    n_obs = 3000
    n_times = 50
    true_theta = 2
    init_theta = 1
    obs, X = duration_data_Nielsen_2022(n_obs, n_times, theta=true_theta)

    print("Mean event time : ", np.mean([obs[i].jump_time for i in range(n_obs)]))
    print("Mean censorship : ", np.mean([obs[i].delta for i in range(n_obs)]))


    # print(obs)
    PM = Parametric_Model(X, obs, init_theta)
    print(PM.loss__(0), PM.loss__(-0.53), PM.loss__(-1), PM.loss__(-2.5), PM.loss__(-3), PM.loss__(4), PM.loss__(-20))
    PM.fit()
