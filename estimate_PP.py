import numpy as np
from gen_data import duration_data_Nielsen_2022
import scipy
import pandas as pd
from KernelReg import NW_hazard


class Parametric_Model:
    def __init__(self, X, obs, init_theta, bw=1, param=True):
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
        self.deltaT = 1

    def intensity__(self, theta):
        return np.exp(np.matmul(self.X, theta))

    def np_intensity__(self, theta):
        df_jump_time = pd.DataFrame([self.obs[i].jump_time for i in range(self.n_obs)]).T
        NP_intensity_est = pd.DataFrame()
        for t in range(self.max_time):
            df_X = np.array(self.X[t, :, :])
            NP_intensity_est = pd.concat([NP_intensity_est, NW_hazard(df_X, df_jump_time, theta, self.bw)])
        return NP_intensity_est.to_numpy()

    def loss__(self, theta):
        log_loss = 0
        n_ind = len(self.obs)
        if self.param_estimator == True:
            intensity_mat = self.intensity__(theta)
        else:
            intensity_mat = self.np_intensity__(theta)

        Z = np.array([self.obs[i].Z for i in range(self.n_obs)])
        N = np.array([self.obs[i].N for i in range(self.n_obs)])

        #survival_mat = np.exp(-np.cumsum(intensity_mat, axis=0))

        """
        for i in range(n_ind):
            log_loss += survival_mat[min(int(self.obs[i].jump_time), self.max_time - 1), i] * intensity_mat[
                min(int(self.obs[i].jump_time), self.max_time - 1), i]
        """



        #print("test 1 1 :", np.diag(np.matmul(N, np.exp(-np.cumsum(np.multiply(Z, intensity_mat.T), axis=0)).T)))
        #print("test 2 2 : ", [survival_mat[min(int(self.obs[i].jump_time), self.max_time - 1), i] for i in range(self.n_obs)])
        log_loss_2 = -np.prod(np.multiply(np.diag(np.matmul(N, np.exp(-np.cumsum(np.multiply(Z, intensity_mat.T), axis=0)).T)), np.diag(np.matmul(N, intensity_mat))))

        #print(log_loss, log_loss_2)
        #print("loss : ", log_loss_2)

        return log_loss_2

    def fit(self):
        optimized = scipy.optimize.minimize(self.loss__, x0=np.array(self.theta).reshape(len(self.theta),))
        self.theta, self.bw = self.scale__(optimized["x"])

    def scale__(self, theta):
        return [1] + [t/theta[0] for t in theta[1:]], np.abs(theta[0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    n_obs = 300
    n_times = 100
    true_theta = [1, -0.3]
    n_rep = 100
    n_starting_point = 10
    estimated_thetas = []
    starting_time = time.time()

    for _ in range(n_rep):
        min_like = 10e999
        obs, X = duration_data_Nielsen_2022(n_ind=n_obs, max_t=n_times, theta=true_theta, n_params=len(true_theta))
        print("Mean event time : ", np.mean([obs[i].jump_time for i in range(n_obs)]), "Censorship rate :",
              1 - np.mean([obs[i].delta for i in range(n_obs)]))
        for _ in range(n_starting_point):
            init_theta = np.random.normal(true_theta, 0.5)
            PM = Parametric_Model(X, obs, [t for t in init_theta], param=False)
            PM.fit()
            #assert not np.array_equal(PM.theta, init_theta)

            if PM.loss__(PM.theta) < min_like:
                best_theta = PM.theta

        estimated_thetas += [PM.theta]
        #print("Estimated theta for this step :", estimated_thetas)
        print("Current median estimated theta : ", np.median(np.array(estimated_thetas), axis=0))
    print("Time : ", time.time() - starting_time)
    df_param = pd.DataFrame(estimated_thetas)

    plt.figure()
    df_param.boxplot()
    plt.show()
