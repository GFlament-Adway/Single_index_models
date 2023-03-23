import pandas as pd

from gen_data import get_data, first_test, sigmoid
import numpy as np
from optim import Kernel_Reg
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from statsmodels.nonparametric import kernel_regression

def test_bandwidth():
    time_start_own_package = time.time()
    init_betas = [3, 1]
    n_ind = 1000
    X, Y, betas = get_data(n=n_ind, true_beta=init_betas, link_func=first_test, Y_type="c")

    X_low = X.dot(init_betas).quantile(0.0025)
    X_up = X.dot(init_betas).quantile(0.975)

    integral_interval = np.linspace(X_low, X_up, num=1000)
    KR = Kernel_Reg(X, Y, init_betas, 1.06 * np.std(X.dot(betas).to_numpy()) * n_ind ** (- 1. / (4 + len(X.columns))))
    KR.fit()
    time_end_own_package = time.time()
    print("init bw : ", 16 * np.std(X.dot(betas).to_numpy()) * n_ind ** (- 1. / (4 + len(X.columns))))
    print("bandwidth found by own package : ", KR.bw)
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
    plt.plot(integral_interval, fitted_values, label="Own package", linestyle=(0, (1,1)))
    plt.plot(integral_interval, fitted_mod[0], label="Statsmodel package", linestyle=(0, (1, 3)))
    plt.legend()
    plt.plot()
    plt.show()
    
    
def test_convergence():
    """
    Do not use it : results are ok.
    :return:
    """
    np.random.seed(12345)
    n_test = 3
    n_sample = 10
    n_ind = 100
    reg_type = "c"
    link_func = first_test
    true_beta = [3, 1]
    starting_beta = [0, 0]

    X, Y, betas = get_data(n=1000, true_beta=true_beta, link_func=link_func, Y_type=reg_type)

    X_low = X.dot(true_beta).quantile(0.0025)
    X_up = X.dot(true_beta).quantile(0.975)

    integral_interval = np.linspace(X_low, X_up, num=10000)

    starting_betas = [[beta + np.random.normal(0, 0.5, 1)[0] for beta in starting_beta] for _ in range(n_test)]

    df_e_betas = pd.DataFrame(columns=[r"$\beta_{k}$".format(k=k) for k in range(len(starting_betas[0]))])
    df_ise = pd.DataFrame(columns=["ise"])
    df_scaled_betas = pd.DataFrame(columns=[r"$\beta_{k}$".format(k=k) for k in range(len(starting_betas[0]))])
    # best_KR_model = 0

    for i in range(n_sample):
        print("Optimisation for sample :", i)
        X, Y, betas = get_data(n=n_ind, true_beta=true_beta, link_func=link_func, Y_type=reg_type)
        best_loss = 1e99
        df_e_betas.loc[len(df_e_betas.index)] = [0 for _ in range(len(starting_betas[0]))]
        for k in range(n_test):
            # print("Iteration : ", k)
            KR = Kernel_Reg(X, Y, starting_betas=starting_betas[k], n_it=1, reg_type=reg_type)
            KR.fit()
            if KR.loss < best_loss:
                df_e_betas.loc[i] = KR.betas
                df_scaled_betas.loc[i] = KR.scaled_betas
                best_loss = KR.loss
                best_KR_model = KR

        # Estimating \hat{\phi}
        #model = KernelReg(Y["Y"], X.dot(df_scaled_betas.loc[i].values).to_numpy(), var_type="c")
        # pred = model.fit()[0]

        #pred = model.fit(integral_interval)[0]

        ise = KR.ise(pred, link_func(integral_interval), integral_interval)
        df_ise.loc[i] = ise
        df_scaled_betas.loc[i] = KR.scaled_betas
        print("ISE : ", ise)
        print("best solution for that sample : ", KR.loss)
        print("betas : ", df_e_betas.loc[i].values)
        print("scaled betas :", df_scaled_betas.loc[i].values)
        print("h", KR.bw)

    if reg_type == "c":
        fig = plt.figure()
        df_ise.boxplot(showfliers=False)
        fig.savefig("images/generated_images/boxplot_ise_{n_sample}.png".format(n_sample=n_sample))

    fig = plt.figure()
    df_scaled_betas.boxplot(showfliers=False)
    fig.savefig("images/generated_images/scaled_betas.png".format(n_sample=n_sample))

if __name__ == "__main__":
    test_bandwidth()    