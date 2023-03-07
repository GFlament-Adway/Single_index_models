import pandas as pd

from gen_data import get_data, f
import numpy as np
from optim import Kernel_Reg
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
import multiprocessing as mp

if __name__ == "__main__":
    np.random.seed(123456)
    n_test = 5
    n_sample = 100
    n_ind = 100
    true_beta = [3,1]
    starting_beta = [0, 0]

    X,Y, betas, c = get_data(n = 100000, true_beta=true_beta)
    X_low = X.dot(true_beta).quantile(0.0025)
    X_up = X.dot(true_beta).quantile(0.975)
    
    integral_interval= np.linspace(X_low, X_up, num=10000)
    
    starting_betas = [[beta + np.random.normal(0, 0.5, 1)[0] for beta in starting_beta] for _ in range(n_test)]

    df_e_betas = pd.DataFrame(columns=[r"$\beta_{k}$".format(k=k) for k in range(len(starting_betas[0]))])
    df_ise = pd.DataFrame(columns=["ise"])
    df_scaled_betas = pd.DataFrame(columns=[r"$\beta_{k}$".format(k=k) for k in range(len(starting_betas[0]))])
    #best_KR_model = 0
    
    
    for i in range(n_sample):
        print("Optimisation for sample :", i)
        X, Y, betas, c = get_data(n=n_ind, true_beta = true_beta)
        best_loss = 1e99
        df_e_betas.loc[len(df_e_betas.index)] = [0 for _ in range(len(starting_betas[0]))]
        for k in range(n_test):
            #print("Iteration : ", k)
            KR = Kernel_Reg(X, Y, starting_betas=starting_betas[k], n_it=1)
            KR.fit()
            if KR.loss < best_loss:
                df_e_betas.loc[i] = KR.betas     
                best_loss = KR.loss
                best_KR_model = KR
        
          
    
        #Estimating \hat{\phi}
        model = KernelReg(Y["Y"], X.dot(betas).to_numpy(), var_type="c")
        #pred = model.fit()[0]
        
        pred = model.fit(integral_interval)[0]
        ise = KR.ise(pred, f(integral_interval), integral_interval)
        df_ise.loc[i] = ise
        df_scaled_betas.loc[i] = KR.scaled_betas 
        print("ISE : ", ise)        
        print("best solution for that sample : ", KR.loss)
        print("betas : ", df_e_betas.loc[i].values)
        print("scaled betas :", KR.scaled_betas)
        print("h", KR.bw)

    fig = plt.figure()    
    df_ise.boxplot(showfliers = False)
    fig.savefig("C:/Users/guillaume.flament_ad/Desktop/non_param_optim/images/generated_images/boxplot_ise_{n_sample}.png".format(n_sample=n_sample))


    fig = plt.figure()
    df_scaled_betas.boxplot(showfliers = False)
    fig.savefig("C:/Users/guillaume.flament_ad/Desktop/non_param_optim/images/generated_images/scaled_betas.png".format(n_sample = n_sample))

    