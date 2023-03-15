import numpy as np
from scipy import stats, optimize
import pandas as pd
from KernelReg import kernel, generate_kernel_matrix, NW


class Kernel_Reg:
    def __init__(self, X, Y, starting_betas = None, bw = None, n_it = None, reg_type="c"):
        """
        :param X: dataframe that contains the features.
        :param Y: dataframe that contains Y values.
        :param starting_betas: betas that needs to be estimated
        :param bw: Float that contains the bandwidth of the kernel.
        :param reg_type: integer 'cont' for linear regression, 'binary' for categorial
        """
        self.X = X
        self.Y = Y
        self.reg_type = reg_type
        if starting_betas is None:
            starting_betas = [np.random.uniform(0, 1) for _ in range(len(self.X.columns))]
        self.betas = starting_betas
        if bw is None:
            bw = 0.1
        self.bw = bw
        if n_it is None:
            n_it = 10
        self.n_it = n_it

    def loss__(self, y, x):
        if self.reg_type == "c":
            return (y-x)**2

        if self.reg_type == "binary":
            return -(np.log(x)*y + (1 - np.log(x))*(1-y))
    
    def evaluate(self, x_new):
        print(self.bw)
        print(self.betas)
        return NW(x_new, self.Y, self.betas, self.bw)
        
    
    def loss_function(self, betas):
        dist = self.loss__(self.Y.iloc[:,0], NW(self.X, self.Y, betas, self.bw)).sum()
        #print(dist)
        return dist
    
    def scale__(self):
        self.scaled_betas = [np.sign(self.betas[0])*beta*self.bw for beta in self.betas]
        
    def fit(self):
        for _ in range(self.n_it):
            betas = optimize.minimize(self.loss_function, self.betas)
            #print(betas)
            self.loss = betas["fun"]
            self.betas = betas["x"]
            self.bw = 1/np.abs(self.betas[0])
            self.scale__()
            #print("h : ", 1/np.abs(self.betas[0]))
            #print("Solution found : ", self.betas)

    def ise(self, pred, true, inter):
        return np.sum([(inter[i+1] - inter[i])*(pred[i] - true[i])**2 for i in range(len(inter) - 1)])
        

