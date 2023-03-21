# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:28:58 2023

@author: guillaume.flament_ad
"""

import numpy as np
import scipy
import pandas as pd


class KM_est():
    def __init__(self, PP_processes):
        self.PP_processes = PP_processes
        self.n_obs = len(PP_processes)
        self.n_times = max([int(PP_processes[i].jump_time) for i in range(self.n_obs)])

    def fit(self):
        survival_times = pd.DataFrame([int(self.PP_processes[i].jump_time) for i in range(self.n_obs)])
        deltas = pd.DataFrame([self.PP_processes[i].delta for i in range(self.n_obs)])
        Z = pd.DataFrame([self.PP_processes[i].Z for i in range(self.n_obs)])
        at_risk = Z.sum(axis=0)
        lambda_hat = [0]
        KM_est = []
        for t in range(self.n_times):
            n_deaths = np.sum(survival_times == t)
            KM_est += [np.prod([1 - lambda_hat[t] for t in range(len(lambda_hat))])]
            lambda_hat += [n_deaths[0] / at_risk[t] if at_risk[t] > 0 else 0]
        self.surv_func = KM_est
        self.instant_hazard = lambda_hat
        self.cum_hazard = [np.sum(self.instant_hazard[:t]) for t in range(self.n_times)]


class homogenous_PP():
    def __init__(self, intensity):
        self.intensity = intensity
        self.jump_time = 0
        self.delta = 0

    def get_jump_time(self):
        self.jump_time = np.random.exponential(self.intensity)


class Poisson_process():
    """
    See : https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasB.pdf for a generating method.
    """

    def __init__(self, intensity, intensity_cens):
        """
        """

        self.intensity = intensity
        self.max_intensity = np.max(
            np.abs(intensity))
        # Make sur intensity is positive to find an intensity that dominates the time varying intensity.

        self.max_time = len(intensity)
        self.intensity_cens = intensity_cens
        self.max_intensity_cens = np.max(np.abs(intensity_cens))

        self.jump_time = self.get_jump__()
        self.jump_time_cens = self.get_cens__()
        if int(self.jump_time) == self.max_time:
            self.Z = [1 for t in range(int(self.jump_time))]
        else:
            self.Z = [1 for t in range(int(self.jump_time) + 1)] + [0 for t in range(
                len(self.intensity) - int(self.jump_time) - 1)]  # +1, at risk
        self.N = [0 if t != int(self.jump_time) else 1 for t in range(len(self.intensity))]
        self.delta = [1 if self.jump_time < self.jump_time_cens else 0][0]

    def Lambda__(self, t):
        assert t <= self.max_time
        return scipy.integrate.quad(self.intensity, 0, t)[0]

    def get_jump__(self):
        """
        """
        Time = 0
        t_star = 0
        no_time = True
        while no_time:
            hpp = homogenous_PP(self.max_intensity)
            hpp.get_jump_time()
            u = np.random.uniform(0, 1)
            intensity_t_star = self.intensity[np.min([int(hpp.jump_time), len(self.intensity) - 1])]
            t_star += hpp.jump_time - np.log(u) / self.max_intensity
            u = np.random.uniform(0, 1)
            if u < intensity_t_star / self.max_intensity:
                self.jump_time = t_star
                no_time = False
            if t_star > self.max_time:
                no_time = False
                self.jump_time = self.max_time

        return self.jump_time

    def get_cens__(self):
        Time = 0
        t_star = 0
        no_time = True
        while no_time:
            hpp = homogenous_PP(self.max_intensity_cens)
            hpp.get_jump_time()
            u = np.random.uniform(0, 1)
            intensity_t_star = self.intensity[np.min([int(hpp.jump_time), len(self.intensity_cens) - 1])]
            t_star += hpp.jump_time - np.log(u) / self.max_intensity_cens
            if u < intensity_t_star / self.max_intensity_cens:
                self.jump_time_cens = t_star
                no_time = False
            if t_star > self.max_time:
                no_time = False
                self.jump_time_cens = self.max_time

        return self.jump_time_cens

    def __str__(self):
        return "T : " + str(np.round(self.jump_time, 2)) + " C : " + str(
            np.round(self.jump_time_cens, 2)) + " delta : " + str(self.delta)
