#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:20:42 2019

@author: abel
"""
import numpy as np
from scipy import *
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulate_oned import *
import time as tm

path = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Biased N=50/"

#s_act = np.loadtxt(path+"sim_50_head_ang_5.csv", delimiter=',')

lambdas = [0.1, 0.01, 0.001, 0.0001]
cross_n = len(lambdas)

#s_act = s_act[:,:-20]
dl = s_act.shape[1]

data_chunks = [s_act[:,int(i*dl/cross_n):int((i+1)*dl/cross_n)] for i in range(cross_n)]

max_steps = 250
h_lambda = .5
J_lambda = h_lambda
reg_method = "sign"
epsilon = 0.001


Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
Nflips = 1
sample_after = 1000 #10**7   
sample_per_steps = 100 # 10 * N

errors = zeros((cross_n, cross_n))
for k, reg_lambda in enumerate(lambdas):
    print(reg_lambda)
    for j in range(cross_n):
        data = np.reshape(np.array([data_chunks[i] for i in range(cross_n) if i!=j]), (N, int((cross_n-1)*dl/cross_n)))
        t0 = tm.time()
        h, J = nMF(data)  
        h, J, min_av_max_plm = plm_separated(data, max_steps,
                                h, J, h_lambda, J_lambda,
                                reg_method, reg_lambda, epsilon, 1.)
    #    np.savetxt(path+"h_" + str(reg_method) + "_rl" + str(reg_lambda) + ".csv", h, delimiter=",")
    #    np.savetxt(path+"J_" + str(reg_method) + "_rl" + str(reg_lambda) + ".csv", J, delimiter=",")
        
        mag_sim = np.average(data_chunks[j], axis=1)
        corrs_sim = calc_correlations(data_chunks[j], mag_sim)
        corrs3_sim = calc_third_order_corr(data_chunks[j], mag_sim)
        
        s_act_inferred = metropolis_mc(h, J, Nsamples, Nflips,
                      sample_after, sample_per_steps, 1.)
        mag_inf = np.average(s_act_inferred, axis=1)
        corrs_inf = calc_correlations(s_act_inferred, mag_inf)
        corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)
        
        embedding = MDS(n_components=p, dissimilarity='precomputed')
        X_transformed = embedding.fit_transform(dissimilarities)
        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(111)
        cax = ax.scatter(X_transformed[:num_patts,0], X_transformed[:num_patts,1], marker="x", c="b", label="Expected patterns")
        cax = ax.scatter(X_transformed[num_patts:,0], X_transformed[num_patts:,1], marker="o", c="r", label="LEMs")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.show()

        
        error = np.sqrt(np.sum(np.square(mag_sim - mag_inf)) + np.sum(np.square(corrs_sim - corrs_inf)) + np.sum(np.square(corrs3_sim - corrs3_inf)))
        errors[k, j] = error
    
        print("Time:", tm.time()-t0)

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
cax = ax.plot(lambdas, np.average(errors, axis=1))
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Error")
ax.set_xscale('log')
plt.show()