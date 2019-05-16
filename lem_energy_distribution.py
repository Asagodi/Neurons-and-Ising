#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:34:39 2019

@author: abel
"""

import numpy as np
from scipy import *
import scipy.io
#import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from simulate_oned import *
import time as tm
from sklearn import manifold
from sklearn.manifold import MDS

path_to_docs = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Head Data N=100/"
#
N, extinp, inh, R = 100, 3, 0.235619449019, 20.0666666667
umax, dtinv, tau, time, ell, alpha, lambda_net = 1., 10, 10, 10**5, 2., 0.25, 13
bsize, shift, scale, dt = 10, 0, 10000, 1

var_over_mean = []
var_list = []
mean_list = [] 
err_list = []

reg_method = "l2"
reg_lambda = .05
max_steps = 2000
h_lambda = 0.5
J_lambda = h_lambda
epsilon = 0.001

s_act = np.loadtxt(path_to_docs + "s_act_head_ang_N100_36.csv", delimiter=',')


#number_of_initial_patterns = 1000
n_list = arange(30, 100, 10)

for n_sampled in n_list:
    print("Number of neurons sampled:", n_sampled)
    sampled_neurons = np.random.choice(range(N), n_sampled, replace=False)
    sampled_neurons = np.sort(sampled_neurons)
    s_act_sampled = s_act[sampled_neurons,:]
     

    h = zeros(n_sampled)
    J = np.random.uniform(-.5, .5, [n_sampled,n_sampled])
    
    h, J, min_av_max_plm = plm_separated(s_act_sampled, max_steps,
                        h, J, h_lambda, J_lambda,
                        reg_method, reg_lambda, epsilon, 1.)
    
    Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    sample_after = 1000 #10**7   
    sample_per_steps = 2*n_sampled # 10 * N
    s_act_inferred = metropolis_mc(h, J, Nsamples, sample_after, sample_per_steps, 1.)
    
    mag_sim = np.average(s_act_sampled, axis=1)
    corrs_sim = calc_correlations(s_act_sampled, mag_sim)
#    corrs3_sim = calc_third_order_corr(s_act_sampled, mag_sim)
    
    #
    mag_inf = np.average(s_act_inferred, axis=1)
    corrs_inf = calc_correlations(s_act_inferred, mag_inf)
#    corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)
    
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    cax = ax.plot([-1, 1], [-1, 1], c="r")
    cax = ax.plot(corrs_sim.flatten(), corrs_inf.flatten(), 'x', c='tab:orange', label="Correlations")
#    cax = ax.plot(corrs3_sim.flatten(), corrs3_inf.flatten(), 'x', c='g', label="Third order correlations")
    cax = ax.plot(mag_sim, mag_inf, 'o', c='b', label="Average magnetization")
    ax.set_xlabel("Data")
    ax.set_ylabel("Inferred")
    ax.legend()
    plt.show()
    
    err = np.sqrt(np.average(np.square(mag_sim - mag_inf)) + np.average(np.square(corrs_sim - corrs_inf)))
    print("Error:", err)
    err_list.append(err)
    
    patterns_gdd, init_final_dict = lem_from_data(h, J, s_act_sampled[:,:], 'random')  
    ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)
    
    #exclude all silent state
    if np.sum(ordered_patterns[-1]) == -float(n_sampled):
        energies = plot_patterns_with_energies(h, J, ordered_patterns[:-1])
    else:
        energies = plot_patterns_with_energies(h, J, ordered_patterns)
    print("Var:", np.var(energies))
    print("Mean:", np.average(energies))
    print("Var/Mean:", np.abs(np.var(energies)/np.average(energies)))
    
    var_list.append(np.var(energies))
    mean_list.append(np.average(energies))
    var_over_mean.append(np.abs(np.var(energies)/np.average(energies)))
    


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(arange(30, 101, 10), np.append(var_over_mean, np.abs(np.var(energies)/np.average(energies))))
ax.set_xlabel("Number of neurons sampled")
ax.set_ylabel("Variance/mean of LEM energies")
plt.show()

#N=20
#Err: 0.00471
#Var: 0.5166
#Mean: -7.11
#Var/Mean: 0.0726582278481

#N=25
#Err: 0.00717
#Var: 0.5348234375
#Mean: -22.39625
#Var/Mean: 0.0238800440922

#N=30
#Err: 0.0710127214515
#Var: 0.375125
#Mean: -22.9
#Var/Mean: 0.0163810043668

#N=50
#Err: 0.0813850027418
#Var: 0.413531360947
#Mean: -48.2838461538
#Var/Mean: 0.00856459031099

#N=75
#Error: 0.362835294756
#Var: 0.220422222222
#Mean: -74.21375
#Var/Mean: 0.0029700995061157803


#N=100
#Var: 0.0500894600592
#Mean: -96.4540384615
#Var/Mean: 0.0005193091016
