#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:54:17 2019

@author: abel
"""
import sys
import numpy as np
from scipy import *
import scipy.io
from lec import *
import pickle

def main():
    n_sampled = int(sys.argv[1])*10
    exp_num = int(sys.argv[2])
    s_act = np.loadtxt("/home/abels/lem_experiment/s_act_head_ang_N100_36.csv", delimiter=',') 
    
    sampled_neurons = np.random.choice(range(n_sampled), n_sampled, replace=False)
    sampled_neurons = np.sort(sampled_neurons)
    s_act_sampled = s_act[sampled_neurons,:]
     
    h = zeros(n_sampled)
    J = np.random.uniform(-.5, .5, [n_sampled,n_sampled])
    A = (np.triu(J, k=1) + np.tril(J, k=-1).T)/2
    A += A.T
    J = A
    np.fill_diagonal(J, 0)
    
    max_steps = int(sys.argv[1])*100
    l_rate = 0.5
    Nsamples = 10**4
    sample_after = 10**3
    sample_per_steps = 2*n_sampled
    
    h, J, min_av_max_plm, error_list = boltzmann_learning(s_act_sampled, max_steps, l_rate, h, J, Nsamples, sample_after, sample_per_steps, epsilon=10**-3, T=1.)
    
    s_act_inferred, _ = metropolis_mc(h, J, Nsamples, sample_after, sample_per_steps, 1.)
    
    mag_sim = np.average(s_act_sampled, axis=1)
    corrs_sim = calc_correlations(s_act_sampled, mag_sim)
    corrs3_sim = calc_third_order_corr(s_act_sampled, mag_sim)
    
    mag_inf = np.average(s_act_inferred, axis=1)
    corrs_inf = calc_correlations(s_act_inferred, mag_inf)
    corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)
    
    err = np.sqrt((np.sum(np.square(mag_sim - mag_inf)) + np.sum(np.square(corrs_sim - corrs_inf))+np.sum(np.square(corrs3_sim - corrs3_inf)))/float(n_sampled+n_sampled*n_sampled+n_sampled*n_sampled*n_sampled))
    
    patterns_gdd, init_final_dict = lem_from_data(h, J, s_act_sampled[:,:], 'ordered')
    ordered_patterns = order_patterns(patterns_gdd)
    
    #exclude all-silent and all-active states
    exc_num = 0
    pattern_energies = []
    if abs(np.sum(ordered_patterns[-2])) == float(n_sampled):
        exc_num+=1
        
    if abs(np.sum(ordered_patterns[-1])) == float(n_sampled):
        exc_num+=1
    for i in range(ordered_patterns.shape[0]-exc_num):
        pattern = ordered_patterns[i,:]
        energy = calc_energy([h,J], pattern)
        pattern_energies.append(energy)
    
    numoflems = ordered_patterns.shape[0]

    
    data = {}
    data["s_act_sampled"] = s_act_sampled
    data["h"] = h
    data["J"] = J
    data["numberoflems"] = numoflems
    data["error"] = err
    data["energies"] = pattern_energies
    data["ordered_patterns"] = ordered_patterns
    
    
    with open("/home/abels/lem_experiment/dir_%s/data_n%s_%s.txt"%(sys.argv[1], str(n_sampled), exp_num), "wb") as fp:   #Pickling
       pickle.dump(data, fp)



if __name__ == "__main__":
        main()

    