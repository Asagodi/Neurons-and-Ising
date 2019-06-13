#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:57:21 2019

@author: abel
"""
import os
import sys
import pickle
import numpy as np
from lec import *

def main():
    path = os.getcwd()
    num = int(sys.argv[1])
    data_file_list = [fn for fn in os.listdir(path) if '.csv' in fn]
    data_name = data_file_list[num]
    s_act = np.loadtxt(data_name, delimiter=',') 
    
    h,J =nMF(s_act)
    N = h.shape[0]
    
    max_steps = 10**4
    l_rate = 0.1
    Nsamples = 10**4
    sample_after = 10**3
    sample_per_steps = 2*N
    epsilon = 10**-2
    
    h, J, min_av_max_plm, error_list = boltzmann_learning(s_act, max_steps, l_rate, h, J, Nsamples, sample_after, sample_per_steps, epsilon=epsilon, T=1.)
    
    s_act_inferred, _ = metropolis_mc(h, J, Nsamples, sample_after, sample_per_steps, 1.)
    
    mag_sim = np.average(s_act, axis=1)
    corrs_sim = calc_correlations(s_act, mag_sim)
    corrs3_sim = calc_third_order_corr(s_act, mag_sim)
    
    mag_inf = np.average(s_act_inferred, axis=1)
    corrs_inf = calc_correlations(s_act_inferred, mag_inf)
    corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)
    
    err = np.sqrt((np.sum(np.square(mag_sim - mag_inf)) + np.sum(np.square(corrs_sim - corrs_inf))+np.sum(np.square(corrs3_sim - corrs3_inf)))/float(N+N*N+N*N*N))
    
    patterns_gdd, init_final_dict = lem_from_data(h, J, s_act[:,:], 'ordered')
    ordered_patterns = order_patterns(patterns_gdd)
    
    pattern_energies = []
    for i in range(ordered_patterns.shape[0]):
        pattern = ordered_patterns[i,:]
        energy = calc_energy([h,J], pattern)
        pattern_energies.append(energy)
    
    numoflems = ordered_patterns.shape[0]
    
    data = {}
    data["file_name"]=data_name
    data["s_act"] = s_act
    data["h"] = h
    data["J"] = J
    data["numberoflems"] = numoflems
    data["error"] = err
    data["energies"] = pattern_energies
    data["ordered_patterns"] = ordered_patterns
    data["min_av_max_plm"] = min_av_max_plm
    data["error_list"] = error_list
    data["s_act_inferred"] = s_act_inferred
    
    with open(path+"/all_%s.txt"%(data_name[:-4]), "wb") as fp:
        pickle.dump(data, fp)

    J_ = make_flat_J(J)
    V_node, node, F_node = setup_nodes(N)
    
    min_max_x=100
    xstep = 0.01
    xs = arange(-min_max_x, min_max_x, xstep)
    edatas = zeros([4, ordered_patterns.shape[0], xs.shape[0]])
    fes = zeros([ordered_patterns.shape[0], xs.shape[0]])
    egs = zeros([ordered_patterns.shape[0], xs.shape[0]])
    
    max_steps=1000
    for num_patt in range(0,ordered_patterns.shape[0]):
        m_ia = initial_message(N)
        ref_sigma = ordered_patterns[num_patt] 
        ds = []
        sd = []
        xs = arange(-min_max_x, min_max_x, xstep)
        for x in xs:
            h_ = h + x * ref_sigma
            m_ia = iteration(m_ia, h_, J_, max_steps, V_node, F_node, delta=10**-3)
            mi = comput_mag_corre(m_ia, h_, J_, max_steps, V_node, F_node)
            q = np.dot(ref_sigma,mi)/float(N)
            s, fe, eg = distance_entropy(m_ia, h, J_, x, ref_sigma, V_node, F_node)
            sd.append(s-x*q)
            q = np.dot(ref_sigma, mi)/N
            d = (1-q)/2.
            ds.append(d)
        edatas[0,num_patt,:] = ds
        edatas[1,num_patt,:] = sd
        ds = []
        sd = []
        fs = []
        gs = []
        xs = arange(min_max_x, -min_max_x, -xstep)
        for x in xs:
            h_ = h + x * ref_sigma
            m_ia = iteration(m_ia, h_, J_, max_steps, V_node, F_node, delta=10**-3)
            mi = comput_mag_corre(m_ia, h_, J_, max_steps, V_node, F_node)
            q = np.dot(ref_sigma,mi)/float(N)
            s, fe, eg = distance_entropy(m_ia, h, J_, x, ref_sigma, V_node, F_node)
            sd.append(s-x*q)
            fs.append(fe)
            gs.append(eg)
            q = np.dot(ref_sigma, mi)/N
            d = (1-q)/2.
            ds.append(d)
        edatas[2,num_patt,:] = ds
        edatas[3,num_patt,:] = sd
        fes[num_patt,:] = fs
        egs[num_patt,:] = gs
        
    data['edatas'] = edatas
    data['fes'] = fes
    data['egs'] = egs
    with open(path+"/all_%s.txt"%(data_name[:-4]), "wb") as fp:
        pickle.dump(data, fp)


if __name__ == "__main__":
        main()