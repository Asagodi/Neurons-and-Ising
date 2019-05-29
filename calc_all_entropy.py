#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:50:26 2019

@author: abel
"""
import sys
import numpy as np
import pickle
from scipy import *
from lec import *
from mf_entro import *

def main():
    n_sampled = int(sys.argv[1])*10 
    exp_num = int(sys.argv[2])
    
    with open("/home/abels/lem_experiment/dir_%s/data_n%s_%s.txt"%(sys.argv[1], str(n_sampled), exp_num), "rb") as f:   
       data = pickle.load(f)
       
    h = data["h"]
    J = data["J"]
    ordered_patterns = data["ordered_patterns"]
    
    J_ = make_flat_J(J)
    N = h.shape[0]
    V_node, node, F_node = setup_nodes(N)
    
    xs = arange(-10., 10., .04)
    edatas = zeros([4, ordered_patterns.shape[0], xs.shape[0]])
    
    max_steps=500
    for num_patt in range(0,ordered_patterns.shape[0]):
        m_ia = initial_message(N)
        ref_sigma = ordered_patterns[num_patt] 
        ds = []
        ds = []
        sd = []
        xs = arange(-10., 10., .04)
        for x in xs:
            h_ = h + x * ref_sigma
            m_ia = iteration(m_ia, h_, J_, max_steps, V_node, F_node, delta=10**-3)
            mi = comput_mag_corre(m_ia, h_, J_, max_steps, V_node, F_node)
            q = np.dot(ref_sigma,mi)/float(N)
            s = distance_entropy(m_ia, h, J_, x, ref_sigma, V_node, F_node)
            sd.append(s-x*q)
            q = np.dot(ref_sigma, mi)/N
            d = (1-q)/2.
            ds.append(d)
#        np.savetxt("/home/abels/lem_experiment/hop_dist_up_n%s_%s_patt%s.csv"%(n_sampled, exp_num, num_patt), ds, delimiter=",")
#        np.savetxt("/home/abels/lem_experiment/hop_entr_up_n%s_%s_patt%s.csv"%(n_sampled, exp_num, num_patt), sd, delimiter=",")
        edatas[0,:] = ds
        edatas[1,:] = sd
        ds = []
        sd = []
        xs = arange(10., -10., -.04)
        for x in xs:
            h_ = h + x * ref_sigma
            m_ia = iteration(m_ia, h_, J_, max_steps, V_node, F_node, delta=10**-3)
            mi = comput_mag_corre(m_ia, h_, J_, max_steps, V_node, F_node)
            q = np.dot(ref_sigma,mi)/float(N)
            s = distance_entropy(m_ia, h, J_, x, ref_sigma, V_node, F_node)
            sd.append(s-x*q)
            q = np.dot(ref_sigma, mi)/N
            d = (1-q)/2.
            ds.append(d)
#        np.savetxt("/home/abels/lem_experiment/hop_dist_down_n%s_%s_patt%s.csv"%(n_sampled, exp_num, num_patt), ds, delimiter=",")
#        np.savetxt("/home/abels/lem_experiment/hop_entr_down_n%s_%s_patt%s.csv"%(n_sampled, exp_num, num_patt), sd, delimiter=",")
        edatas[2,:] = ds
        edatas[3,:] = sd
        
        np.savetxt("/home/abels/lem_experiment/dir_%s/entro_data_n%s_%s_patt%s.csv"%(sys.argv[1], str(n_sampled), exp_num, str(num_patt)), edatas, delimiter=",")

if __name__ == "__main__":
        main()

    