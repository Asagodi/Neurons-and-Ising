#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:50:26 2019

@author: abel
"""
import os
import sys
import numpy as np
import pickle
from scipy import *
from lec import *

def main():
    path = os.getcwd()
    n = int(sys.argv[1])
    exp_num = sys.argv[2]
    n_sampled = 10*n
#    data_file_list = [fn for fn in os.listdir(path) if '.txt' in fn]
#    data_name = data_file_list[n]
    
    with open(path+"/dir_%s/data_n%s_%s.txt"%(sys.argv[1], str(n_sampled), exp_num), "rb") as f:   
       data = pickle.load(f)
#    with open(path+"/%s"%(data_name), "rb") as fp:
#       data = pickle.load(fp)
       
    h = data["h"]
    J = data["J"]
    ordered_patterns = data["ordered_patterns"]
    
    J_ = make_flat_J(J)
    N = h.shape[0]
    V_node, node, F_node = setup_nodes(N)
    
    min_max_x=10
    stepx=0.04
    data['min_max_x']=min_max_x
    data['stepx']=stepx
    xs = arange(-min_max_x, min_max_x, stepx)
    edatas = zeros([4, ordered_patterns.shape[0], xs.shape[0]])
    fes = zeros([ordered_patterns.shape[0], xs.shape[0]])
    egs = zeros([ordered_patterns.shape[0], xs.shape[0]])
    
    max_steps=1000
    for num_patt in range(0,ordered_patterns.shape[0]):
        m_ia = initial_message(N)
        ref_sigma = ordered_patterns[num_patt] 
        ds = []
        sd = []
        xs = arange(-min_max_x, min_max_x, stepx)
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
        xs = arange(min_max_x, -min_max_x, -stepx)
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
    
    data["edatas"] = edatas
    data['fes'] = fes
    data['egs'] = egs
    with open(path+"/dir_%s/data_n%s_%s.txt"%(sys.argv[1], str(n_sampled), exp_num), "wb") as fp:   #Pickling
       pickle.dump(data, fp)
#    with open(path+"/%s"%(data_name), "wb") as fp:
#       pickle.dump(data, fp)

if __name__ == "__main__":
        main()

    