#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:46:38 2019

@author: abel
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt 
path_to_docs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/Entropy/LE3/'
colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
fig = plt.figure()
ax=fig.add_subplot(111)
maxN=11
for n in range(1,maxN):
    n_sampled = 10*n
    tot_var_mean = 0.
    min_exp = 0
    min_err = 0.5
    for exp_num in range(10):
        if n!=10:
            try:
                with open(path_to_docs+"dir_%s/data_n%s_%s.txt"%(str(n), str(n_sampled), exp_num), "rb") as f:   
                    data = pickle.load(f)
                
            except:
                None
            if data['error']<min_err:
                min_exp = exp_num
                min_err = data['error']

        else:
            min_exp=0
    with open(path_to_docs+"dir_%s/entro_data_n%s_%s.txt"%(str(n), str(n_sampled), min_exp), "rb") as f:
        edatas = pickle.load(f)
    
    lowestjumpt=10000
    for num_patt in range(edatas.shape[1]):
        ds = edatas[2,num_patt,:]
        diff_list = []
        for i,d in enumerate(ds[:-1]):
            diff = ds[i+1] - d
            diff_list.append(diff)
        diff_array = np.array(diff_list)
        try:
            jump=np.where(diff_array>0.025)[0][0]
        except:
            jump=diff_array.shape[0]
 
        lowestjumpt = min(jump,lowestjumpt)

    totds=set()
    for num_patt in range(edatas.shape[1]):
        ds = edatas[2,num_patt,:]
        for d in ds[0:lowestjumpt]:
            totds.add(d)
    totds=np.array(list(totds))
    totds=sorted(totds)
    all_sds=np.zeros(len(totds))
    for num_patt in range(edatas.shape[1]):
        ds = edatas[2,num_patt,:]
        sd = edatas[3,num_patt,:]
        sdinterp =np.interp(totds,ds,sd)
        all_sds+=sdinterp
    all_sds/=edatas.shape[1]
    label=n_sampled
    cax = ax.plot(totds, all_sds, color=colours[n-1], label=label)
    


plt.legend(title="Percentage sampled", bbox_to_anchor=(.95, -0.125),fancybox=True, shadow=True, ncol=5)
plt.title("Entropy landscape of a network of N=100 averaged over all LEMs")
plt.ylabel("s(d)")
plt.xlabel("d")
plt.plot()