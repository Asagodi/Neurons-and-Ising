#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:05:41 2019

@author: abel
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt 

path_to_docs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/Entropy/'
fig = plt.figure()
ax=fig.add_subplot(111)
for n in range(1, 11):
    n_sampled = 10*n
    tot_var_mean = 0.
    for exp_num in range(10):
        with open(path_to_docs+"dir_%s/data_n%s_%s.txt"%(str(n), str(n_sampled), exp_num), "rb") as f:   
           data = pickle.load(f)
           print(data['error'])
           
        
#        with open(path_to_docs+"dir_%s/entro_data_n%s_%s.txt"%(str(n), str(n_sampled), exp_num), "rb") as f:   
#           edatas = pickle.load(f)
#           
        edatas = data['edatas']
        for num_patt in range(edatas.shape[1]):
            ds = edatas[2,num_patt,:]
            sd = edatas[3,num_patt,:]
            diff_list = []
            for i,d in enumerate(ds[:-1]):
                diff = ds[i+1] - d
                diff_list.append(diff)
            diff_array = np.array(diff_list)
            from_val = 0
            for lp, jump in enumerate(np.where(diff_array>0.05)[0]):
                cax = ax.plot(ds[from_val:jump], sd[from_val:jump], color=(0.,num_patt/float(edatas.shape[1]),num_patt/float(edatas.shape[1])))
                from_val = jump+1
               
plt.plot()



path_to_docs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/Entropy/tsao/bs250/'
data_file_list = [fn for fn in os.listdir(path_to_docs) if '.txt' in fn]
min_max_x=50
for n,data_name in enumerate(data_file_list):
    with open(path_to_docs+data_name, "rb") as f:
        data = pickle.load(f)
    edatas=data['edatas']
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(data['ordered_patterns'])
    plt.show()
    fig = plt.figure()
    ax=fig.add_subplot(111)
    for num_patt in range(edatas.shape[1]):
        xs = np.arange(-min_max_x, min_max_x, .04)
        ds = edatas[0,num_patt,:]
        sd = edatas[1,num_patt,:]
        ax.plot(xs, ds, 'k')
        xs = np.arange(min_max_x, -min_max_x, -.04)
        ds = edatas[2,num_patt,:]
        sd = edatas[3,num_patt,:]
        ax.plot(xs, ds, '--r')
    plt.show()
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    for num_patt in range(edatas.shape[1]):
       ds = edatas[2,num_patt,:]
       sd = edatas[3,num_patt,:]
       diff_list = []
       for i,d in enumerate(ds[:-1]):
           diff = ds[i+1] - d
           diff_list.append(diff)
       diff_array = np.array(diff_list)
       from_val = 0
       print(np.where(diff_array>0.05)[0], sd)
       if np.where(diff_array>0.05)[0].size==0:
           cax = ax.plot(ds[from_val:jump], sd[from_val:jump], color=(0.5,num_patt/float(edatas.shape[1]),num_patt/float(edatas.shape[1])))
       for lp, jump in enumerate(np.where(diff_array>0.05)[0]):
           cax = ax.plot(ds[from_val:jump], sd[from_val:jump], color=(0.5,num_patt/float(edatas.shape[1]),num_patt/float(edatas.shape[1])))
           from_val = jump+1
    plt.show()
    
    
    
path_to_docs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/Entropy/LE5/'
maxN=8
for n in range(1, maxN):
    n_sampled = 10*n
    tot_var_mean = 0.
    for exp_num in range(10):
        try:
            fig = plt.figure()
            ax=fig.add_subplot(111)
            with open(path_to_docs+"dir_%s/data_n%s_%s.txt"%(str(n), str(n_sampled), exp_num), "rb") as f:   
                data = pickle.load(f)
                print(n, exp_num, data['error'])
            edatas = data['edatas']
            for num_patt in range(edatas.shape[1]):
                ds = edatas[2,num_patt,:]
                sd = edatas[3,num_patt,:]
                diff_list = []
                for i,d in enumerate(ds[:-1]):
                    diff = ds[i+1] - d
                    diff_list.append(diff)
                diff_array = np.array(diff_list)
                from_val = 0
                if np.where(diff_array>0.025)[0].size==0:
                    cax = ax.plot(ds[from_val:-1], sd[from_val:-1], color=(0.,num_patt/float(edatas.shape[1]),num_patt/float(edatas.shape[1])), label=num_patt)
                for lp, jump in enumerate(np.where(diff_array>0.025)[0]):
                    if lp==0:
                        label=num_patt
                    else:
                        label=None
                    cax = ax.plot(ds[from_val:jump], sd[from_val:jump], color=(0.,num_patt/float(edatas.shape[1]),num_patt/float(edatas.shape[1])), label=label)
                    from_val = jump+1
            plt.legend(title="Pattern number", bbox_to_anchor=(.95, -0.125),fancybox=True, shadow=True, ncol=5)
            plt.title("Entropy landscape of a network of N=100 \n Percentage sampled = %s"%n_sampled)
            plt.ylabel("s(d)")
            plt.xlabel("d")
            plt.plot()
        except:
            None