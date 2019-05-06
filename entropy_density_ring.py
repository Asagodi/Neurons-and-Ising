#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:27:21 2019

@author: abel
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import *
from lec import *


path_to_docs = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Head Data N=100/"
h = np.loadtxt(path_to_docs+"h_N100l2_bsize10_rlam01.csv", delimiter=',')
J = np.loadtxt(path_to_docs+"J_N100l2_bsize10_rlam01.csv", delimiter=',')
J_ = make_flat_J(J)
s_act = np.loadtxt(path_to_docs + "s_act_head_ang_N100_36.csv", delimiter=',')
V_node, node, F_node = setup_nodes(N)

number_of_initial_patterns = 5000
patterns_gdd, init_final_dict = lem_from_data(h, J, s_act[:,:number_of_initial_patterns], 'random')  
ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)


num_patt = 1
###make d-x plot
N = h.shape[0]
max_steps=50
for num_patt in range(2,ordered_patterns.shape[0]-1):
    m_ia = initial_message(N)
    ref_sigma = ordered_patterns[num_patt]
    ds = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ds = []
    sd = []
    xs = arange(-3., 3., 0.04)
    for x in xs:
        print(x)
        h_ = h + x * ref_sigma
        m_ia = iteration(m_ia, h_, J_, max_steps, delta=10**-2)
        mi = comput_mag_corre(m_ia, h_, J_, max_steps)
        q = np.dot(ref_sigma,mi)/float(N)
        s = distance_entropy(m_ia, h, J_, x, ref_sigma)
        sd.append(s-x*q)
        q = np.dot(ref_sigma, mi)/N
        d = (1-q)/2.
        ds.append(d)
    cax = ax.plot(xs, ds, 'k', label='increasing x')
    ds = []
    sd = []
    xs = arange(3., -3., -0.04)
    for x in xs:
    #    print(x)
        h_ = h + x * ref_sigma
        m_ia = iteration(m_ia, h_, J_, max_steps, delta=10**-2)
        mi = comput_mag_corre(m_ia, h_, J_, max_steps)
        q = np.dot(ref_sigma,mi)/float(N)
        s = distance_entropy(m_ia, h, J_, x, ref_sigma)
        sd.append(s-x*q)
        q = np.dot(ref_sigma, mi)/N
        d = (1-q)/2.
        ds.append(d)
    cax = ax.plot(xs, ds, '--r', label='decreasing x')
    ax.set_ylabel("d")
    ax.set_xlabel("x")
    ax.set_title(r"$\beta$={0:0.1f}".format(beta))
    ax.legend()
    plt.show()
    
    np.savetxt(path_to_docs+"distances_pattern"+str(num_patt+1)+".csv", ds, delimiter=",")
    np.savetxt(path_to_docs+"entropies_pattern"+str(num_patt+1)+".csv", sd, delimiter=",")


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(ds, sd, '--r')
ax.set_ylabel("s(d)")
ax.set_xlabel("d")
ax.set_title("Entropy")
plt.show()

diff_list = []
for i,d in enumerate(ds[:-1]):
    diff = ds[i+1] - d
    diff_list.append(diff)
diff_array = np.array(diff_list)

#plot only without jumps
fig = plt.figure()
ax = fig.add_subplot(111)
from_val = 0
for jump in np.where(diff_array>0.05)[0]:
    print(from_val,jump)
    cax = ax.plot(ds[from_val:jump], sd[from_val:jump], '--r')
    from_val = jump+1
ax.set_ylabel("s(d)")
ax.set_xlabel("d")
ax.set_title("Entropy")
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(diff_list)
ax.set_ylabel("Diff")
ax.set_xlabel("d")
ax.set_title("Differences")
plt.show()
    
