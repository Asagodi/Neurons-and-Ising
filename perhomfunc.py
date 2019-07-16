#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:54:28 2019

@author: abel
"""

from collections import Counter
import numpy as np
from scipy import *
import pickle
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import itertools
from lec import *
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
import networkx as nx


path_to_buzs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/Entropy/buzsaki/'

with open(path_to_buzs+"data_resam.txt", "rb") as fp:
    data = pickle.load(fp)

h = data['h']
J = data['J']
N = h.shape[0]
all_states = np.array([list(seq) for seq in itertools.product([-1,1],repeat=N)])


exp_patts = make_expected_patterns(N, 1, 6)
h = zeros(N)
J = make_hopfield_weights(exp_patts)

all_states_num = all_states.shape[0]

#step 1
all_energies = np.array(calc_energy_list([h,J], all_states))


#patterns_gdd = all_states[np.where(np.array(all_energies) < min(all_energies)+4)[0],:]



#step 3
nsteps = 2
step = .75
ulist = []
aroundval = min(all_energies) + step
mx = step
for i in range(nsteps):
    patterns_gdd = all_states[np.where( abs(np.array(all_energies) - aroundval) < mx)[0],:]
    ulist.append(patterns_gdd)
    aroundval += step
    



#step 4, clustering
E = 1/12


clustlist = []
for i in range(nsteps): 
    y = pdist(ulist[i], 'hamming')
    Z = single(y)
    clinkmat = fcluster(Z, E, criterion='distance')
#    clinkmat = scipy.cluster.hierarchy.linkage(pdist, method='single',
#                                               metric='hamming')
    clustlist.append(clinkmat)
    
    


#step 5, construct simplicial complex
G = nx.Graph()
for i, ui in enumerate(ulist):
    uindx = clustlist[i]
    uniqueValues, occurCount = np.unique(uindx, return_counts=True)
    for j,uv in enumerate(uniqueValues):
        G.add_node(str(i)+"."+str(j))
        ##check for overlap between clusters

        for patt in ui[np.where(uindx==uv)[0],:]:
#            print(patt.shape)
#            plot_single_pattern(patt)
            for m, um in enumerate(ulist):
                
                for n, patt_un in  enumerate(um):
#                    plot_single_pattern(patt_un)
#                    print(i, m, n, np.all(patt==patt_un))
                    if np.all(patt==patt_un) and i!=m:
                        l = clustlist[m][n] - 1
                        G.add_edge(str(i)+"."+str(j), str(m)+"."+str(l))
    