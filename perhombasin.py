#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:52:34 2019

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



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:01:18 2019

@author: abel
"""

import networkx as nx
import itertools
import numpy as np
from lec import *
from scipy import *
import matplotlib.pyplot as plt
import pickle


N=12 #max 10?

all_states = np.array([list(seq) for seq in itertools.product([-1,1],repeat=N)])

exp_patts = make_expected_patterns(N, 1, int(N/2)) #0.5
h = zeros(N)
J = make_hopfield_weights(exp_patts)

all_energies = np.array(calc_energy_list([h,J], all_states))

#take subset of all states (f^-1  of neighbourhoods around minima)
all_states = all_states[np.where(np.array(all_energies) < min(all_energies)+5)[0],:]
all_energies = np.array(calc_energy_list([h,J], all_states))

all_states_num = all_states.shape[0] # == 2**N

ham_matrix = zeros((all_states_num, all_states_num))

for i,patt1 in enumerate(all_states):
    for j,patt2 in enumerate(all_states[i:]):
        if hamming_distance(patt1, patt2) <= 1.:
            ham_matrix[i,i+j] = 1.
            ham_matrix[i+j,i] = 1.
            
            

can_matrix = zeros((all_states_num, all_states_num))
for i,en1 in enumerate(all_energies):
    for j,en2 in enumerate(all_energies[i+1:]):
        if ham_matrix[i,i+j+1] == 1. :
            if np.round(en2,13)<np.round(en1,13):
                can_matrix[i,i+j+1] = 1
#            elif en2==en1:
#                can_matrix[i,i+j+1] = 1
#                can_matrix[i+j+1,i] = 1
            elif en2==en1:
                None
            else:
                can_matrix[i+j+1,i] = 1

DG=nx.DiGraph(can_matrix)

#make this faster
lem_matrix = zeros((all_states_num,all_states_num))
for i in range(all_states_num):
    for j in range(all_states_num):
        try: 
            nx.shortest_path(DG, i, j)
            if i!=j:
                lem_matrix[i,j] = 1
        except:
            None
            
            
all_lems = zeros(all_states_num, dtype=bool)
for j, patt in enumerate(all_states):
    
    en_list = []
    for i in range(N):
        neigh = np.array(patt, copy=True)
        neigh[i] = -neigh[i]
        en_list.append(calc_energy([h,J],neigh))
    if min(en_list) >= all_energies[j]:
        all_lems[j] = 1
        
#Rounding fricking errors
#all_lems = zeros(all_states_num)
#for j, patt in enumerate(all_states):
#    
#    en_list = []
#    for i in range(N):
#        neigh = np.array(patt, copy=True)
#        neigh[i] = -neigh[i]
#        en_list.append(calc_energy([h,J],neigh))
#    if np.round(min(en_list),13) >= np.round(all_energies[j], 13):
#        all_lems[j] = 1
#    print(j, min(en_list), all_energies[j])

#for each state the LEMs one can reach by GDD
list_of_lems = [] 
for i in range(all_states_num):
    if np.where(lem_matrix[i,:]==1.)[0].size == 0:
        
        list_of_lems.append(np.reshape(all_states[i,:], (1,N)))
    else:
        
        lems_indx_from_here = [a for a in list(np.where(lem_matrix[i,:]==1.)[0]) if a in list(np.where(all_lems==1.)[0])]
        if len(lems_indx_from_here) == 1:
            list_of_lems.append(np.reshape(all_states[lems_indx_from_here,:], (1,N)))
        else:
            list_of_lems.append(all_states[lems_indx_from_here,:])
            
            
            
#order#?
ordered_patterns = order_patterns(all_states[all_lems,:])

#get basins
list_of_basins = []  #one for each LEM
for lem in ordered_patterns:
    basin = []
    for i, patterns in enumerate(list_of_lems):
        for patt in patterns:
            if np.all(lem==patt):
                
                basin.append(all_states[i])
    
    list_of_basins.append(np.array(basin))
    
    
G = nx.Graph()
for i, ui in enumerate(list_of_basins):
    G.add_node(str(i+1))
    for j, uj in enumerate(list_of_basins):
        
        ##check for overlap between clusters (basins)
        for patti in ui:
            for pattj in uj:
                if np.all(patti==pattj) and i!=j:
                    G.add_edge(str(i+1), str(j+1))
    
plt.figure(figsize=(7,7))   
nx.draw_spectral(G, with_labels = True)
plt.show()


plt.figure()
plt.imshow(ordered_patterns)
plt.xticks(range(0,N,2), range(1,N+1, 2))
plt.yticks(range(0,N,2), range(1,N+1, 2))
plt.xlabel("Neuron")
plt.ylabel("LEM")
plt.title("LEMs")
plt.show()


#for i in range(all_states_num):
#    fig = plt.figure()
#    plt.subplot(211)
#    plt.imshow(np.reshape(all_states[i,:], (1,N)))
#    plt.subplot(212)
#    plt.imshow(list_of_lems[i])
#    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#    plt.colorbar(cax=cax)
#    plt.show()



#path_to_docs = '/home/abel/Documents/Projects/BioMath/LEC/Saves/buzsaki/'
#
#with open(path_to_docs+"data_t16_wo.txt", "rb") as f:   
#    data = pickle.load(f)
#    
#    
#h = data['h']
#J = data['J']
#N = h.shape[0]
#all_states = np.array([list(seq) for seq in itertools.product([-1,1],repeat=N)])
#
#all_energies = np.array(calc_energy_list([h,J], all_states))
#
##memory friendly
#elist = []
#for i in range(2**N):
#    elist.append(calc_energy([h,J],np.array(list(bin(i)[2:].zfill(N)), dtype='int')))
#
#all_states_num = all_states.shape[0]
#
#
#plot_ordered_patterns(all_states[np.where(np.array(all_energies) < min(all_energies)+1)[0],:], h, J)