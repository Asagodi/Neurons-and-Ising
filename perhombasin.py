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
from ripser import Rips


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


N=16

all_states = np.array([list(seq) for seq in itertools.product([-1,1],repeat=N)])
#all_states =  get_lem_neighbourhood(exp_patts, 2)

exp_patts = make_expected_patterns_with_intermediary(N) 
h = -1.75*ones(N)
J = make_hopfield_weights(exp_patts)

#h = data['h']
#J = data['J']
#N=h.shape[0]

all_energies = np.array(calc_energy_list([h,J], all_states))

#take subset of all states (f^-1  of neighbourhoods around minima)
all_states = all_states[np.where(np.array(all_energies) < min(all_energies)+13.)[0],:]
all_energies = np.array(calc_energy_list([h,J], all_states))

lems, list_of_basins, list_of_lems = determine_basins(h, J, all_states)
    
G = nx.Graph()
for i, ui in enumerate(list_of_basins):
    G.add_node(str(i+1))
    for j, uj in enumerate(list_of_basins):
        
        ##check for overlap between clusters (basins)
        for patti in ui:
            for pattj in uj:
                if np.all(patti==pattj) and i!=j:
                    G.add_edge(str(i+1), str(j+1))
    
plt.figure(figsize=(4,4))   
nx.draw_spectral(G, with_labels = True)
plt.show()


#if looking at Hopfield Ring:
#plt.figure(figsize=(4,4))
#plt.imshow(exp_patts)
#plt.xticks(range(0,N,2), range(1,N+1, 2))
#plt.yticks(range(0,N,2), range(1,N+1, 2))
#plt.xlabel("Neuron")
#plt.ylabel("LEM")
#plt.title("Local Energy Minima")
#plt.show()


for cc in nx.connected_components(G):
    plt.figure(figsize=(4,4))
    nx.draw_spring(G.subgraph(cc))
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


