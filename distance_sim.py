#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:11:20 2019

@author: abel
"""

import numpy as np
from scipy import *
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lec import *
import time as tm
from sklearn import manifold
from sklearn.manifold import MDS


#get lems from data
patterns_gdd, init_final_dict = lem_from_data(h, J, s_act[:,:number_of_initial_patterns], 'random')  
ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)


#get transition rates from MC simulations
lem_patterns, matrix_attempted_flips, matrix_both_flips, matrix_energy_barriers, path_list, basin_size_list = get_transition_rates(h, J, ordered_patterns, 10**6)


n_patt = ordered_patterns.shape[0]
d_matrix = zeros([n_patt, n_patt])
for i in range(n_patt):
    for j in range(n_patt):
        d_matrix[i,j] = np.average(matrix_attempted_flips[i][j])
            
#put this at least as high as highest value
d_matrix[np.isnan(d_matrix)]  = 1000

max_s = 100
distances = dissimilarities
for step in range(max_s):
    for k in range(distances.shape[0]):
        for i,dik in enumerate(distances[k,:]):
            for j,djk in enumerate(distances[k,:]):
                distances[i,j] = min(distances[i,j], dik+djk)

np.fill_diagonal(d_matrix, 0)
            
for i in range(d_matrix.shape[0]):
    for j,djk in enumerate(d_matrix[i,:]):
        d_matrix[i,j] = min(d_matrix[i,j], d_matrix[j,i])  #average?
        
p = 2
embedding = MDS(n_components=p, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(d_matrix)
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
#ax = fig.gca(projection='3d')
cax = ax.scatter(X_transformed[:,0], X_transformed[:,1], marker="x", c="b", label="Expected patterns")
#X_transformed[:,2]
ax.set_xlabel("X")
ax.set_ylabel("Y")
#ax.set_ylabel("Z")
#ax.legend()
plt.show()