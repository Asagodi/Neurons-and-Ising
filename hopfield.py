#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:09:17 2019

@author: abel
"""
from collections import Counter
from neurodynex.hopfield_network import network, pattern_tools, plot_tools
import matplotlib.pyplot as plt
import numpy as np
from scipy import *
from lec import *


#nr_neurons
N=100

# the letters we want to store in the hopfield network
#letter_list = ['A', 'B', 'C', 'D']
#letter_list = ['E', 'F', 'G', 'H', 'I']
#letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
letter_list = ['A', 'B', 'E', 'H', 'I']

# set a seed to reproduce the same noise in the next run
# np.random.seed(123)

abc_dictionary =pattern_tools.load_alphabet()
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons= N)
#hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_shape[0]*pattern_shape[1])

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list ]
plot_tools.plot_pattern_list(pattern_list)

# how similar are the letter patterns
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
plot_tools.plot_overlap_matrix(overlap_matrix)

# store the patterns
hopfield_net.store_patterns(pattern_list)

## # create a noisy version of a pattern and use that to initialize the network
#noisy_init_state = pattern_tools.get_noisy_copy(abc_dictionary['A'], noise_level=0.2)
#hopfield_net.set_state_from_pattern(noisy_init_state)
#
## from this initial state, let the network dynamics evolve.
#states = hopfield_net.run_with_monitoring(nr_steps=4)
#
## each network state is a vector. reshape it to the same shape used to create the patterns.
#states_as_patterns = pattern_tools.reshape_patterns(states, pattern_list[0].shape)

## plot the states of the network
#plot_tools.plot_state_sequence_and_overlap(
#    states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

h = zeros(N)
J = hopfield_net.weights

number_of_initial_patterns = 10**3
patterns_gdd = lem(h, J, number_of_initial_patterns, 'random', 0.5)
tuple_codewords = map(tuple, patterns_gdd)
freq_dict_gdd = Counter(tuple_codewords)
code_probs_gdd = np.array(list(sorted(freq_dict_gdd.values(),reverse=True)), dtype="float64")/np.sum(list(freq_dict_gdd.values()))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(code_probs_gdd, 'o', label="GGD")
ax.set_yscale('log')
ax.set_xlabel("Codeword")
ax.set_ylabel("Probability")
plt.show()

num_5 = int(len(freq_dict_gdd.values())/5)+1

print(len(freq_dict_gdd.values()))
fig = plt.figure(figsize=(10,10))
energies = []
for i in range(len(freq_dict_gdd.keys())):
    energies.append(calc_energy([h1,h2], [h, hopfield_net.weights], freq_dict_gdd.keys()[i]))
    ax = fig.add_subplot(5,num_5,i+1)
    cax = ax.imshow(np.reshape(freq_dict_gdd.keys()[i],(10,10)))
    ax.set_title(freq_dict_gdd.values()[i]/float(number_of_initial_patterns))
    ax.axis('off')

#fig.tight_layout() 
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(energies, 'o', label="GGD")
ax.set_xlabel("Codeword")
ax.set_ylabel("Energy")
plt.show()



#MACOF
from sklearn.manifold import MDS
p=2
number_of_patterns = len(freq_dict_gdd.values())
dissimilarities = zeros([number_of_patterns, number_of_patterns])
pattern_array = np.array(freq_dict_gdd.keys())
for r, dp_r in enumerate(pattern_array):
    for s, dp_s in enumerate(pattern_array):
        delta_rs = hamming_distance(dp_r, dp_s)
        dissimilarities[r,s] = delta_rs

embedding = MDS(n_components=p, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(dissimilarities)
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
cax = ax.scatter(X_transformed[:,0], X_transformed[:,1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = ax.imshow(dissimilarities)
plt.show()


Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
Nflips = 1
sample_after = 1000 #10**7   
sample_per_steps = N*10 # 10 * N
s_act_inferred = metropolis_mc(h, J, Nsamples, Nflips,
                  sample_after, sample_per_steps, 1.)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(s_act_inferred[:,:1000])
plt.show()


ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)
#paths = make_shortest_paths_between_patterns(ordered_patterns[12], ordered_patterns[13])
#enss = calculate_energies_for_paths(paths, h, J)
#average_of_paths = np.average(np.array(enss), axis=1)
#min(average_of_paths)
#
####minimum average
#paths[np.where(average_of_paths == np.min(min(average_of_paths)))[0][0]]
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.imshow(paths[np.where(average_of_paths == np.min(min(average_of_paths)))[0][0]])
#plt.show()

def plot_path_hopfield(path):
    for pattern in path:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(np.reshape(pattern,(10,10)))
        plt.show()


###minmax
#paths[np.where(max_along_paths==np.min(max_along_paths))[0]]