#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:09:17 2019

@author: abel
"""
from collections import Counter
from neurodynex.hopfield_network import network, pattern_tools, plot_tools
import matplotlib.pyplot as plt
import numpy
from scipy import *

# the letters we want to store in the hopfield network
#letter_list = ['A', 'B', 'C', 'D']
letter_list = ['E', 'F', 'G', 'H', 'I']
#letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# set a seed to reproduce the same noise in the next run
# numpy.random.seed(123)

abc_dictionary =pattern_tools.load_alphabet()
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_shape[0]*pattern_shape[1])

# create a list using Pythons List Comprehension syntax:
pattern_list = [abc_dictionary[key] for key in letter_list ]
#pattern_list.append(abc_dictionary['R'])
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

def gdd(weights, initial_state):
    """for each neuron, we flip its activity if the flip will decrease the
    energy. If we could not decrease the energy by flipping any
    neuron’s activity, then a local energy minimum is identified"""
    Nneur = initial_state.shape[0]
    current_state = numpy.zeros(Nneur)
    current_state[:] = initial_state[:]
    while True:
        e_old = calc_energy(weights, current_state)

#       attempt to flip spins i~1,N from their current state into {s i , in order of increasing i.
        indices = range(Nneur)

        #random order of spin flip
#        indices = np.random.permutation(Nneur)
        stop_ind = 0
        for ind in indices:

            new_state = current_state
            new_state[ind] = -current_state[ind]
            e_new = calc_energy(weights, new_state)
            e_delta = e_new - e_old

            if e_delta < 0:
                e_old = e_new
                current_state = new_state

            else:
                stop_ind += 1
                current_state[ind] = -current_state[ind]

            #stop if could not flip any spin during step
            if stop_ind == Nneur:
                return current_state

    return current_state

def calc_energy(weights, acts):
    return -numpy.sum(.5*numpy.multiply(weights, numpy.outer(acts, acts)))

def lem(weights, number_of_initial_patterns):
    """Determine local energy minima (for an Ising model)
    by Greedy Descent Dynamics (Huang and Toyoizumi, 2016)"""
    patterns = []
    for i_p in range(number_of_initial_patterns):
        initial_state = numpy.random.choice([-1,1], weights.shape[0])
        patterns.append(gdd(weights, initial_state))
    return patterns

def find_frequencies(codewords):
    tuple_codewords = map(tuple, codewords)
    freq_dict = Counter(tuple_codewords)
    freqs = numpy.array(sorted(list(freq_dict.values()),
                            reverse=True))
    
    return freqs

number_of_initial_patterns = 10**3
patterns_gdd = lem(hopfield_net.weights, number_of_initial_patterns)
tuple_codewords = map(tuple, patterns_gdd)
freq_dict_gdd = Counter(tuple_codewords)
code_probs_gdd = numpy.array(list(sorted(freq_dict_gdd.values(),reverse=True)), dtype="float64")/numpy.sum(list(freq_dict_gdd.values()))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(code_probs_gdd, 'o', label="GGD")
ax.set_yscale('log')
ax.set_xlabel("Codeword")
ax.set_ylabel("Probability")
plt.show()

print(len(freq_dict_gdd.values()))
fig = plt.figure()
for i in range(len(freq_dict_gdd.keys())):
    ax = fig.add_subplot(4,4,i+1)
    cax = ax.imshow(numpy.reshape(freq_dict_gdd.keys()[i],(10,10)))
    ax.set_title(freq_dict_gdd.values()[i]/float(number_of_initial_patterns))
    ax.axis('off')

fig.tight_layout() 
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
