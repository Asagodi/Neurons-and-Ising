#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:13:08 2019

@author: abel
"""
import numpy as np
from scipy import *
import scipy.io
import matplotlib.pyplot as plt
from simulate_oned import *
import time as tm

path = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Biased N=50/"

#N, extinp, inh, R = 50, 3, 0.235619449019, 20.0666666667
#umax, dtinv, tau, time, ell, alpha, lambda_net = 1., 10, 10, 10**4, 2., 0.1, 13
#bsize, shift, scale, dt = 10, 0, 10000, 1
##
#mat = scipy.io.loadmat('animal_movement.mat')
#data = [ravel(array(mat['posx'])), ravel(array(mat['posy']))]
##data =ravel(array(mat['posx']))
##data_1 = -min(data) + data
##head_data = 2*data_1/max(data_1)
#time  = array(mat['posx']).shape[1]-1
#
##
#t0 = tm.time()
#activities = sim_dyn_one_d(N, extinp, inh, R, umax, dtinv,
#              tau, time, ell, alpha, data, dt)
#print("Time:", tm.time()-t0)
#####bin
#b_act = bin_data(activities, time, bsize, shift)
###detect spikes
#spiked_act = detect_spikes(b_act, 120)
###determine state
#s_act = determine_states(spiked_act)
###
#mag_sim = np.average(s_act, axis=1)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.plot((mag_sim + 1)/2.)
#plt.show()
##
#print(np.average((mag_sim + 1)/2.))
##
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(s_act[:,:1000])
#plt.show()
#
#def animate(i):
#    im.set_data(np.reshape(s_act[:,i*100:(i+1)*100], (N,100)))
#    return im,
#fig = plt.figure()
#im =  plt.imshow(np.reshape(s_act[:,0:100], (N,100)), animated=True)
#plt.colorbar()
#def init():  
#    im.set_data(np.reshape(s_act[:,0:100], (N,100)))
#    return im,
#
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                   frames=100, interval=100, blit=True)

#
#np.savetxt(path+"sim_50_b_106_9.csv", b_act, delimiter=",")
#np.savetxt(path+"sim_50_s_106_9.csv", s_act, delimiter=",")

#s_act = np.loadtxt(path+"sim_50_head.csv", delimiter=',')
#mag_sim = np.average(s_act, axis=1)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.plot(mag_sim)
#plt.show()

#h, J = nMF(s_act)  
reg_method = "l2"
#h = np.loadtxt(path+"random_50_h_" + reg_method + "_head.csv", delimiter=',')
#J = np.loadtxt(path+"random_50_J_" + reg_method + "_head.csv", delimiter=',')

max_steps = 100  
h_lambda = .25
J_lambda = h_lambda
reg_lambda = .00001
epsilon = 0.0001
t0 = tm.time()
h, J, min_av_max_plm = plm_separated(s_act, max_steps,
                        h, J, h_lambda, J_lambda,
                        reg_method, reg_lambda, epsilon, 1.)
#np.savetxt(path+"random_50_h_" + reg_method + "_head.csv", h, delimiter=",")
#np.savetxt(path+"random_50_J_" + reg_method + "_head.csv", J, delimiter=",")
print("Time:", tm.time()-t0)

#############Test inferred model 
#Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
#Nflips = 1
#sample_after = 1000 #10**7   
#sample_per_steps = 100 # 10 * N
#s_act_inferred = metropolis_mc(h, J, Nsamples, Nflips,
#                  sample_after, sample_per_steps, 1.)
#
#mag_sim = np.average(s_act, axis=1)
##corrs_sim = calc_correlations(s_act, mag_sim)
##corrs3_sim = calc_third_order_corr(s_act, mag_sim)
##corrs_sim = np.loadtxt(path+"random_50_corrs_sim_9.csv", delimiter=',')
##corrs3_sim = np.loadtxt(path+"random_50_corrs3_sim_9.csv", delimiter=',')
#
##
#mag_inf = np.average(s_act_inferred, axis=1)
#corrs_inf = calc_correlations(s_act_inferred, mag_inf)
#corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)
#
#fig = plt.figure(figsize=(7.5, 7.5))
#ax = fig.add_subplot(111)
#cax = ax.plot([-1, 1], [-1, 1], c="r")
#cax = ax.plot(corrs_sim.flatten(), corrs_inf.flatten(), 'x', c='tab:orange', label="Correlations")
#cax = ax.plot(corrs3_sim.flatten(), corrs3_inf.flatten(), 'x', c='g', label="Third order correlations")
#cax = ax.plot(mag_sim, mag_inf, 'o', c='b', label="Average magnetization")
#ax.set_xlabel("Data")
#ax.set_ylabel("Inferred")
#ax.legend()
#plt.show()


#######Tuning Curve
#def get_tuned(patterns, i, max_d):
#    tc_list = []
#    for a in np.where(patterns[:, i] > 0)[0]:
#        for j in range(i-max_d, i+max_d, 1):
#            if patterns[j % 50, a] == 1.:
#                tc_list.append(j % 50)
#    return tc_list

#for n in range(N):
#    tcl = get_tuned(patterns.T, n, 10)
#    tc = zeros(N)
#    tc[tcl] = 1
#    ##or
##    for tci in tcl:
##        tc[tci] += 1
#    fig=plt.figure(1)
#    plt.clf()
#    ax = fig.gca(projection='3d')
#    theta = linspace(0, 2*pi, N)
#    x = sin(theta)
#    y = cos(theta)
#    ax.scatter(x, y, tc, color='blue')
#    plt.axis('off')
#    plt.show()


###############LEM
number_of_initial_patterns = 10**2
T = 1.
patterns_gdd = lem(h, J, number_of_initial_patterns)
tuple_codewords = map(tuple, patterns_gdd)
freq_dict_gdd = Counter(tuple_codewords)
code_probs_gdd = np.array(list(sorted(freq_dict_gdd.values(),reverse=True)), dtype="float64")/np.sum(freq_dict_gdd.values())

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(code_probs_gdd, 'o', label="GGD")
ax.set_yscale('log')
ax.set_xlabel("Codeword")
ax.set_ylabel("Probability")
plt.show()

indexed = sorted(range(len(freq_dict_gdd.values())), key=lambda k: freq_dict_gdd.values()[k])
indexed_patterns = [freq_dict_gdd.keys()[i] for i in indexed]

stored_energies = []
oel = []
for pattern in freq_dict_gdd.keys():
    energy = calc_energy([h1,h2], [h,J], pattern)
    stored_energies.append(energy)
    oel.append([freq_dict_gdd.get(pattern)/float(number_of_initial_patterns), round(energy, 2)])
    
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(sorted(stored_energies, reverse=True), 'x')
plt.show()


###order and plot found local energy minima
N=50
ordered_indices = []
for j in range(N):
    for i in range(len(freq_dict_gdd.keys())): 
        if freq_dict_gdd.keys()[i][j-1] == -1. and freq_dict_gdd.keys()[i][j] == 1. and i not in ordered_indices:
            ordered_indices.append(i)
            

for i in range(len(freq_dict_gdd.keys())): 
    if i not in ordered_indices:
        ordered_indices.append(i)
            
ordered_patterns = [freq_dict_gdd.keys()[i] for i in ordered_indices]
ordered_energies = [oel[i] for i in ordered_indices]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(ordered_patterns)
ax.set_yticklabels(['']+ordered_energies)
ax.set_yticks([i for i in np.arange(-1, len(stored_energies), 1.)])
plt.show()           



##look for expected pattern energies and check for local energy minimum
exp_patterns = -ones([50,25])
lenght = 8
shift = int((50 - 2*lenght)/2)
for i in range(25):
    for n in range(lenght):
        exp_patterns[n+i, i] = 1
        exp_patterns[(n + lenght+shift +i)% 50, i] = 1
        
expected_pattern_energies = []
lems_expected_patterns = []
lems = zeros([50,25])
for i,pattern in enumerate(exp_patterns.T):
    energy = calc_energy([h1,h2], [h,J], pattern)
    expected_pattern_energies.append(round(energy,2))
    lem_patt = gdd([h, J], pattern)
    lems[:,i] = lem_patt
    lems_expected_patterns.append(calc_energy([h1,h2], [h,J], lem_patt))
    
    
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(lems.T)
ax.set_yticklabels(['']+expected_pattern_energies)
ax.set_yticks([i for i in np.arange(-.5, len(expected_pattern_energies), 1.)])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(expected_pattern_energies, label="Energies of the expected patterns")
cax = ax.plot(lems_expected_patterns, label="Energies of the LEMs of these patterns")
ax.legend()
plt.show()

#energy_indices = sorted(range(len(stored_energies)), key=lambda k: stored_energies[k])
#indexed_energies = [stored_energies[i] for i in energy_indices]
#energy_indexed_patterns = [freq_dict_gdd.keys()[i] for i in energy_indices]
#energy_indexed_occurances = [freq_dict_gdd.values()[i] for i in energy_indices]

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(energy_indexed_patterns)
#plt.show()

#fig = plt.figure(figsize=(7.5, 7.5))
#ax = fig.add_subplot(111)
#coordinates_lem = mds(np.array([sigma for sigma in list(freq_dict_gdd.keys())]), p, 2)
#cax = ax.scatter(coordinates_lem[:,0], coordinates_lem[:,1], s=list(freq_dict_gdd.values()), marker="x")
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#for i in range(coordinates_lem.shape[0]):
##    ax.annotate(i, (coordinates_lem[i, 0], coordinates_lem[i, 1]))
#    ax.annotate(freq_dict_gdd.values()[i], (coordinates_lem[i, 0], coordinates_lem[i, 1]))
#plt.show()

##MDS
p=2
#MACOF
from sklearn.manifold import MDS

number_of_patterns = len(freq_dict_gdd.values())
dissimilarities = zeros([50, 50])
combined_patterns = np.concatenate((exp_patterns.T, np.array(lems).T))
print combined_patterns.shape
pattern_array = np.array(freq_dict_gdd.keys())
for r, dp_r in enumerate(combined_patterns):
    for s, dp_s in enumerate(combined_patterns):
        delta_rs = hamming_distance(dp_r, dp_s)
        dissimilarities[r,s] = delta_rs

embedding = MDS(n_components=p, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(dissimilarities)
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
cax = ax.scatter(X_transformed[:25,0], X_transformed[:25,1], marker="x", c="b", label="Expected patterns")
cax = ax.scatter(X_transformed[25:,0], X_transformed[25:,1], marker="o", c="r", label="LEMs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()

#sklearn.manifold.smacof(dissimilarities, metric=True, n_components=2,
#                        init=None, n_init=8, n_jobs=None, max_iter=300,
#                        verbose=0, eps=0.001, random_state=None,
#                        return_n_iter=False)