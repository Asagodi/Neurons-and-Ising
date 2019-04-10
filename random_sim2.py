#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:13:08 2019

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

path_to_docs = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Head Data N=100/"
#
N, extinp, inh, R = 100, 3, 0.235619449019, 20.0666666667
umax, dtinv, tau, time, ell, alpha, lambda_net = 1., 10, 10, 10**5, 2., 0.25, 13
bsize, shift, scale, dt = 10, 0, 10000, 1
##
#mat = scipy.io.loadmat('animal_movement.mat')
#data = [ravel(array(mat['posx'])), ravel(array(mat['posy']))]
##data =ravel(array(mat['posx']))
##data_1 = -min(data) + data
##head_data = 2*data_1/max(data_1)
#time  = 25*array(mat['posx']).shape[1]
##
###
#mat = scipy.io.loadmat('Mouse12-120806_stuff_simple_awakedata.mat')
#angles = mat['headangle'][~np.isnan(mat['headangle'])] - pi
#time  = 25*len(angles)
#t0 = tm.time()
#activities, vs, thetas = sim_dyn_one_d(N, extinp, inh, R, umax, dtinv,
#              tau, time, ell, alpha, angles, dt)
#activities = sim_dyn_one_d_random(N, extinp, inh, R, umax, dtinv,
#              tau, time, ell, scale)
#print("Time:", tm.time()-t0)
##


####data from ben sim
#mat  = scipy.io.loadmat('data2.mat')
#activities = mat['S']
#
########bin
#time = activities.shape[1]
#b_act = bin_data(activities, time, 10, shift)
###detect spikes
#spiked_act = detect_spikes(b_act, 12)
###determine state
#s_act = determine_states(spiked_act)

#s_act = np.loadtxt(path+"random_50_h_l2_4.csv", delimiter=',')
#s_act = np.loadtxt(path+"s_act_N100_bsize_100.csv", delimiter=',')
###
#mag_sim = np.average(s_act, axis=1)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.plot((mag_sim + 1)/2.,label="Simulated")
##cax = ax.plot((mag_inf + 1)/2.,label="Inferred")
#ax.set_title("Average activity per neuron")
#ax.set_xlabel("Neuron")
#ax.set_ylabel("Probability of firing")
#ax.legend()
#plt.show()
print("Average spiking:" + str(np.average((mag_sim + 1)/2.)))
##
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.plot(vs[:])
#plt.show()
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.plot(thetas[:])
#plt.show()
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.imshow(s_act[:,1000:2000])
#plt.show()


#np.savetxt(path+"head_ang_N100.csv", s_act, delimiter=",")
#h, J = nMF(s_act)   
reg_method = "l2"
reg_lambda = .05
max_steps = 250 
h_lambda = 0.1672
J_lambda = h_lambda
epsilon = 0.001
t0 = tm.time()
h, J, min_av_max_plm = plm_separated(s_act, max_steps,
                        h, J, h_lambda, J_lambda,
                        reg_method, reg_lambda, epsilon, 1.)
#print("Time:", tm.time()-t0)

###np.savetxt(path_to_docs+"random_50_h_" + reg_method + "_head.csv", h, delimiter=",")
##np.savetxt(path_to_docs+"random_50_J_" + reg_method + "_head.csv", J, delimiter=",")

#############Test inferred model 
Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
sample_after = 1000 #10**7   
sample_per_steps = 2*N # 10 * N
s_act_inferred = metropolis_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, 1.)

mag_sim = np.average(s_act, axis=1)
#corrs_sim = calc_correlations(s_act, mag_sim)
#corrs3_sim = calc_third_order_corr(s_act, mag_sim)
#corrs_sim = np.loadtxt(path_to_docs+"random_50_corrs_sim_9.csv", delimiter=',')
#corrs3_sim = np.loadtxt(path_to_docs+"random_50_corrs3_sim_9.csv", delimiter=',')

#
mag_inf = np.average(s_act_inferred, axis=1)
corrs_inf = calc_correlations(s_act_inferred, mag_inf)
corrs3_inf = calc_third_order_corr(s_act_inferred, mag_inf)

fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
cax = ax.plot([-1, 1], [-1, 1], c="r")
cax = ax.plot(corrs_sim.flatten(), corrs_inf.flatten(), 'x', c='tab:orange', label="Correlations")
cax = ax.plot(corrs3_sim.flatten(), corrs3_inf.flatten(), 'x', c='g', label="Third order correlations")
cax = ax.plot(mag_sim, mag_inf, 'o', c='b', label="Average magnetization")
ax.set_xlabel("Data")
ax.set_ylabel("Inferred")
ax.legend()
plt.show()

print("Error:", np.sqrt(np.sum(np.square(mag_sim - mag_inf)) + np.sum(np.square(corrs_sim - corrs_inf)) + np.sum(np.square(corrs3_sim - corrs3_inf))))
print("Error:", np.sqrt(np.average(np.square(mag_sim - mag_inf)) + np.average(np.square(corrs_sim - corrs_inf)) + np.average(np.square(corrs3_sim - corrs3_inf))))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(s_act_inferred[:,:1000])
plt.show()


###############LEM
number_of_initial_patterns = 10**3
T = 1.
#patterns_gdd = lem(h, J, number_of_initial_patterns)
#########from data
patterns_gdd, init_final_dict = lem_from_data(h, J, s_act[:,:number_of_initial_patterns], 'random')  
ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)

#####RANDOM #or ordered 
init_part_active = 0.5
patterns_gdd = lem(h, J, number_of_initial_patterns, init_part_active, 'random')  
ordered_patterns = plot_ordered_patterns(patterns_gdd, h, J)


##look for expected pattern energies and check for local energy minimum
n_bumps = 1
length_bump = 40
exp_patterns = make_expected_patterns(N, n_bumps, length_bump)
lems = plot_patterns_with_energies(h, J, exp_patterns, n_bumps)


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
num_patts=100
tuple_codewords = map(tuple, patterns_gdd)
freq_dict_gdd = Counter(tuple_codewords)
number_of_patterns = len(freq_dict_gdd.values())
dissimilarities = zeros([2*num_patts, 2*num_patts])
combined_patterns = np.concatenate((exp_patterns.T, np.array(lems).T))
pattern_array = np.array(freq_dict_gdd.keys())
for r, dp_r in enumerate(combined_patterns):
    for s, dp_s in enumerate(combined_patterns):
        delta_rs = hamming_distance(dp_r, dp_s)
        dissimilarities[r,s] = delta_rs

embedding = MDS(n_components=p, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(dissimilarities)
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(111)
cax = ax.scatter(X_transformed[:num_patts,0], X_transformed[:num_patts,1], marker="x", c="b", label="Expected patterns")
cax = ax.scatter(X_transformed[num_patts:,0], X_transformed[num_patts:,1], marker="o", c="r", label="LEMs")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()



###ISOMAP
#df = pd.DataFrame(combined_patterns)
#iso = manifold.Isomap(n_neighbors=7, n_components=2)
#iso.fit(df)
#manifold_2Da = iso.transform(df)
#manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])
#fig = plt.figure()
#fig.set_size_inches(10, 10)
#ax = fig.add_subplot(111)
#ax.set_title('2D Components from Isomap')
#ax.scatter(manifold_2D['Component 1'][:num_patts], manifold_2D['Component 2'][:num_patts], marker='.',c='b',alpha=0.7)
#ax.scatter(manifold_2D['Component 1'][num_patts:], manifold_2D['Component 2'][num_patts:], marker='.',c='r',alpha=0.7)
#plt.show()

####http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/



number_of_patterns = len(ordered_patterns)
dissimilarities = zeros([number_of_patterns, number_of_patterns])
for r, dp_r in enumerate(ordered_patterns):
    for s, dp_s in enumerate(ordered_patterns):
        delta_rs = hamming_distance(np.array(dp_r), np.array(dp_s))
        dissimilarities[r,s] = delta_rs
        
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(dissimilarities)
fig.colorbar(cax)
plt.show()

get_indices_where_different(ordered_patterns[12], ordered_patterns[13]).shape
paths = make_shortest_paths_between_patterns(ordered_patterns[12], ordered_patterns[13])
enss = np.array(calculate_energies_for_paths(paths, h, J))
average_of_paths = np.average(np.array(enss), axis=1)
min(average_of_paths)

###minimum average
paths[np.where(average_of_paths == np.min(min(average_of_paths)))[0][0]]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(paths[np.where(average_of_paths == np.min(min(average_of_paths)))[0][0]])
plt.show()


###minmax
max_along_paths = np.max(enss, axis=1)
paths[np.where(max_along_paths==np.min(max_along_paths))[0]]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(paths[np.where(enss == np.min(max_along_paths))[0][0]])
plt.show()


###minimum of the sum of local maxima #minsummax
def local_maxima_on_path(energies_on_path):
    local_maxs = []
    mine = min(energies_on_path[0], energies_on_path[-1])
    for i,energy in enumerate(energies_on_path[1:-1]):
        if energies_on_path[i] < energy and energy > energies_on_path[i+2]:
            local_maxs.append(energy - mine)
    return local_maxs


def maxima_along_paths(enss):
    all_local_maxs = []
    for energies_on_path in enss:
        all_local_maxs.append(local_maxima_on_path(energies_on_path))       
    return all_local_maxs

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.imshow(paths[np.where(all_local_extrs == min(all_local_extrs))][0])
#plt.show()

####differences between minima and maxima
def local_extrema_on_path(energies_on_path):
    local_extr = []
    mine = energies_on_path[0]
    print(mine)
    sign = 1
    for i,energy in enumerate(energies_on_path[1:-1]):
        if sign*energies_on_path[i] < sign*energy and sign*energy > sign*energies_on_path[i+2]:
            if sign==1:
                local_extr.append(energy - mine)
            elif sign==-1:
                mine = energy
            sign *= -1
            print(mine, sign)
    return local_extr

def sum_extrema_along_paths(enss):
    all_local_extrs = []
    for energies_on_path in enss:
        maxs = 0
        for maximum in local_maxima_on_path(energies_on_path):
            maxs += maximum
        all_local_extrs.append(maxs)
    return all_local_extrs


### distance or spike-counts evolution of the neural data
dt = (1 - np.dot(s_act_inferred.T, ordered_patterns[1])/N)/2.

dt = zeros(s_act_inferred.shape[1]-1)
for t in range(s_act_inferred.shape[1]-1):
    dt[t] = (N - np.dot(s_act_inferred[:,t], s_act_inferred[:,t+1]))/2.
    
    
energies_along_path = zeros(get_indices_where_different(ordered_patterns[12], ordered_patterns[13]).shape)
fig = plt.figure()
ax = fig.add_subplot(111)
for i,pattern in enumerate(paths[np.where(max_along_paths==np.min(max_along_paths))[0]][0]):
    energies_along_path[i] = calc_energy([h1,h2], [h,J], pattern)
cax = ax.plot(energies_along_path)
plt.show()



number_of_patterns = len(ordered_patterns)
dissimilarities = zeros([number_of_patterns, number_of_patterns])
for r, dp_r in enumerate(ordered_patterns):
    for s, dp_s in enumerate(ordered_patterns):
        paths = make_np_shortest_paths(dp_r, dp_s, 2**5)
        enss = np.array(calculate_energies_for_paths(paths, h, J))
        max_along_paths = np.max(enss, axis =1) #ave_along_paths=ave_along_paths
        dissimilarities[r,s] = np.min(max_along_paths)-min(calc_energy([h1,h2], [h,J], dp_r), calc_energy([h1,h2], [h,J], dp_s))
        
        
for r, dp_r in enumerate(ordered_patterns):
    for s, dp_s in enumerate(ordered_patterns):
        dissimilarities[r,s] = min(dissimilarities[r,s], dissimilarities[s,r])
        

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
graph = csr_matrix(dissimilarities)
distances = dijkstra(graph)


#p=3
embedding = MDS(n_components=p, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(distances)
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


from scipy.sparse.csgraph import minimum_spanning_tree


###minmax with paths through other states.
max_s = 100
for step in range(max_s):
    for k in range(distances.shape[0]):
        for i,dik in enumerate(distances[k,:]):
            for j,djk in enumerate(distances[k,:]):
                distances[i,j] = min(distances[i,j],max(dik,djk))
                
                

#### min sum max through other states             
max_s = 100
distances = dissimilarities
for step in range(max_s):
    for k in range(distances.shape[0]):
        for i,dik in enumerate(distances[k,:]):
            for j,djk in enumerate(distances[k,:]):
                distances[i,j] = min(distances[i,j], dik+djk)