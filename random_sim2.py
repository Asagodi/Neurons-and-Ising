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
from simulate_oned import *
import time as tm
from sklearn import manifold
from sklearn.manifold import MDS

path = "/home/abel/Documents/Projects/BioMath/LEC/Saves/Random N=100/"
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

#activities = np.loadtxt(path + "activities_105.csv", delimiter=',')

#######bin
#time = activities.shape[1]
b_act = bin_data(activities, time, 100, shift)
##detect spikes
spiked_act = detect_spikes(b_act, 120)
##determine state
s_act = determine_states(spiked_act)

#s_act = np.loadtxt(path+"head_ang_N100.csv", delimiter=',')
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
#print("Average spiking:" + str(np.average((mag_sim + 1)/2.)))
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
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(s_act[:,:1000])
plt.show()

def plot_single_pattern(pattern):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(np.reshape(pattern, (1,-1)))
    plt.show()


#np.savetxt(path+"head_ang_N100.csv", s_act, delimiter=",")
h, J = nMF(s_act)   
reg_method = "sign"
j=0
reg_lambda = 0.01
#h = np.loadtxt(path+"h_N100_" + str(reg_method) + "_" + str(reg_lambda) + "_" + str(j) + ".csv", delimiter=',')
#J = np.loadtxt(path+"J_N100_" + str(reg_method) + "_" + str(reg_lambda) + "_" + str(j) + ".csv", delimiter=',')
max_steps = 1000
h_lambda = .5
J_lambda = h_lambda
epsilon = 0.001
t0 = tm.time()
h, J, min_av_max_plm = plm_separated(s_act, max_steps,
                        h, J, h_lambda, J_lambda,
                        reg_method, reg_lambda, epsilon, 1.)
print("Time:", tm.time()-t0)
###np.savetxt(path+"random_50_h_" + reg_method + "_head.csv", h, delimiter=",")
##np.savetxt(path+"random_50_J_" + reg_method + "_head.csv", J, delimiter=",")

#############Test inferred model 
Nsamples = 10**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
Nflips = 1
sample_after = 1000 #10**7   
sample_per_steps = 100 # 10 * N
s_act_inferred = metropolis_mc(h, J, Nsamples, Nflips,
                  sample_after, sample_per_steps, 1.)

mag_sim = np.average(s_act, axis=1)
#corrs_sim = calc_correlations(s_act, mag_sim)
#corrs3_sim = calc_third_order_corr(s_act, mag_sim)
#corrs_sim = np.loadtxt(path+"random_50_corrs_sim_9.csv", delimiter=',')
#corrs3_sim = np.loadtxt(path+"random_50_corrs3_sim_9.csv", delimiter=',')

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


###############LEM
number_of_initial_patterns = 10**3
T = 1.
#patterns_gdd = lem(h, J, number_of_initial_patterns)
patterns_gdd, init_final_dict = lem_from_data(h, J, s_act[:,:100])  



##look for expected pattern energies and check for local energy minimum
n_bumps = 4
length_bump = 8
exp_patterns = make_expected_patterns(N, n_bumps, length_bump)
plot_patterns_with_energies(exp_patterns)


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



def get_indices_where_different(pattern1, pattern2):
    indxs = np.where(pattern1 != pattern2)[0]
    return indxs

def make_shortest_paths_between_patterns(pattern1, pattern2):
    indxs = get_indices_where_different(pattern1, pattern2)
    all_paths = []
    try:
        for indexset in list(itertools.permutations(indxs)):
            pattern_path = make_patterns_from_indices(indexset, pattern1, pattern2)
            all_paths.append(pattern_path)
    except:
        print("Patterns are identical")
    return all_paths

def make_patterns_from_indices(indxs, begin, end):
    pattern_path = [begin]
    prev_pattern = np.copy(begin)
    for ind in indxs:
        next_pattern = np.copy(prev_pattern)
        next_pattern[ind] = -next_pattern[ind]
        prev_pattern = np.copy(next_pattern)
        pattern_path.append(next_pattern)
#    print(np.all(next_pattern == end))
    return np.array(pattern_path)

def calculate_energies_for_paths(paths):
    energies_per_path = []
    for path_between in paths:
        energies_on_this_path = []
        for pattern in path_between:
            energies_on_this_path.append(calc_energy([h1,h2], [h,J], pattern))
        energies_per_path.append(energies_on_this_path)
    return energies_per_path

paths = make_paths_between_patterns(exp_patterns[:,3], exp_patterns[:,4])
enss = calculate_energies_for_paths(paths)
average_of_paths = np.average(np.array(enss), axis=1)
min(average_of_paths)

paths[np.where(average_of_paths == np.min(min(average_of_paths)))[0][0]]
