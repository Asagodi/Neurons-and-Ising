#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:24:17 2019

@author: abel
"""

#matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from scipy import *
from scipy import optimize
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import itertools
import numpy as np
#from progress.bar import Bar


def make_connection_matrix(N, inh, R, ell):
    #construct a connection matrix as in Couey 2013
    theta = zeros([N])
    theta[0:N:2] = 0
    theta[1:N:2] = 1
    theta = 0.5*pi*theta
#    theta = arange(N)*2.*pi/float(N) - pi
    theta = ravel(theta)
    xes = zeros(N)
    for x in range(N):
      xes[x] = x
    
    Wgen = zeros([N,N], dtype=bool)
    for x in range(N):
      xdiff = abs(xes-x-ell*cos(theta))
      xdiff = minimum(xdiff, N-xdiff)
      Wgen[xdiff<R,x] = 1
    W = zeros([N,N])
    W[Wgen>0.7] = -inh
    return W, theta


def sim_dyn_one_d(N, extinp, inh, R, umax, dtinv,
              tau, time, ell, alpha, data, dt):
    """perform simulation as in Couey 2013
    N: number of neurons
    extinp: external input
    inh: inhibitory connection strength
    R: radius of inhibitory connections
    umax:
    dtinv: inverse of step size
    tau: 
    time: to perform the simulation
    ell: shift of neuron preference for direction
    alpha: coupling to head direction
    data: positions: posx, posy
    dt: timestep size to calculate head direction and velocity 
    """
    W, theta = make_connection_matrix(N, inh, R, ell)
    W = sparse.csc_matrix(W)
#    posx = data[0]
#    posy = data[1]

    S = zeros(N)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
        
    activities = zeros([N,time])
    Stemp = zeros(N)
    vs = []
    thetas = []
    angle_i = 0
    
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15, metadata=dict(title=''))
    fig = plt.figure(2)
    with writer.saving(fig, 'mav2.mp4', 300):
        for t in range(0, time, 1):
        ##use if animal position is given:
#        if t % 100 == 0 and t + 500 < time:
#            tn = t / 5
#            v = 10*np.sqrt((posy[tn+50]-posy[tn-50])**2 + (posx[tn+50]-posx[tn-50])**2)
#            theta_t = arctan2( posy[tn+dt]-posy[tn-dt], posx[tn+dt]-posx[tn-dt]) + 0.5*pi*theta
            
            ##use if animal head direction is given:
            if t % 25 == 0 and angle_i+1 < len(data): 
                theta_t = data[angle_i] #- data[angle_i-1] + .5*pi
                v = np.abs(data[angle_i-1]  - data[angle_i+1])
                if v > pi:
                    v = 2*pi - v
                v *= 40
                angle_i += 1 
                vs.append(v)
                thetas.append(cos(theta_t - theta))
    
            S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*W + alpha * v * cos(theta_t - theta)))
            S[S<0.00001] = 0.
            activities[:,t] = S
#            if (10*t) % (time) == 0:
#                print("Process:" + str(100*t/time) + '%')
#            
            if(t<5000 and mod(t,10)==0):
                plt.clf()
                ax = plt.subplot(2,2,1)
                ax.plot(S, '-')
#                plt.plot([ni, ni], [min(S),max(S)], '-', color='red')
#                plt.ylabel('neural activity')
#                plt.xlabel('neurons')
                ax = fig.add_subplot(2,2,2)
                ang=theta_t
                x0 = cos(ang)*0.5
                y0 = sin(ang)*0.5
                ax.plot([0,x0], [0,y0])
                ax.axis([-0.5, 0.5, -0.5, 0.5])
                writer.grab_frame()
                plt.xlabel('heading direction')
    return activities, vs, thetas

def make_burak(N, inh, R, ell, lambda_net):
    theta = zeros([N])
    theta[0:N:2] = 0
    theta[1:N:2] = 2
    theta = 0.5*pi*theta
    theta = ravel(theta)
    
    beta = 3./(lambda_net**2)
    gamma = 1.05*beta
    W = zeros([N,N])
    for x in range(N):
        for y in range(N):
            xdiff = abs(x-y-ell*cos(theta[y]))
#            print(x, y, xdiff)
            W[x, y] = exp(-gamma*xdiff) - exp(-beta*xdiff)
#            print(W[x, y])
#
    return W, theta

def sim_burak(N, extinp, inh, R, umax, dtinv,
              tau, time, ell, alpha, lambda_net):
    W, theta = make_burak(N, inh, R, ell, lambda_net)
    W = sparse.csc_matrix(W)

    S = zeros(N)
    ## generate random activity (doesn't matter much)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
        
    activities = zeros([N,time])
    Stemp = zeros(N)
    for t in range(0, time, 1):
        v = .0
        if t % 10000 == 0:
            theta_t = 0.5*pi*np.random.choice([0])
#            print(theta_t,  alpha * v* cos(theta_t - theta))
        S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*W + alpha * v * cos(theta_t - theta)))
        S[S<0.00001] = 0.
        activities[:,t] = S
        if 10*t % time == 0:
            print("Process:" + str(100*t/time) + '%')
    S = ravel(S)
    return activities


def sim_dyn_one_d_random(N, extinp, inh, R, umax, dtinv,
              tau, time, ell, scale):
    W_dynamic, theta = make_connection_matrix(N, inh, R, ell)
    W, theta = make_connection_matrix(N, inh, R, 0)
    W_difference = W_dynamic - W
    W = sparse.csc_matrix(W)
    
    S = zeros(N)
    ## generate random activity (doesn't matter much)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
        
    activities = zeros([N,time])
    Stemp = zeros(N)
    
    for t in range(0, time, 1):
        if t % scale == 0:
#            gamma = np.random.choice([-1., 0., 1.])
            gamma = np.random.normal(0, .1)
#            gamma = 0.1
        if (t + int(scale/2.)) % scale == 0:    
            gamma = -gamma
        S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*(W + gamma * W_difference)))
        S[S<0.00001] = 0.
        activities[:,t] = S
        if 100*t % time == 0:
            print("Process:" + str(100*t/time) + '%')

#    S = ravel(S)
    return activities
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(activities)
#fig.colorbar(cax)
#plt.show()

######Data + codewords
#discretize
def bin_data(act, time, bsize, shift):
    """"Bin data into bins of size bsize with a possible shift"""
    nbin = int(time/bsize)
    b_act = zeros([act.shape[0],nbin])
    for i in range(nbin-shift):
        b_act[:,i]=np.sum(act[:,shift+i*bsize:shift+(i+1)*bsize],axis=1)
    return b_act

#detect spikes
def detect_spikes(act, factor):
    """Get spikes from Poisson process with threshold factor, adapted from Couey"""
    r = factor*np.random.rand(act.shape[0], act.shape[1])
    spiked_act = np.greater_equal(act, r)
    return spiked_act.astype(int)

#mean spike rate
def mean_spike_rate(act):
    return np.sum(act, axis=1)/(2*act.shape[1])+1./2
    
def spiking_probs(act, delta_t):
    # calculates spking probabilities
    msr = np.average(act, axis=1)
    return -1 + 2*msr*delta_t

#determine state
def determine_states(spike_act):
    #inactive state: -1, active state: +1
    states = zeros(spike_act.shape)
    states[spike_act>0] = 1 
    states[spike_act==0] = -1 
    return states


def calc_firing_probs(binned_act, time, bsize):
    #calculate firing probabilities
    nbin = int(time/bsize)
    probs = np.sum(binned_act, axis=1)/nbin
    return probs

def find_codewords(binned_act):
    #determines the frequencies of the patterns in binned neural data
    codewords = []
    for i in range(binned_act.shape[1]):
        col = binned_act[:,i]
        #look which neurons are active
        I = np.where(col[:] == 1) ###different value?
        codewords.append(I[0].tolist())
    
    return codewords


#find frequencies
def find_frequencies(codewords):
    tuple_codewords = map(tuple, codewords)
    freq_dict = Counter(tuple_codewords)
    freqs = np.array(sorted(list(freq_dict.values()),
                            reverse=True))
    
    return freqs #transpose?


##############ENERGY
#H1
def h1(h_coeffs, acts):
    #h is a spiking bias vector
    return -np.dot(h_coeffs, acts)

#H2 
def h2(j_coeffs, acts):
    #j_coeffs a functional coupling matrix
    return -np.sum(.5*np.multiply(j_coeffs, np.outer(acts, acts)))

#Hk
def hk(lambda_coeffs, acts):
    N = acts.shape[0]
    h = 0
    for k in range(1,N+1):
        h -= lambda_coeffs[k-1]*np.multiply(np.sum(acts),np.sum(acts))
    return(h)

        


###combine
def calc_energy(coeff_list, acts, e_funcs=[h1,h2]):
    #calculates the energy of some neural pattern(s) for some energy functions
    e = 0
    for i in range(len(e_funcs)):
        e += e_funcs[i](coeff_list[i], acts)
    return e

def calc_entropy(e_funcs, h, J, acts, beta):
    #calculates the entropy of some neural pattern(s)
    p_vec = calc_state_probs(acts, [h,J], beta)
    entropy = -np.sum(np.multiply(p_vec, np.log(p_vec)))
    return entropy

#calculate average energy
def average_E(e_funcs, coeff_list, acts):
    sum_E = 0
    nbin = acts.shape[1]
    for i in range(nbin):
        sum_E += calc_energy(coeff_list, acts[:,i])
        ave_E = sum_E/nbin
    return ave_E

###################Probs + MC

def calc_state_probs(all_states, h, J, beta):
    """Calculates the probabilities of all the possible patterns 
    for exact learning (suitable up to N=20)"""
    p_tot = 0
    p_vec = zeros(all_states.shape[0])
    for n, state in enumerate(all_states):
        e = calc_energy([h,J], state)
        #a modified probability distribution P (σ ) ∝ exp(−βE(σ ) + xσ T σ ∗ )?
        #reference state is one of the stored patterns
        #here: the "bumps"
        p = np.exp(-beta*e)
        p_vec[n] = p
        p_tot += p
    p_vec /= p_tot
    return p_vec

def calc_model_expecations(all_states, h, J, beta):
    """Calculates the average magnatization and 
    correlation of all the possible patterns """
    p_vec = calc_state_probs(all_states, h, J, beta)
    #calc expectation values
    model_exps = np.dot(all_states.T, p_vec)
    
    #calc correlations
    corrs = zeros([all_states.shape[1], all_states.shape[1]])
    for n, state in enumerate(all_states):
        corrs += np.outer(state, state)*p_vec[n]
#        for r in range(all_states.shape[1]):
#            for s in range(all_states.shape[1]):
#                corrs[r,s] += state[r]*state[s]*p_vec[n]
    
    return model_exps, corrs

#correlations (#covariance)
def calc_correlations(s_act, mag):
    #calculate correlations from neural activity
    Nneur = s_act.shape[0]
    nbin = s_act.shape[1]
    c = zeros([Nneur,Nneur])
    for i in range(nbin):
       c += np.outer(s_act[:,i]-mag, s_act[:,i]-mag)
    return c/nbin

def calc_correlations_without(s_act):
    #calculate correlations from neural activity
    Nneur = s_act.shape[0]
    nbin = s_act.shape[1]
    c = zeros([Nneur,Nneur])
    for i in range(nbin):
       c += np.outer(s_act[:,i], s_act[:,i])
    return c/nbin

def calc_third_order_corr(s_act, mag):
    #calculate third order correlations from neural activity
    Nneur = s_act.shape[0]
    nbin = s_act.shape[1]
    c = zeros([Nneur,Nneur,Nneur])
    
    for t in range(nbin):
        c += np.reshape(np.outer(np.reshape(np.outer(s_act[:,t]-mag,s_act[:,t]-mag), (Nneur, Nneur)), s_act[:,t]-mag), (Nneur, Nneur, Nneur))
#    for t in range(nbin):
#        for i in range(Nneur):
#            for j in range(Nneur):
#                for k in range(Nneur):
#                    c += (s_act[i,t]-mag)*(s_act[j,t]-mag)*(s_act[k,t]-mag)
#                    
    return c/nbin

#generalize calc_correlations to include arbitrary delay?
def calc_osd_corr(s_act, mag):
    #calculates the one-step-delayed correlation matrix
    Nneur = s_act.shape[0]
    nbin = s_act.shape[1]
    c = zeros([Nneur,Nneur])
    for i in range(nbin-1):
       c += np.outer(s_act[:,i+1]-mag,s_act[:,i]-mag)
    return c/(nbin-1)

def mc_step(N, h, J, current_state, Nflips, T):
    """Perform one step of Metropolis Monte Carlo
    as described in Landau-Monte Carlo Simulationsin Statistical Physic
    Chapter 4"""
    start_i = np.random.randint(0,N)
    if start_i+Nflips < N:
        indices = [ijk for ijk in range(start_i, start_i+Nflips)]
    else:
        indices = [ijk for ijk in range(start_i, N)]
        indices = indices + [ijk for ijk in range(-N+start_i, -N+Nflips)]
    for ijk in  indices:
#        for n in range(Nflips):
        #step 2
#            ijk = np.random.randint(0,N)
        
        #step 3
        #calculate energy of old state
        e_old = calc_energy([h, J], current_state)
        
        new_state = zeros(N)
        new_state[:] = current_state[:]
        new_state[ijk] =   - current_state[ijk]
        
        #calculate energy of new state
        e_new = calc_energy([h, J], new_state)
        
        #energy difference
        e_delta = e_new - e_old
        
        #step 4
        r = np.random.rand()
        
#            print(e_delta)
        #step 5
        if r < np.exp(- e_delta/T):
            current_state = new_state
    return current_state

def mc_step_2(N, h, J, current_state, e_old, T):
    """Same as mc_step but with Nflips=1"""
    ijk = np.random.randint(0,N)
    new_state = np.array(current_state, copy=True)
    new_state[ijk] =   - current_state[ijk]
    
    #calculate energy of new state
    e_new = -np.sum(.5*np.multiply(J, np.outer(new_state, new_state)))- np.dot(h, new_state)
    e_delta = e_new - e_old

    r = np.random.rand()
    if r < np.exp(- e_delta/T):
        current_state = new_state
        e_old = e_new
    return current_state, e_old

def mc_step_3(N, h, J, current_state, T):
    """Same as mc_step but with Nflips=1"""
    ijk = np.random.randint(0,N)
    flip=False
    
    #calculate energy of new state    
    e_delta = 2*current_state[ijk]*(h[ijk] + np.dot(J[:,ijk], current_state))
    if np.random.rand() < np.exp(-e_delta/T):
        current_state[ijk] = - current_state[ijk]
        flip=True
    return np.array(current_state), e_delta, flip

def metropolis_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, T):
    """Metropolis Monte Carlo simulation with spin flip
    Nsteps: maximal number of steps
    Nflips: is number of flips before choosing another starting point
    sample_after: sample after number of steps
    sample_per_steps sample every sample_per_steps steps"""
    N = h.shape[0]
    initial_state = np.random.rand(N)
    thr = 0.9
    initial_state[initial_state>thr] = 1 
    initial_state[initial_state<=thr] = -1
    mc_samples = zeros([N, Nsamples])
    energies = []
    Nsteps = int(Nsamples * sample_per_steps)
    current_state = initial_state
#    e_old = -np.sum(.5*np.multiply(J, np.outer(current_state, current_state)))- np.dot(h, current_state)
    for step in range(0, sample_after, 1):
        current_state, _, _ = mc_step_3(N, h, J, current_state, T)
    
    current_state, e_delta, _ = mc_step_3(N, h, J, current_state, T)
    mc_samples[:, 0] = current_state
    energy = calc_energy([h,J], current_state)
    energies = [energy]
    for step in range(1, Nsteps, 1):
        current_state, e_delta, flip = mc_step_3(N, h, J, current_state, T)
        
        if step % sample_per_steps == 0:
            mc_samples[:, int(step / sample_per_steps)] = current_state
            if flip:
                ener = energies[-1]+e_delta
            else:
                ener = energies[-1]
            energies.append(ener)
               
    return mc_samples, energies

def mc_with_start_pattern(h, J, Nsamples, initial_state, sample_after, sample_per_steps, T):
    """Metropolis Monte Carlo simulation with starting point"""
    N = h.shape[0]
    mc_samples = zeros([N, Nsamples])
    Nsteps = int(Nsamples * sample_per_steps)
    current_state = initial_state
    for step in range(0, sample_after, 1):
        current_state = mc_step_3(N, h, J, current_state, T)
    for step in range(0, Nsteps, 1):
        current_state = mc_step_3(N, h, J, current_state, T)
        
        if step % sample_per_steps == 0:
            mc_samples[:, int(step / sample_per_steps)] = current_state
               
    return mc_samples



def entropy_integrated(h, J, Nsamples, sample_after=1000,
                       sample_per_steps=None, dT=0.1):
    """
    dT = temp_step_size"""
    ###Overflow if dT=0.01
    temp = dT/2.  #variance is zero at T=0
    entropy =.0
    N = h.shape[0]
    if sample_per_steps==None:
        sample_per_steps = 2*N
    for i in range(int(1./dT)-1):
        _, energies = metropolis_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, temp)
        
        energy_var = np.var(energies)
        entropy += energy_var/temp**3*dT
        temp+=dT
    
    return entropy


def entropy_integrated_from_T1(h, J, Nsamples, max_steps, delta=0.01,
                               sample_after=1000, sample_per_steps=None, dT=0.1):
    """
    dT = temp_step_size"""
    temp = 1.  #variance is zero at T=0
    entropy =.0
    N = h.shape[0]
    if sample_per_steps==None:
        sample_per_steps = 2*N
    for i in range(max_steps):
        _, energies = metropolis_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, temp)
        
        energy_var = np.var(energies)
        delta_entropy = energy_var/temp**3*dT
        entropy += delta_entropy
        temp+=dT
        if delta_entropy < delta:
            break
    
    return entropy


def multi_information(h, J, Nsamples, sample_after=1000,
                       sample_per_steps=None, dT=0.1):
    N = h.shape[0]
    s_p_1 = entropy_integrated(h, zeros([N,N]), Nsamples, sample_after=sample_after,
                       sample_per_steps=sample_per_steps, dT=dT)
    
    s_p_12 = entropy_integrated(h, J, Nsamples, sample_after=sample_after,
                       sample_per_steps=sample_per_steps, dT=dT)
    
    return s_p_1 - s_p_12

##############LEM + MDS
#greedy descent dynamics
def gdd(h, J, initial_state=1, ordered_or_random='ordered', inverse=False):
    """for each neuron, we flip its activity if the flip will decrease the
    energy. If we could not decrease the energy by flipping any
    neuron’s activity, then a local energy minimum is identified"""
    Nneur = initial_state.shape[0]
    current_state = zeros(Nneur)
    current_state[:] = initial_state[:]
    tracked_states = [np.array(initial_state, copy=True)]
    n_flips = 0
    while True:
        e_old = calc_energy([h,J], current_state)
        
#       attempt to flip spins i~1,N from their current state into {s i , in order of increasing i.
        indices = np.arange(Nneur)
        if ordered_or_random == 'random':
            np.random.shuffle(indices)
        
        #random order of spin flip
#        indices = np.random.permutation(Nneur)
        stop_ind = 0
        for ind in indices:
            current_state[ind] = -current_state[ind]
            e_new = calc_energy([h,J], current_state)
            e_delta = e_new - e_old
#            e_delta = 2*current_state[ind]*(h[ind] + np.sum(np.dot(J[:,ind], current_state)))
            #uphill walk if True
            if inverse==True:
                e_delta = -e_delta
                
            if e_delta < 0:
                e_old = e_new
#                current_state[ind] = -current_state[ind]
                tracked_states.append(np.array(current_state, copy=True))
                n_flips += 1
            else:
                stop_ind += 1
                current_state[ind] = -current_state[ind]
                
            #stop if could not flip any spin during step
            if stop_ind == Nneur:
                return np.array(current_state), tracked_states, n_flips
#                return current_state
                
#    return current_state


def gdd_dyn(coeffs, initial_state, reference_state, max_steps):
    """calculate distance and entropy from reference (final) state
    when doing greedy descent dynamics"""
    [h,J] = coeffs
    acts = []
    Nneur = initial_state.shape[0]
    current_state = zeros(Nneur)
    current_state[:] = initial_state[:]
    new_state = current_state
    for step in range(max_steps):
#        d_list.append(hamming_distance(initial_state, reference_state))
#        s_list.append(calc_entropy([h1, h2], coeffs, current_state))
        indices = range(Nneur)

        for ind in indices:
            #ind = np.random.choice(range(N))
            trans_prob = np.exp(-2*current_state[ind]*(h[ind] + np.sum(.5*np.dot(J[:,ind], current_state))))
            r = np.random.rand()
            if r < trans_prob:
                new_state = current_state
                new_state[ind] = -current_state[ind]
            acts.append(new_state)    
            
            if np.all(new_state == reference_state):
                break
    return np.array(acts)


def lem(h, J, number_of_initial_patterns, init_part_active, ordered_or_random):
    """Determine local energy minima (for an Ising model)
    by Greedy Descent Dynamics (Huang and Toyoizumi, 2016)"""
    N = h.shape[0]
    patterns = []
    thr = 1 - init_part_active
    for i_p in range(number_of_initial_patterns):
#        initial_state = np.random.choice([-1,1], N)
        initial_state = np.random.rand(N)
        initial_state[initial_state>thr] = 1 
        initial_state[initial_state<=thr] = -1
        patt, _, _ = gdd(h, J, initial_state, ordered_or_random=ordered_or_random)
        patterns.append(patt)
    return patterns

def lem_init_final(h, J, number_of_initial_patterns):
    """same as lem but stores initial state-final state dictionary"""
    N = h.shape[0]
    init_final_dict = {}
    for i_p in range(number_of_initial_patterns):
        initial_state = np.random.choice([-1,1], N)
        final_state, _, _ = gdd(h, J, initial_state)
        try:
            init_final_dict[final_state.tobytes()].append(initial_state)
        except KeyError:
            init_final_dict[final_state.tobytes()] = [initial_state]
    return init_final_dict

def lem_from_data(h, J, s_act, ordered_or_random):
    """Determines LEM with GDD for all states from data"""
    init_final_dict = {}
    patterns = []
    for pattern in s_act.T:
        final_state, _, _ = gdd(h, J, pattern, ordered_or_random)
        patterns.append(final_state)
        try:
            init_final_dict[final_state.tobytes()].append(pattern)
        except KeyError:
            init_final_dict[final_state.tobytes()] = [pattern]
    return np.array(patterns), init_final_dict


###distance by changing states one-for-one and then using gdd to determine if they
#belonh to another basin now


def mc_with_gdd(h, J, init_pattern, lem_patterns, time_steps,
                T, transition_time_list = [], e_list = [], n_list = [],
                lems = [], mc_states = [], path_list = [],
                matrix_attempted_flips=[], matrix_both_flips = [],
                matrix_energy_barriers = [], basin_size_list = []):
    "Exploring the energy landscape Tkacik 2014"
    """
    init_pattern: pattern to start the MC with
    lem_patterns: identified LEMs 
    time_steps: number of MC steps (including attempted ones)
    T: temperature
    matrix_attempted_flips: for each pair of LEMS a list of the attempted 
    spin flips to get from one LEM to the other
    matrix_both_flips: matrix_attempted_flips + the spin flips needed 
    to reach the barrier (which we get from gdd)
    matrix_energy_barriers: list of the maximal energy encoundered 
    along the path between LEMs
    basin_size_list: size of the basins of the LEMs (number of patterns that
    reach this LEM as their minimum through gdd)
    """
    N = h.shape[0]
    num_patt = lem_patterns.shape[0]
    mc_states.append(init_pattern)
    lems.append(init_pattern) 
    if matrix_attempted_flips == []:
        matrix_attempted_flips = [[[] for i in range(num_patt)] for j in range(num_patt)]
    if matrix_both_flips == []:
        matrix_both_flips = [[[] for i in range(num_patt)] for j in range(num_patt)]
    if matrix_energy_barriers == []:
        matrix_energy_barriers = [[[] for i in range(num_patt)] for j in range(num_patt)]
    if basin_size_list == []:
        basin_size_list = [1 for i in range(num_patt)]
    time_spent_in_previous_basin = 0
    n_failed_attempts = 0
    current_state = init_pattern
    pi = np.where(np.dot(lem_patterns, init_pattern) == N)[0][0]
    t_at_previous_basin_crossing = -1
    for t in range(time_steps):
        current_state, e_delta, _ = mc_step_3(N, h, J, current_state, T)
        #ordered spin flips in Tkacik, but is that good?
        if np.any(current_state != mc_states[-1]):
            mc_states.append(current_state)
            e_list.append(e_delta)
            lem, path, n_flips = gdd(h,J, current_state, ordered_or_random='ordered')
            try:
                pj = np.where(np.dot(lem_patterns, lem) ==N)[0][0]
            except:#if lem was not in ordered_patterns
                #update lem patterns   
                lem_patterns = np.append(lem_patterns, np.reshape(lem, (1,N)), axis=0)
                
                #update size matrices
                matrix_attempted_flips.append([[] for i in range(num_patt)])
                for j in range(num_patt+1):
                    matrix_attempted_flips[j].append([])
                matrix_both_flips.append([[] for i in range(num_patt)])
                for j in range(num_patt+1):
                    matrix_both_flips[j].append([])
                matrix_energy_barriers.append([[] for i in range(num_patt)])  
                for j in range(num_patt+1):
                    matrix_energy_barriers[j].append([])   
                basin_size_list.append(1)
                num_patt += 1
                pj = np.where(np.dot(lem_patterns, lem) == N)[0][0]
            
            print(pj)
            basin_size_list[pj] += 1
            n_list.append(n_flips)    
            
            if np.any(lem != lems[-1]):
                transition_time_list.append(t-n_failed_attempts)
                time_spent_in_previous_basin = 0
                matrix_attempted_flips[pi][pj].append(t - t_at_previous_basin_crossing)
                matrix_both_flips[pi][pj].append(t - t_at_previous_basin_crossing + n_list[-1])
                matrix_energy_barriers[pi][pj].append(np.max(e_list[time_spent_in_previous_basin:]))
                path_list.append(path)
                lems.append(lem)
                pi = pj
                t_at_previous_basin_crossing = t
            else:
                time_spent_in_previous_basin += 1
        else:
            n_failed_attempts += 1
    return mc_states, lems, np.array(lem_patterns), e_list, n_list, matrix_attempted_flips, matrix_both_flips, matrix_energy_barriers, transition_time_list, path_list, basin_size_list
    
def get_transition_rates(h, J, lem_patterns, mc_time, T=1.):    
    transition_time_list = []
    e_list = []
    n_list = []
    lems = []
    mc_states = []
    path_list = []
    matrix_attempted_flips = []
    matrix_both_flips = []
    matrix_energy_barriers = []
    basin_size_list = []
    for lem in lem_patterns:
#        plot_single_pattern(lem)
        mc_states, lems, lem_patterns, e_list, n_list, matrix_attempted_flips, matrix_both_flips, matrix_energy_barriers, transition_time_list, path_list, basin_size_list = mc_with_gdd(h, J, lem, lem_patterns, mc_time,
                                                                                                                                                                    T, transition_time_list, e_list, n_list,
                                                                                                                                                                    lems, mc_states, path_list,
                                                                                                                                                                    matrix_attempted_flips, matrix_both_flips,
                                                                                                                                                                    matrix_energy_barriers, basin_size_list)
    return lem_patterns, matrix_attempted_flips, matrix_both_flips, matrix_energy_barriers, path_list, basin_size_list
    
    

def cross_val(s_act, reg_lambda_list, max_steps,
              h_lambda, J_lambda, reg_method, epsilon, T,
              Nsamples, sample_per_steps, sample_after=1000):
    N = s_act.shape[0]
    cross_n = len(reg_lambda_list)
    dl = s_act.shape[1]
    data_chunks = [s_act[:,int(i*dl/cross_n):int((i+1)*dl/cross_n)] for i in range(cross_n)]

    errors = zeros((cross_n, cross_n))
    for k, reg_lambda in enumerate(reg_lambda_list):
        
        for j in range(cross_n):
            data = np.reshape(np.array([data_chunks[i] for i in range(cross_n) if i!=j]), (N, int((cross_n-1)*dl/cross_n)))
            h = zeros(N)
            J = np.random.uniform(-.5, .5, [N,N])
            h, J, min_av_max_plm = plm_separated(data, max_steps,
                                    h, J, h_lambda, J_lambda,
                                    reg_method, reg_lambda, epsilon, 1.)
                        
            
            mag_sim = np.average(data_chunks[j], axis=1)
            corrs_sim = calc_correlations(data_chunks[j], mag_sim)
            corrs3_sim = calc_third_order_corr(data_chunks[j], mag_sim)
            
            s_act_inferred = metropolis_mc(h, J, Nsamples,
                          sample_after, sample_per_steps, 1.)
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
            
            error = np.sqrt((np.sum(np.square(mag_sim - mag_inf)) + np.sum(np.square(corrs_sim - corrs_inf)) + np.sum(np.square(corrs3_sim - corrs3_inf)))/float(N+N*N+N*N*N))
            errors[k, j] = error
            
            print(error)
    
    ave_err = np.average(errors, axis=1)
    min_reg_lambda = reg_lambda_list[np.where(np.min(ave_err)==ave_err)[0]]
    return min_reg_lambda, errors

def cross_val_all(s_act, reg_lambda_list, max_steps,
              h_lambda, J_lambda, reg_method, epsilon, T,
              Nsamples, sample_per_steps, sample_after=1000):
    N = s_act.shape[0]
    cross_n = len(reg_lambda_list)
    
    mag_sim = np.average(s_act, axis=1)
    corrs_sim = calc_correlations(s_act, mag_sim)
    corrs3_sim = calc_third_order_corr(s_act, mag_sim)

    h_list = []
    J_list = []
    errors = zeros((cross_n))
    for k, reg_lambda in enumerate(reg_lambda_list):
        
        h = zeros(N)
        J = np.random.uniform(-.5, .5, [N,N])
        h, J, min_av_max_plm = plm_separated(s_act, max_steps,
                                h, J, h_lambda, J_lambda,
                                reg_method, reg_lambda, epsilon, 1.)
        h_list.append(h)
        J_list.append(J)
        s_act_inferred = metropolis_mc(h, J, Nsamples,
                      sample_after, sample_per_steps, 1.)
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
        
        error = np.sqrt(np.average(np.square(mag_sim - mag_inf)) + np.average(np.square(corrs_sim - corrs_inf)) + np.average(np.square(corrs3_sim - corrs3_inf)))
        errors[k] = error
        
        print(error)
    
    min_reg_lambda_index = np.where(np.min(errors)==errors)[0][0]
    min_reg_lambda = reg_lambda_list[min_reg_lambda_index]
    min_h = h_list[min_reg_lambda_index]
    min_J = J_list[min_reg_lambda_index]
    return min_reg_lambda, min_h, min_J, errors

def hamming_distance(sigma_1, sigma_2):
    #Calculates the Hamming distance between two neural patterns
    N = sigma_1.shape[0]
    return (N - np.sum(np.multiply(sigma_1, sigma_2)))/2

def sampled_distance_from_lem(sigma, lem):
    #Calculates the Hamming distance between a reference pattern
    # and another neural pattern
    N = sigma.shape[0]
    return (1- np.sum(np.multiply(sigma, lem)/N))/2

def B_matrix(data_points, c):
    #calculate B matrix for multidimensional scaling
    #see Cox-Multidimensional scaling analysis
    n_dp = len(data_points)
    H = np.identity(n_dp) - 1/n_dp*np.outer(ones(n_dp), ones(n_dp))
    A = zeros([n_dp,n_dp])
    for r, dp_r in enumerate(data_points):
        for s, dp_s in enumerate(data_points):
#        for s, dp_s in enumerate(data_points[r:]):
            delta_rs = hamming_distance(np.array(dp_r), np.array(dp_s))
            A[r,s] = -delta_rs**2/2 + c*(1-delta_rs)
#            A[r,s] = A[s,r]
    #B = HAH        
    return np.dot(H, np.dot(A, H))

def mds(data_points, p, option):
    """reduce data to p dimensions with multidimensional scaling
    option 1 ignores the negative values,
    option 2 adds an appropriate constant c to the dissimilarities """
    
    #As in Multidimensional scaling analysis, Cox
    B = B_matrix(data_points, c=0)
    eig_val, eig_vec = np.linalg.eig(B)
#    np.less_equal(eig_val, 0) #?
#    ignore the negative values and proceed
    if option == 1:
        eig_vec = eig_vec[eig_val >= 0]
    
#    add an appropriate constant c to the dissimilarities
    if option == 2:
        #c=-2lambda_n where
        #lambda_n, is the smallest eigenvalue of B
    #            algebraically smallest eigenvalue = smallest (in terms of absolute value)?
        
        c = -2*eig_val[np.argmin(np.abs(eig_val))]
        B = B_matrix(data_points, c)
        eig_val, eig_vec = np.linalg.eig(B)
    #    #    X = np.dot(eig_vec, np.sqrt(eig_val))
    
    X = np.array(eig_vec) # np.dot(np.array(eig_vec), np.sqrt(np.diag(eig_val)))#?
    return X[:, 0:p]
    

####################PLM
def exp_sigma(act_t, rth, h, J, T):
    #conditional probability of one variable given all the others
    beta=1./T
    sum_without_r = np.sum(np.delete(np.multiply(J[:,rth], act_t), rth, 0))
    return np.exp(-2*beta*act_t[rth]*(h[rth]+sum_without_r))

def p_sigma(act_t, r, h, J, T):
    """calculated The conditional probability of one variable sigma_r
    given all the others"""
    "act_t is one d vector (at time t)"
    "beta=1/T"
    e = exp_sigma(act_t, r, h, J, T)
    return 1/(1+e)


#def f_prime_h(sigmas, h_prime, J_prime, T):
#    "gradient of all the f_r's wrt all the h_r's (respectively)"
#    beta = 1/T
#    f_primes = zeros(sigmas.shape[0])
#    for sigma in sigmas.T:
#        for r in range(sigmas.shape[0]):
#            p_s = p_sigma(sigma, r, h_prime, J_prime, T)
#            e_s = exp_sigma(sigma, r, h_prime, J_prime, T)
#            f_primes[r] += 2*p_s * e_s * sigma[r]/T
#    f_primes /= sigmas.shape[1]
#    return f_primes
    
def f_prime_seq(sigmas, h_prime, J_prime, reg_lambda, T):
    #calculates derivative for plm_algorithm sequential gradient descent
    Nneur = sigmas.shape[0]
    f_primes_h = zeros(Nneur)
    f_primes_J = zeros([Nneur,Nneur])
    for sigma in sigmas.T:
        for r in range(sigmas.shape[0]):
            e_s = exp_sigma(sigma, r, h_prime, J_prime, T)
            p_s = 1/(1+e_s)
            f_primes_h[r] -= 2*p_s * e_s * sigma[r]/T
            
            norm_J_r = np.sum(J_prime[:, r])
            for s in range(sigmas.shape[0]):
                if s != r:
                    f_primes_J[s,r] -= 2*p_s*e_s*sigma[r]*sigma[s]/T  
    f_primes_J /= sigmas.shape[1]
    f_primes_h /= sigmas.shape[1]
    f_primes_J[s,r] += reg_lambda*J_prime[s, r]
    return np.nan_to_num(f_primes_h), np.nan_to_num(f_primes_J)


def f_prime_NR(sigmas, h_prime, J_prime, reg_lambda, T):
    #calculates derivative for plm_algorithm Newton Raphson
    beta = 1./T
    Nneur = sigmas.shape[0]
    f_primes_h = zeros(Nneur)
    f_primes_J = zeros([Nneur,Nneur])
    hessian = zeros([Nneur+ Nneur * Nneur, Nneur + Nneur * Nneur])
    for sigma in sigmas.T:
        for r in range(sigmas.shape[0]):
            e_s = exp_sigma(sigma, r, h_prime, J_prime, T)
            p_s = 1/(1+e_s)
            f_primes_h[r] -= 2*p_s * e_s * sigma[r]/T
            
            norm_J_r = np.sum(J_prime[:r])
            hessian[r, r] += 4*beta**2*e_s*(1/(1+e_s) + 1)/(1+e_s)
            for s in range(sigmas.shape[0]):
                if s != r:
                    f_primes_J[s,r] -= 2*p_s*e_s*sigma[r]*sigma[s]/T + reg_lambda*J_prime[s, r]/norm_J_r
                 
                    hessian[r,Nneur+ s*Nneur+r] += 4*beta**2*e_s*sigma[s]*(1/(1+e_s) + 1)/(1+e_s)
                    hessian[Nneur+ s*Nneur+r, r] += 4*beta**2*e_s*sigma[s]*(1/(1+e_s) + 1)/(1+e_s)
                    for t in range(sigmas.shape[0]):
                        hessian[Nneur+s*Nneur+r, Nneur+t*Nneur+r] += 4*beta**2*e_s*sigma[s]*sigma[t]*(1/(1+e_s) + 1)/(1+e_s) 
    f_primes_J /= sigmas.shape[1]
    f_primes_h /= sigmas.shape[1]
    hessian /=  sigmas.shape[1]
    eye = np.eye(hessian.shape[0])
    print(np.sum(hessian))
    hess_inv = np.linalg.inv(hessian + eye*10**(-6)) #singular matrix
    h_and_J = np.concatenate((np.nan_to_num(f_primes_h),
                                         np.nan_to_num(f_primes_J).flatten()))
    full_primes = np.dot(hess_inv, h_and_J)
    f_primes_h = full_primes[:Nneur]
    f_primes_J = full_primes[Nneur:]
    return np.nan_to_num(f_primes_h), np.reshape(np.nan_to_num(f_primes_J), (Nneur, Nneur))

def plm_algorithm(sigmas, max_steps, h, J, h_lambda, J_lambda,
                  reg_lambda, epsilon, T):
    "Pseudo_likelihood Maximization"
    "method = NR: Newton-Raphson, method = seq: sequential gradient descent"
    min_av_max = [] 
    for step in range(max_steps):
        #learning step
        h_primes, J_primes = f_prime_seq(sigmas, h, J, reg_lambda, T)
        h -= h_lambda*h_primes        
        J -= J_lambda*J_primes
        np.fill_diagonal(J_primes, 0)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J)]))
        
        if np.sum(h_primes**2) + np.sum(J_primes**2) < epsilon: 
            break
        
        print("Step", step, "Error:", np.sum(h_primes**2) + np.sum(J_primes**2))
        
    A = (np.triu(J, k=1) + np.tril(J, k=-1).T)/2
    A += A.T
    J = A
    return h, J, np.array(min_av_max)

def plm_separated(sigmas, max_steps, h, J, h_lambda, J_lambda,
                  reg_method, reg_lambda, epsilon, T):
    """Pseudo_likelihood Maximization with gradient descent
    with l1 (sign and lasso) and l2 regularisation
    sigmas: neural data
    max_steps: maximal number of steps
    h: initial h
    J: initial coupling matrix J
    h_lambda: learning constant for h parameters
    J_lambda: learning constant for J parameters
    reg_method: regularisation method: 
        -'sign' for l1 with sign approximation
        -'lasso' for l1 with lasso 
        -'l2' for l2 regularisation
    reg_lambda: parameter for regularisation
    epsilon: stopping condition
    T: temperature
    J_org: original matrix to track reconstruction error
    """
    Nneur = sigmas.shape[0]
    min_av_max = []
    previous_loss = 0.
    for step in range(max_steps):
        #learning step
        h_primes = zeros(Nneur)
        J_primes = zeros([Nneur,Nneur])
        total_loss = 0.
        for r in range(Nneur):
            params = np.append(h[r], J[:,r])
            
            h_primes[r], J_primes[:,r], loss = f_prime_r(sigmas, r, params,
                                                    reg_method, reg_lambda, T)
            total_loss += loss
        np.fill_diagonal(J_primes, 0)
        h -= h_lambda*h_primes        
        J -= J_lambda*J_primes
#        rec_err = reconstruction_error(J_org, J)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J)]))
#        if step % 10 == 0:
        A = (np.triu(J, k=1) + np.tril(J, k=-1).T)/2
        A += A.T
        J = A
        if step % 10 == 0:
            print("Step: "+str(step),"Min J: " + str(np.min(J)), "Max J: " + str(np.max(J)))
            print("lambda", h_lambda)
#            print("Min d_J: " + str(np.min(J_primes)), "Max d_J: " + str(np.max(J_primes)))
        if step != 0 and (min_av_max[-2][3] > np.min(J) or min_av_max[-2][-1] < np.max(J)):
            h_lambda *= .99
            J_lambda *= .99
#            print(h_lambda)
#        if total_loss < previous_loss:
#            h_lambda *= 1.05
#            J_lambda *= 1.05
#        elif total_loss > previous_loss + 10**-10: 
#            #undo weight change
#            h += h_lambda*h_primes        
#            J += J_lambda*J_primes
#            h_lambda *= .85
#            J_lambda *= .85

        
        previous_loss = total_loss
        if np.all(np.abs(J_primes) < epsilon) and np.all(np.abs(h_primes) < epsilon):
            break
        if h_lambda < 10**-4:
            break
    print("lambda", h_lambda)
    return h, J, np.array(min_av_max)

def f_prime_r(sigmas, rth, params, reg_method, reg_lambda, T):
    #calculates derivative of f_r (as in Aurell, 2012)
    Nneur = sigmas.shape[0]
    f_primes_J = zeros(Nneur)
#    hessian = zeros([Nneur+1,  Nneur+1])
    
    e_s = exp_sigma_r(params, sigmas, rth, T)
    pes = e_s/(1+e_s)
    f_primes_h = -np.sum(2* pes * sigmas[rth,:]/T)
    
    loss = 0
    Jnorm = np.linalg.norm(params[1:])
    for s in range(Nneur):
        Jsr = params[1:]
        f_primes_J[s] -= np.sum(2*pes*sigmas[rth,:]*sigmas[s,:]/T) 
        loss += -np.sum(np.log(1/(1+e_s)))
#        upp_hess = np.sum(4*beta**2*e_s*sigma[s,:]*(1/(1+e_s) + 1)/(1+e_s))
#        hessian[0, s+1] = upp_hess
#        hessian[s+1, 0] = upp_hess
#        for t in range(Nneur):
#            hessian[t+1, s+1] = np.sum(4*beta**2*e_s*sigma[s,:]*sigma[t,:]*(1/(1+e_s) + 1)/(1+e_s))
            
#    np.fill_diag(hessian, 0)
#    hessian[0, 0] += 4*beta**2*e_s*(1/(1+e_s) + 1)/(1+e_s)
    f_primes_J /= sigmas.shape[1]
    f_primes_h /= sigmas.shape[1]
    loss /= sigmas.shape[1]
    ###l1-reg
    if reg_method == "sign":
        f_primes_J += reg_lambda*np.sign(params[1:])
        
    if reg_method == "lasso":
#        if np.abs(f_primes_h)<=reg_lambda:
#            f_primes_h = 0
#        elif f_primes_h>reg_lambda:
#            f_primes_h -= reg_lambda
#        else: 
#            f_primes_h += reg_lambda
        f_primes_J[np.abs(f_primes_J)<=reg_lambda] = 0 
        f_primes_J[f_primes_J>reg_lambda] -= reg_lambda
        f_primes_J[f_primes_J<reg_lambda] += reg_lambda
        
    ###l2-reg
    if reg_method == "l2":
        f_primes_J += reg_lambda*params[1:]
#    if hessian == True:
#        join_ = np.concatenate((f_primes_h, f_primes_J))
#        a = np.dot(np.linalg(hessian), join_)
#        f_primes_h = a[0]
#        f_primes_J = a[1:]
    return f_primes_h, f_primes_J, loss

def exp_sigma_r(params, sigmas, rth, T):
    rth = int(rth)
    act_without_r = np.delete(sigmas, rth, 0)    
    sum_without_r = np.dot(act_without_r.T, np.delete(params[1:], rth, 0))
    return np.exp(-2*sigmas[rth,:]*(params[0]+sum_without_r)/T)

def batch_plm(sigmas, max_steps, batch_size, h, J, h_lambda,
                  J_lambda, reg_method, reg_lambda, epsilon, T, J_org):
    "Stochstic Pseudo_likelihood Maximization"
    Nneur = sigmas.shape[0]
    min_av_max = []
    B_tot = sigmas.shape[1]
    for step in range(max_steps):
        np.random.shuffle(np.transpose(sigmas))
        for b in range(int(B_tot/batch_size)):
            h_primes = zeros(Nneur)
            J_primes = zeros([Nneur,Nneur])
            for r in range(Nneur):
                params = np.append(h[r], J[:,r])
                batch = sigmas[:, b*batch_size:(b+1)*batch_size]
    
                h_primes[r], J_primes[:,r] = f_prime_r(batch, r, params,
                                                    reg_method, reg_lambda, T)
            h -= h_lambda*h_primes        
            J -= J_lambda*J_primes
            
        rec_err = reconstruction_error(J_org, J)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J), rec_err]))
        
        print("Step",step," Rec. Error:", rec_err)
#    A = (np.triu(J_primes, k=1) + np.tril(J_primes, k=-1).T)/2
#    A += A.T
#    J = A
    return h, J

def exp_sigma_r_min(params, sigmas, arth, reg_lambda, T):
    "Calculates exponent for f_prime_r_min"
    arth = int(arth)
    h_r = params[0]
    J_r = params[1:]
    act_without_r = np.delete(sigmas, arth, 0)
    sum_without_r = np.dot(act_without_r.T, J_r)
    return np.average(np.exp(-2*sigmas[arth,:]*(h_r+sum_without_r)/T)) + reg_lambda*np.linalg.norm(params[1:])

def f_prime_r_min(params, sigmas, arth, reg_lambda, T):
    "Calculates derivatives for plm_min"
    Nneur = sigmas.shape[0]
    gradient = zeros(Nneur+1)
    arth = int(arth)
    e_s = exp_sigma_r(params, sigmas, arth, T)
    pes = e_s/(1+e_s)
    h_pr = np.sum(2* pes * sigmas[arth,:]/T)

    Jnorm = np.linalg.norm(params[1:])
    for s in [s for s in range(Nneur) if s != arth]:
        if s != arth:
            a = 2*pes*sigmas[arth,:]*sigmas[s,:]/T
            Jsr = params[1:]
            gradient[s] -= np.sum(2*pes*sigmas[arth,:]*sigmas[s,:]/T) - reg_lambda*Jsr[s]
    
    gradient[arth] = 0
    gradient[0] = h_pr
    gradient /= sigmas.shape[1]
    return gradient

def plm_min(sigmas, h, J, reg_lambda, T):
    "Pseudo_likelihood Maximization with scipy minimizer BFGS"
    Nneur = sigmas.shape[0]
#    params = zeros([Nneur,Nneur])
#    params[:,0] = h
#    params[:,1:] = J[~np.eye(J.shape[0],dtype=bool)].reshape(J.shape[0],-1)

    for rth in range(Nneur):
        print(rth)
        params = np.append(h[rth], np.delete(J[:,rth], rth))
        min_params_r = optimize.fmin_bfgs(exp_sigma_r_min, params, args=(sigmas, reg_lambda, rth, T))
#        scipy.optimize.minimize(exp_sigma_r_min, params, args=(sigmas, reg_lambda, rth, T), method='SLSQP', jac=None, bounds=None, constraints=())
#        min_params_r = optimize.minimize(exp_sigma_r_min, params, args=(sigmas, reg_lambda, rth, T), method="Nelder-Mead", jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        h[rth] = min_params_r[0]
        J[:, rth] = np.append(min_params_r[1:rth+1], np.append(0, min_params_r[rth+1:]))
    
    A = (np.triu(J[:,:], k=1) + np.tril(J[:,:], k=-1).T)/2
    A += A.T
    J = A
    return h, J

def exp_sigma_r_min(params, sigmas, arth, reg_lambda, T):
    arth = int(arth)
    h_r = params[0]
    J_r = params[1:sigmas.shape[0]]
    act_without_r = np.delete(sigmas, arth, 0)
    sum_without_r = np.dot(act_without_r.T, J_r)
    a = np.average(np.exp(-2*sigmas[arth,:]*(h_r+sum_without_r)/T)) + reg_lambda*np.dot(ones(sigmas.shape[0]-1).T, params[sigmas.shape[0]:])
    return a

def plm_sqp(sigmas, h, J, reg_lambda, T):
    "Pseudo_likelihood Maximization with scipy minimizer SLSQP"
    Nneur = sigmas.shape[0]
#    params = zeros([Nneur,Nneur])
#    params[:,0] = h
#    params[:,1:] = J[~np.eye(J.shape[0],dtype=bool)].reshape(J.shape[0],-1)

    for rth in range(Nneur):
        print(rth)
        params = np.append(np.append(h[rth], np.delete(J[:,rth], rth)), 0.01*ones(Nneur-1))
        a =  reg_lambda*np.dot(ones(sigmas.shape[0]-1).T, params[sigmas.shape[0]:])
#         inequality means that it is to be non-negative
        cons_per_i = [{'type':'ineq', 'fun': lambda params, i=i: cons_i(params, Nneur, i)} for i in np.arange(Nneur-1)]
        min_params_r = optimize.minimize(exp_sigma_r_min, params, args=(sigmas, reg_lambda, rth, T), method='SLSQP', jac=None, bounds=None, constraints=(cons_per_i))
#        print(min_params_r.x)
        h[rth] = np.nan_to_num(min_params_r.x[0])
        J[:, rth] = np.nan_to_num(np.append(min_params_r.x[1:rth+1], np.append(0, min_params_r.x[rth+1:Nneur])))
    
    A = (np.triu(J[:,:], k=1) + np.tril(J[:,:], k=-1).T)/2
    A += A.T
    J = A
    return h, J

def cons_i(params, Nneur, i):
    w = params[1:Nneur]
    u = params[Nneur:]
    return u[i] - np.abs(w[i])

####exact learning
def exact_learning(N, s_act, max_steps, l_rate, h, J, beta):
    """Exact learning considering all possible patterns"""
    all_states = np.array([list(seq) for seq in itertools.product([-1,1],
     repeat=N)])
    #stopping condition?
    mag = np.average(s_act, axis=1)
    corrs = calc_correlations(s_act, mag)
    
    min_av_max = []
    for step in range(max_steps):
        model_exps, model_corrs = calc_model_expecations(all_states, h, J, beta)
        h += l_rate*(mag - model_exps)
        J += l_rate*(corrs - model_corrs)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J)]))

    return h, J, min_av_max

#memory efficient
def exact_learning_mem(N, s_act, max_steps, l_rate, h, J, beta):
    """Exact learning considering all possible patterns more memory friendly"""
    #stopping condition?
    mag = np.average(s_act, axis=1)

    corrs = zeros([N,N])
    for i in range(s_act.shape[1]):
       corrs += np.outer(s_act[:,i], s_act[:,i])
    corrs /= s_act.shape[1]
    
    min_av_max = []
    for step in range(max_steps):
        model_exps, model_corrs = calc_exp_corr(N, h, J, beta)
        h += l_rate*(mag - model_exps)
        J += l_rate*(corrs - model_corrs)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J)]))
    return h, J, np.array(min_av_max)


def calc_exp_corr(N, h, J, beta):
    #calculates mean magnetixzation (<s_i>, expectation)
    #and mean correlation  <s_is_j>
    p_tot = 0
    model_exps = zeros(N)
    corrs = zeros([N, N])
    for n in range(0, 2**N, 1):
        string_state = format(n, 'b').zfill(N)
        array_state = np.array([int(s) if s=="1" else -1 for s in string_state])
        e = .5*np.sum(np.multiply(J, np.outer(array_state, array_state)))+np.dot(h, array_state)
        prob = np.exp(beta*e)
        p_tot += prob
        
        #calc contribution of state to expectation values
        model_exps += prob * array_state
        
        #calc correlations
        corrs += np.outer(array_state, array_state)*prob
        
#        if n % 64 == 0:
#            print(n, e, prob, p_tot)
    
    #normalize
    model_exps /= p_tot    
    corrs /= p_tot
    return model_exps, corrs

def boltzmann_learning(s_act, max_steps, l_rate, h, J, Nsamples,
                  sample_after, sample_per_steps, epsilon=10**-3, T=1.):
    """Boltzmann Learning with Monte Carlo sampling to determine 
    model expectations and correlations"""
    N = s_act.shape[0]
    mag = np.average(s_act, axis=1)
    corrs = calc_correlations_without(s_act)
    np.fill_diagonal(corrs, 0.)
    
    min_av_max = []
    error_list = []
    for step in range(max_steps):
        model_exps, model_corrs = calc_exps_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, T)
        
        h += l_rate*(mag - model_exps)
        J += l_rate*(corrs - model_corrs)
        min_av_max.append([np.min(h), np.average(h), np.max(h), np.var(h),
                           np.min(J), np.average(J), np.max(J), np.var(J)])
    
        error = np.sqrt((np.sum(np.square(mag - model_exps)) + np.sum(np.square(corrs - model_corrs)))/float(N+N*N))
        error_list.append(error)
#        l_rate = np.min([error, 0.05])
        l_rate = np.exp(-step/(0.5*max_steps)-2.25)
        if np.all(np.abs(mag - model_exps) < epsilon)  and np.all(np.abs(corrs - model_corrs) < epsilon):
            break

    return h, J, np.array(min_av_max), error_list

def calc_exps_mc(h, J, Nsamples, sample_after, sample_per_steps, T):
    "Calculates average magnetization and correlations for Boltzmann Learning"
    mc_samples, _ = metropolis_mc(h, J, Nsamples,
                  sample_after, sample_per_steps, T)
    model_exps = np.average(mc_samples, axis=1)
#    print(model_exps.shape)
    
    c = zeros([h.shape[0],h.shape[0]])
    for i in range(Nsamples):
       c += np.outer(mc_samples[:,i], mc_samples[:,i])
    model_corrs =  c/Nsamples
    np.fill_diagonal(model_corrs, 0.)
    return model_exps, model_corrs


#all_states = np.array([list(seq) for seq in itertools.product([-1,1],
#     repeat=Nexact)])
#Nexact = 10
#h, J = nMF(b_act, "tap")
#e_funcs = [h1,h2]
#idx = np.random.choice(N, Nexact, replace=False)
#h_small = h[idx]
#J_small = J[idx, :]
#J_small = J_small[:, idx]
#spiked_act_small = spiked_act[idx]
#h_coeffs, J_coeffs = exact_learning(all_states, spiked_act_small,
#                                    1000, 0.01, [h_small, J_small], 1)

######Native Mean Field
def nMF(s_act = 0, approx_type = ""):
    """Mean Field approximation of inverse Ising 
    (Hertz Ising Models for Inferring Network Structure from Spike Data)"""
    mag = np.average(s_act, axis=1)
    C = calc_correlations(s_act, mag)
    #include 1/beta?
    J = -np.linalg.inv(C)
    np.fill_diagonal(J, 0)
    h = np.arctanh(mag) - np.dot(J, mag) 
    
#    else:
#        J = np.dot(np.linalg.inv(A), np.dot(D, np.linalg.inv(C)))
    
    if approx_type == "tap":
        A = np.diag(1-np.square(mag))
        D = calc_osd_corr(s_act, mag)
        F = np.multiply(1-np.square(mag),
                        np.sum(np.multiply(np.square(J),
                                           1-np.square(mag)), axis=1))
        
        roots = np.array([np.roots([1,-2,1,-xi]) for xi in F.tolist()])
        lowest_roots = np.array([np.sort_complex(xi)[0] for xi in roots.tolist()])

        #.real?
        A = np.multiply(lowest_roots.real, np.diag(1-np.square(mag)))
        J = np.dot(np.linalg.inv(A), np.dot(D, np.linalg.inv(C)))
        np.fill_diagonal(J, 0)
        h = np.arctanh(mag) - np.dot(J, mag) + np.multiply(mag, np.dot(np.square(J),  1-np.square(mag)))
    
    return h,J

def reconstruction_error(original_J, inferred_J):
    """Calculates reconstruction error as in Aurell 2012 """
    return np.sqrt(original_J.shape[0])*np.sqrt(np.average(np.square(original_J - inferred_J)))

def sherrington_kirkpatrick(N, p):
    """Sherrington-Kirkpatrick model: every J_ij is nonzero with probability p
    and if so drawn from a Gaussian distribution with zero mean 
    and variance 1=c, c/pN"""
    J = zeros([N,N])
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < p:
                coupling = np.random.normal(0, 1/np.sqrt(p*N))
                J[i,j] = coupling
                J[j,i] = coupling
                
    return J

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


#writer1 = FFMpegWriter(fps=15, metadata=dict(title=''))
#fig = plt.figure(2)
#num_sam = 250
#with writer.saving(fig, "head_act_2.mp4", num_sam):
#    for i in range(num_sam):
#        plt.clf()
#        ax = fig.add_subplot(1,2,1)
#        cax = ax.matshow(s_act[:,10*i:10*(i+1)])
#        #plt.show()
#        
#        ax = fig.add_subplot(1,2,2)
#        ang=angles[2*i]
#        x0 = cos(ang)*0.5
#        y0 = sin(ang)*0.5
#        ax.plot([0,x0], [0,y0])
#        ax.axis([-0.5, 0.5, -0.5, 0.5])
#        
#        writer.grab_frame()
    
#
#length = 500
#def animate(i):
#    im.set_data(np.reshape(s_act[:,i*length:(i+1)*length], (N,length)))
#    return im,
#fig = plt.figure()
#im =  plt.imshow(np.reshape(s_act[:,0:length], (N,length)), animated=True)
#plt.colorbar()
#def init():  
#    im.set_data(np.reshape(s_act[:,0:length], (N,length)))
#    return im,
###
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                   frames=20, interval=100, blit=True)
##
#anim.save("s_act_8.mp4", fps=5, extra_args=['-vcodec', 'libx264'])





##########plotting functions
def make_expected_patterns(N, n_bumps, length_bump):
    ##look for expected pattern energies and check for local energy minimum
    # exp_patterns.shape(npatterns, N)
    num_patts = int(N/n_bumps)
    exp_patterns = -ones([N,num_patts]) # second value half if two bumps
    shift = int((N-length_bump*n_bumps)/n_bumps) #int((N - 2*lenght)/4)
    for i in range(num_patts): #half if two bumps
        for n in range(length_bump):
            for b_i in range(n_bumps):
                exp_patterns[i, (n +  b_i*(length_bump + shift) + i)% N] = 1
    return exp_patterns
    

def plot_patterns_with_energies(h, J, patterns):
    pattern_energies = []
    for i,pattern in enumerate(patterns):
        energy = calc_energy([h,J], pattern)
        pattern_energies.append(round(energy,2))
        
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(patterns)
    ax.set_yticklabels(['']+pattern_energies)
    ax.set_yticks([i for i in np.arange(-.5, len(pattern_energies), 1.)])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.plot(pattern_energies, 'x', label="Energies of the expected patterns")
    ax.legend()
    ax.set_ylabel("Energy")
    ax.set_xlabel("Pattern")
    plt.show()
    
    return pattern_energies
    
    
def plot_patternsfromexpected_with_energies(h, J, exp_patterns, n_bumps):
    N = exp_patterns.shape[1]
    num_patts = int(N/n_bumps)
    expected_pattern_energies = []
    lems_expected_patterns = []
    lems = zeros([N,num_patts])
    for i,pattern in enumerate(exp_patterns):
        energy = calc_energy([h,J], pattern)
        expected_pattern_energies.append(round(energy,2))
        lem_patt = gdd(h, J, pattern)
        lems[:,i] = lem_patt
        lems_expected_patterns.append(calc_energy([h,J], lem_patt))
        
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(lems.T)
    ax.set_yticklabels(['']+expected_pattern_energies)
    ax.set_yticks([i for i in np.arange(-.5, len(expected_pattern_energies), 1.)])
    ax.set_ylabel("Pattern")
    ax.set_xlabel("Neuron")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.plot(expected_pattern_energies, label="Energies of the expected patterns")
    cax = ax.plot(lems_expected_patterns, label="Energies of the LEMs of these patterns")
    ax.set_ylabel("Energy")
    ax.set_xlabel("Pattern")
    ax.legend()
    plt.show()

    print( "Energy difference:", min(expected_pattern_energies) - max(expected_pattern_energies))
    return lems

def plot_ordered_patterns(patterns_gdd, h, J):
    "ordered_patterns.shape = (number of patterns, number of neurons)"
    N = h.shape[0]
    tuple_codewords = map(tuple, patterns_gdd)
    freq_dict_gdd = Counter(tuple_codewords)
    code_probs_gdd = np.array(list(sorted(freq_dict_gdd.values(),reverse=True)), dtype="float64")/np.sum(list(freq_dict_gdd.values()))
    
#    indexed = sorted(range(len(freq_dict_gdd.values())), key=lambda k: list(freq_dict_gdd.values())[k])
    #indexed_patterns = [list(freq_dict_gdd.keys())[i] for i in indexed]
    
    stored_energies = []
    oel = []
    n_oel = []
    energies = []
    for pattern in freq_dict_gdd.keys():
        energy = calc_energy([h,J], pattern)
        stored_energies.append(energy)
        oel.append("Prob. %.3f, Energy: %.1f" % (freq_dict_gdd.get(pattern)/float(np.sum(list(freq_dict_gdd.values()))), energy))
        n_oel.append("Energy: %.1f" % (energy))
        energies.append(energy)
        
    code_probs_gdd_cp = np.array(list(freq_dict_gdd.values()))/np.sum(list(freq_dict_gdd.values()))
    indexed_cp = sorted(range(len(code_probs_gdd)), key=lambda k: code_probs_gdd_cp[k], reverse=True)
    #oel_cp = [oel[i] for i in indexed_cp]
#    n_oel_cp = [n_oel[i] for i in indexed_cp]
    energies_cp = [energies[i] for i in indexed_cp]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.plot(code_probs_gdd, 'o', label="GGD")
    ax.set_yscale('log')
    ax.set_xlabel("Codeword")
    ax.set_ylabel("Probability")
#    ax.set_xticklabels(['']+n_oel_cp, rotation='vertical')
    #ax.set_xticklabels(['']+oel_cp, rotation='vertical')
#    ax.set_xticks([i for i in np.arange(-1, len(stored_energies), 1.)])
    plt.show()
    
    energies_cp = [energies[i] for i in indexed_cp]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.plot(code_probs_gdd, energies_cp, 'x')
    ax.set_xscale('log')
    ax.set_xlabel("Prob")
    ax.set_ylabel("Energy")
    plt.show()
    
    
    ###order and plot found local energy minima
    ordered_indices = []
    for j in range(N):
        for i in range(len(freq_dict_gdd.keys())): 
            if list(freq_dict_gdd.keys())[i][j-3] == -1. and list(freq_dict_gdd.keys())[i][j-2] == -1. and list(freq_dict_gdd.keys())[i][j-1] == -1. and list(freq_dict_gdd.keys())[i][j] == 1. and i not in ordered_indices:
                ordered_indices.append(i)
                
    
    for i in range(len(freq_dict_gdd.keys())): 
        if i not in ordered_indices:
            ordered_indices.append(i)
                
    ordered_patterns = [list(freq_dict_gdd.keys())[i] for i in ordered_indices]
#    ordered_energies = [oel[i] for i in ordered_indices]
    fig = plt.figure(figsize=(7.5,7.5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(ordered_patterns, aspect=5)
#    ax.set_yticklabels(['']+ordered_energies)
#    ax.set_yticks([i for i in np.arange(-1, len(stored_energies), 1.)])
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Pattern")
    plt.show()         
    return np.array(ordered_patterns)


def order_patterns(patterns_gdd):
    N = patterns_gdd.shape[1]
    "ordered_patterns.shape = (number of patterns, number of neurons)"
    tuple_codewords = map(tuple, patterns_gdd)
    freq_dict_gdd = Counter(tuple_codewords)    
    ###order and plot found local energy minima
    ordered_indices = []
    for j in range(N):
        for i in range(len(freq_dict_gdd.keys())): 
            if list(freq_dict_gdd.keys())[i][j-3] == -1. and list(freq_dict_gdd.keys())[i][j-2] == -1. and list(freq_dict_gdd.keys())[i][j-1] == -1. and list(freq_dict_gdd.keys())[i][j] == 1. and i not in ordered_indices:
                ordered_indices.append(i)
    for i in range(len(freq_dict_gdd.keys())): 
        if i not in ordered_indices:
            ordered_indices.append(i)
               
    ordered_patterns = [list(freq_dict_gdd.keys())[i] for i in ordered_indices]
    
    return np.array(ordered_patterns)

def calc_pattern_energies(patterns, h, J):
    energies = []
    for pattern in patterns:
        energy = calc_energy([h,J], pattern)
        energies.append(energy)
    return energies


def get_indices_where_different(pattern1, pattern2):
    indxs = np.where(np.array(pattern1) != np.array(pattern2))[0]
    return indxs


def make_np_shortest_paths(pattern1, pattern2, Np):
    Np = min(Np, 2**get_indices_where_different(pattern1, pattern2).shape[0])
    indxs = get_indices_where_different(pattern1, pattern2)
    all_paths = []
    try:
        for ip in range(Np):
            np.random.shuffle(indxs)
            pattern_path = make_patterns_from_indices(indxs, pattern1, pattern2)
            all_paths.append(pattern_path)
    except:
        all_paths = [pattern1]
        print("Patterns are identical")
    return np.array(all_paths)

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

def calculate_energies_for_paths(paths, h, J):
    energies_per_path = []
    for path_between in paths:
        energies_on_this_path = []
        for pattern in path_between:
            energies_on_this_path.append(calc_energy([h,J], pattern))
        energies_per_path.append(energies_on_this_path)
    return energies_per_path

def plot_single_pattern(pattern):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(np.reshape(pattern, (1,-1)))
    plt.show()
    
    
def bethe(m, C, J_init, eta, max_steps, damp_fact):
    """SUSPROP Huang 2012
    m:  measured magnetization m_i = <sigma_i>_data
    C: connected correlation <sigma_i sigma_j>_data - m_im_j
    eta: precision
    max_steps: 
    damp_fact: damping factor
    """
    N = m.shape[0]
    J = J_init
    
    #the message m_{i→j} is randomly initialized in the interval [−1.0,1.0]
    mij_mat = np.random.uniform(-1.,1., [N,N])
    
    #g i→j,k = 0, if i = k, and 1.0 otherwise
    gijk = zeros([N,N,N])
    for i in range(N):
        gijk[i,:,i] = ones(N)
    
    for t in range(max_steps):
        for i in range(N):
            for j in range(N):
                #cavity magnetization of variable i in the absence of variable j denoted
                mij = (m[i] - mij_mat[j,i]*np.tanh(J[i,j]))/(1-m[i]*mij_mat[j,i]*np.tanh(J[i,j]))
                mij_mat[i,j] = mij       
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    #cavity susceptibility
                    gijk[i,j,k] = np.dot(np.multiply(1-np.square(mij_mat[i,:]), 1-np.square(np.multiply(mij_mat[i,:],np.tanh(J[:,i])))), np.tanh(np.multiply(J[:,i], gijk[:,i,k])))
                    if i==k:
                        gijk[i,j,k] += 1.
        
        C_tilde = zeros([N,N])
        for i in range(N):
            for j in range(N):
                C_tilde[i,j] = C[i,j] - (1-np.square(m[i])) * gijk[i,j,j]/(gijk[j,i,j]) + m[i]*m[j]
                
#        C_tilde = C - np.multiply((1-np.square(m)), np.diagonal(gijk, axis1=0, axis2=2))/np.diagonal(gijk, axis1=0, axis2=2) #transpose??
        
        #J_new 
        for i in range(N):
            for j in range(N): 
                J[i,j] = damp_fact/2.*np.log(((1+C_tilde[i,j])*(1-mij_mat[i,j]*mij_mat[j,i]))/((1+C_tilde[i,j])*(1+mij_mat[i,j]*mij_mat[j,i])))  + (1-damp_fact)*J[i,j]
        
        np.fill_diagonal(J, 0)
        
    h = zeros(N)
    for i in range(N):
        h[i] = np.arctanh(m[i]) - np.dot(J[:,i], mij_mat[:,i])
    return h, J
    

def belief_propagation(m, C, h, J, max_steps, eta):
    """BP Huang 2012
    m:  measured magnetization m_i = <sigma_i>_data
    C: connected correlation <sigma_i sigma_j>_data - m_im_j
    """
    N = m.shape[0]
#    J = J_init
    
    #the message m_{i→j} is randomly initialized in the interval [−1.0,1.0]
    mij_mat = np.random.uniform(-1.,1., [N,N])
    mij_tilde = zeros([N,N])
    
    #g i→j,k = 0, if i = k, and 1.0 otherwise
    gijk = zeros([N,N,N])
    for i in range(N):
        gijk[i,:,i] = ones(N)
    
    for t in range(max_steps):
        
        for i in range(N):
            for j in range(N):
                mnj = mij_tilde[:,i][~(np.arange(len(mij_tilde)) == j)]
                mij = (exp(h[i]) * np.prod(1+mnj) - exp(-h[i])* np.prod(1-mnj))/(exp(h[i]) * np.prod(1+mnj) + exp(-h[i])* np.prod(1-mnj))
                mij_mat[i,j] = mij     
                
        
        for i in range(N):
            for j in range(N):
                mij_tilde[j,i] = np.tanh(J[j,i]*mij_mat[j,i])
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    sum_gijk = 0
                    for l in range(N):
                        if l!=j:
                            sum_gijk += (1-mij_mat[l,i]**2)/(1-(mij_mat[l,i]*np.tanh(J[l,i]))**2)*np.tanh(J[l,i]*gijk[l,i,k])
                    
                    gijk[i,j,k] = sum_gijk
                    if i==k:
                        gijk[i,j,k] += 1.
        
    C_tilde = zeros([N,N])
    for i in range(N):
        for j in range(N):
            C_tilde[i,j] = C[i,j] - (1-np.square(m[i])) * gijk[i,j,j]/(gijk[j,i,j]) + m[i]*m[j]
                
        
    return C_tilde, mij_mat, mij_tilde


def entropy_bethe(h, J, max_steps):
    s = 0
    N = h.shape[0]
    mij_mat, mij_tilde = calc_mij_mats(h, J, max_steps)
    for i in range(N):
        s += site_contribution(i, h, J, mij_mat, mij_tilde)
        for j in range(N):
            s -= edge_contribution(i, j, h, J, mij_mat, mij_tilde)
    return s
    
def calc_mij_mats(h, J, max_steps):
    N = h.shape[0]
    mij_mat = np.random.uniform(-1.,1., [N,N])
    mij_tilde = zeros([N,N])
    N = h.shape[0]
    for t in range(max_steps):
        
        for i in range(N):
            for j in range(N):
                #mij[i,i]??
                if i == j:
                    mij_mat[i,j] = 0
                else:
                    mnj = mij_tilde[:,i][~(np.arange(len(mij_tilde)) == j)]
                    mij = (exp(h[i]) * np.prod(1+mnj) - exp(-h[i])* np.prod(1-mnj))/(exp(h[i]) * np.prod(1+mnj) + exp(-h[i])* np.prod(1-mnj))
                    mij_mat[i,j] = mij 
#                mij_mat[i,j] = (exp(h[i])*np.prod(1+mij_tilde[:,i][~(np.arange(N) == j)]) - exp(h[i])*np.prod(1-mij_tilde[:,i][~(np.arange(N) == j)]))/(exp(h[i])*np.prod(1+mij_tilde[:,i][~(np.arange(N) == j)]) + exp(h[i])*np.prod(1-mij_tilde[:,i][~(np.arange(N) == j)]))
        np.fill_diagonal(mij_mat, 0)        
        
        for i in range(N):
            for j in range(N):
                mij_tilde[j,i] = np.tanh(J[j,i]*mij_mat[j,i])
        np.fill_diagonal(mij_tilde, 0)
    return mij_mat, mij_tilde
    

def site_contribution(i, h, J, mij_mat, mij_tilde):
    zi = z_i(i, h, J, mij_mat, mij_tilde)
    s_contr = np.log(zi)
    s_contr -= 1/zi * ( X_i(i, 1., h, J, mij_mat, mij_tilde) -  X_i(i, -1., h, J, mij_mat, mij_tilde))
    s_contr -= 1/zi * (Y_i(i, 1., h, J, mij_mat, mij_tilde) + Y_i(i, -1., h, J, mij_mat, mij_tilde))
    return s_contr
    
def X_i(i, y, h, J, mij_mat, mij_tilde):
    return h[i]*exp(y*h[i])*np.prod(np.cosh(np.multiply(J[:,i],1+y*np.tanh(np.multiply(J[:,i], mij_mat[:,i])))))

def Y_i(i, y, h, J, mij_mat, mij_tilde):
    N = h.shape[0]
    yi = 0
    for l in range(N):
        fy = J[l,i]*sinh(J[l,i]*(1+y*tanh(J[l,i]*mij_mat[l,i])))
        fy += y*J[l,i]*cosh(J[l,i]*(1-tanh(J[l,i])**2)*mij_mat[l,i])
        fy *= np.prod(np.cosh(np.multiply(J[i,:][~(np.arange(len(mij_mat)) == l)],1+y*np.tanh(np.multiply(J[i,:][~(np.arange(len(mij_mat)) == l)], mij_mat[:,i][~(np.arange(len(mij_mat)) == l)])))))
        yi += fy
    yi *= exp(-y*h[i])
    return yi

def z_i(i, h, J, mij_mat, mij_tilde):
    return exp(h[i])*np.prod(np.cosh(np.multiply(J[:,i], 1+mij_tilde[:,i]))) + exp(-h[i])*np.prod(np.cosh(np.multiply(J[:,i], 1-mij_tilde[:,i])))


def edge_contribution(i, j, h, J, mij_mat, mij_tilde):
    zij = np.log(cosh(J[i,j]*(1+tanh(J[i,j]*mij_mat[i,j]*mij_mat[j,i]))))
    zij -= J[i,j]*(tanh(J[i,j]+mij_mat[i,j]*mij_mat[j,i]))/(1+tanh(J[i,j]*mij_mat[i,j]*mij_mat[j,i]))
    return zij

###Huang 2016 Clustering...

def message_passing2(h, J, max_steps):
    N = h.shape[0]
#    if x != 0.:
#        #with x \sigma^* \sigma
#        h_var = beta*h + x*ref_sigma
#    else:
#        #without x \sigma^* \sigma
#        h_var = h
#    if beta != 1.:
#        J_var = beta*J
#    else:
#        J_var = J
#    #the message m_{i→j} is randomly initialized in the interval [−1.0,1.0]
    mia = np.random.uniform(-1.,1., [N,N])
#    mia = zeros([N,N])
#    mbi_tilde = zeros([N,N])
    mbi_tilde = np.random.uniform(-1.,1., [N,N])
    
    for step in range(max_steps):
        for i in range(N):
            for j in range(N):
                mia[i,j] = np.tanh(h[i] + np.sum(np.arctanh(mbi_tilde[i,:][~(np.arange(N) == j)]))) 
        for i in range(N):
            for j in range(N):
                mbi_tilde[j,i] = np.tanh(J[j,i] * mia[j,i])
    
    return mia, mbi_tilde

def message_passing(h, J, max_steps):
    N = h.shape[0]
    mia = np.random.random((N, N))
    mia = np.triu(mia,k=1)
    
    for step in range(max_steps):
        for i in range(N):
            for j in range(i,N):
                sum_b = 0.
                for k in range(N):
                    if k!=j and k!=i:
                        a=k
                        b=j
                        if k>j:
                            a=j
                            b=k
                        sum_b += J[a,b] * mia[a,b]
                mia[i,j] = np.tanh(h[i] + sum_b)
    
    return mia

def message_passing_with(mia, h, J, max_steps):
    N = h.shape[0] 
    for step in range(max_steps):
        for i in range(N):
            for j in range(i,N):
                sum_b = 0.
                for k in range(N):
                    if k!=j and k!=i:
                        a=k
                        b=j
                        if k>j:
                            a=j
                            b=k
                        sum_b += tanh(J[a,b]) * mia[a,b]
                mia[i,j] = np.tanh(h[i] + np.arctanh(sum_b))
    
    return mia

def message_passing_with2(mia, mbi_tilde, h, J, max_steps):
    N = J.shape[0]
    for step in range(max_steps):
        for i in range(N):
            for j in range(N):
                mia[i,j] = np.tanh(h[i] + np.sum(np.arctanh(mbi_tilde[i,:][~(np.arange(N) == j)]))) 
        for i in range(N):
            for j in range(N):
                mbi_tilde[j,i] = np.tanh(J[j,i]) * mia[j,i]
    
    return mia, mbi_tilde


def free_energy(h, J, mia):
    N = h.shape[0]
    F = 0
    for i in range(N):
        #F_i
        F += free_energy_contribution_one_neuron(i, h, J, mia)
        for j in range(i,N):
            #F_a
            F -= free_energy_contribution_one_interaction(i, j, h, J, mia)
    return F

#def free_energy_density(h, J, mia, mbi_tilde):
#    N = h.shape[0]
#    f = 0
#    for i in range(N):
#        f += free_energy_contribution_one_neuron(i, h, J, mbi_tilde)/float(N)
#        for j in range(N):
#            f -= float(N-1)/N*free_energy_contribution_one_interaction(i, j, h, J, mia)
#    return f

def free_energy_contribution_one_neuron(i, h, J, mia):
    Hi = H(i,1,h,J,mia)#np.exp(h[i])*np.prod(np.cosh(np.multiply(J[:,i],(1+mbi_tilde[:,i]))))
    Hi += H(i,-1,h,J,mia)#np.exp(-h[i])*np.prod(np.cosh(np.multiply(J[:,i],(1-mbi_tilde[:,i]))))
    return -np.log(Hi)#-ln Z_i = 

def free_energy_contribution_one_interaction(i, j, h, J,  mia):
    return -np.log(np.cosh(J[i,j])) - np.log(1+np.tanh(J[i,j]*mia[i,j]*mia[j,i]))
    #return -np.log(np.cosh(J[i,j])) - np.log(1+np.tanh(J[i,j]*np.prod(mia[:,j]))) #-ln Z_a



def energy_of_neural_population(h, J, mia):
    E = 0
    N = h.shape[0]
    for i in range(N):
        E -= energy_contribution_one_neuron(i, h, J, mia)
        for j in range(i,N):
            #E_a
            E += energy_contribution_one_interaction(i, j, h, J, mia)
    return E

def H(i,y,h,J,mia):
    product = np.exp(y*h[i])
    for j in range(N):
        a=j
        b=i
        if j<i:
            a=i
            b=j
        product *= np.cosh(J[a,b]*(1+y*tanh(J[a,b]*mia[a,b])))
    return product
#    return np.exp(y*h[i])*np.prod(np.cosh(np.multiply(J[:,i],(1+y*mbi_tilde[:,i]))))

#def G(i,y,h,J,mia,mbi_tilde):
#    Gi = 0
#    N= h.shape[0]
#    for j in range(N):
#        expyhi = np.exp(y*h[i])
#        gamsin = J[i,j]*np.sinh(J[i,j]*(1+y*mbi_tilde[j,i]))
#        ygam = y*J[i,j]*np.cosh(J[i,j]*(1-np.tanh(J[i,j])**2) * mia[j,i])
#        endprod = np.prod(np.cosh(np.multiply(J[i,:][~(np.arange(N) == j)], (1+y*mbi_tilde[i,:][~(np.arange(N) == j)]))))
#        Gi += expyhi * (gamsin + ygam) * endprod 
##        print(expyhi, gamsin, ygam, endprod, Gi)
##    print(Gi)
#    return Gi

def G(i,y,h,J,mia):
    Gi = 0
    N= h.shape[0]
    for j in range(N):
        a=j
        b=i
        if j>i:
            a=i
            b=j
        expyhi = np.exp(y*h[i])
        gamsin = J[i,j]*np.sinh(J[i,j]*(1+y*tanh(J[i,j]*mia[a,b])))
        ygam = y*J[i,j]*np.cosh(J[i,j]*(1-np.tanh(J[i,j])**2) * mia[j,i])
        endprod = 1.
        for k in range(N):
            c=k
            d=i
            if k!=j:
                c=i
                d=k
                endprod *= np.cosh(J[c,d]*(1+y*tanh(J[c,d]*mia[c,d])))
        Gi += expyhi * (gamsin + ygam) * endprod 
#        print(expyhi, gamsin, ygam, endprod, Gi)
#    print(Gi)
    return Gi


def energy_contribution_one_neuron(i, h, J, mia,):
    return  (h[i]*(H(i,1.,h,J,mia) - H(i,-1.,h,J,mia)) + G(i,1.,h,J,mia) + G(i,-1.,h,J,mia))/(H(i,1.,h,J,mia) + H(i,-1.,h,J,mia))
     
    
def energy_contribution_one_interaction(i, j, h, J, mia):
    ###np.prod(mia[i,:]) should be just i and j? mia[i,j]*mia[i,j]
    #delta_E = J[i,j] * (np.tanh(J[i,j] + np.prod(mia[i,:])))/(1+np.tanh(J[i,j]*np.prod(mia[i,:])))
    delta_E = J[i,j] * (np.tanh(J[i,j]) + mia[i,j]*mia[j,i])/(1+np.tanh(J[i,j]*mia[i,j]*mia[j,i]))
    return delta_E


def S(h, J, x, beta, ref_sigma, max_steps):
    h_var = beta*h + x*ref_sigma
    J_var = beta*J
    #returns S = -F + E ##Huang 2016
    mia = message_passing(h_var, J_var, max_steps)
    F = free_energy(h_var, J_var, mia)
    E = energy_of_neural_population(h_var, J_var, mia)
    return -F + E


def q_tilde(h_var, J_var, ref_sigma, mia, mbi_tilde):
    N = h_var.shape[0]
    q_til = zeros(N)
    for i in range(N):
        q_til[i] = -(ref_sigma[i]*H(i,1,h_var,J_var,mbi_tilde) - ref_sigma[i]*H(i,-1,h_var,J_var,mbi_tilde))/(H(i,1,h_var,J_var,mbi_tilde) + H(i,-1,h_var,J_var,mbi_tilde))
    
    return np.sum(q_til)



def s_q(h, J, xs, beta, ref_sigma, max_steps):
    """Eq 4 in Huand 2013
    s( q̃) = min [ f (x) − xq̃],"""
    ent_list = []
    for x in xs:
        h_var = beta*h + x*ref_sigma
        J_var = beta*J
        mia, mbi_tilde = message_passing(h_var, J_var, max_steps)
        f = free_energy_density(h_var, J_var, mia, mbi_tilde)
        q_til = q_tilde(h_var, J_var, ref_sigma, mia, mbi_tilde)
        ent_list.append(f - x*q_til)
    
    return ent_list
#    return min(ent_list) #, xs[np.where(ent_list == min(ent_list))]

def model_spiking_rate(h, J, mia):
    N = h.shape[0]
    mi = zeros(N)
    for i in range(N):
        for j in range(N):
            sum_b = 0.
            if j!=i:
                a=i
                b=j
                if j>i:
                    a=j
                    b=i
                sum_b += J[a,b]*mia[a,b]
        mi[i] = np.tanh(h[i] + sum_b) 
    return mi
#    return np.tanh(h + np.sum(np.arctanh(mbi_tilde), axis=1))


def multi_neuron_correlation(J, mia):
    #corrs = (np.tanh(J) + np.prod(mia, axis=1))/(1+np.tanh(np.prod(np.multiply(J,mia), axis=1)))
    N = J.shape[0]
    corrs = zeros([N,N])
    for i in range(N):
        for j in range(i,N): 
            corrs[i,j] = (np.tanh(J[i,j]) + mia[i,j]*mia[j,i])/(1+np.tanh(J[i,j]*mia[i,j]*mia[j,i]))
    return corrs


def learning_eqs(mag_sim, corrs_sim, h, J, learning_rate, max_steps, max_steps_mp):
    min_av_max = []
    h_mp=np.array(h,copy=True)
    J_mp=np.array(J,copy=True)
    for step in range(max_steps):
        mia = message_passing(h_mp, J_mp, max_steps_mp)
        h_mp += learning_rate * (mag_sim - model_spiking_rate(h_mp, J_mp, mia))
        J_mp += learning_rate * (corrs_sim - multi_neuron_correlation(J_mp, mia))
        min_av_max.append(np.array([np.min(h_mp), np.average(h_mp), 
                                     np.max(h_mp),
                                     np.min(J_mp), np.average(J_mp), 
                                     np.max(J_mp)]))
    return h_mp, J_mp, np.array(min_av_max)
 
    

def calc_q(h, ref_sigma, mbi_tilde):
    N=h.shape[0]
    mi = model_spiking_rate(h, mbi_tilde)
    return np.dot(ref_sigma, mi)/N

def q_to_d(q):
    return (1-q)/2

def d_to_q(d):
    return 1-2*d

def secant(d, x0, x1, epsilon, max_steps_k, h, J, xs, beta, ref_sigma, max_steps_sq, max_steps_mp):
    q = d_to_q(d)
    J_var = beta*J
    #if q_tilde = q
    #calc s(d)
    s_q_tilde = min(s_q(h, J, xs, beta, ref_sigma, max_steps_sq))
    xk_1 = x0
    xk = x1
    h_var = beta*h + xk_1*ref_sigma
    mia, mbi_tilde = message_passing(h_var, J_var, max_steps_mp)
    fx_prev = free_energy_density(h_var, J_var, mia, mbi_tilde)
    F_prev = fx_prev + xk_1*q + s_q_tilde
    for k in range(max_steps_k):
        h_var = beta*h + xk*ref_sigma
        mia, mbi_tilde = message_passing(h_var, J_var, max_steps_mp)
        fx_curr = free_energy_density(h_var, J_var, mia, mbi_tilde)
        F_curr = fx_curr + xk*q + s_q_tilde
        
        B_k = (F_curr - F_prev)/(xk - xk_1)
        pk = -F_curr/B_k
        xk_1 = xk
        xk = xk + pk
        F_prev = F_curr
        print(xk)
        if abs(xk - xk_1) < epsilon:
            break
    
    return xk


def secant2(d, x0, x1, epsilon, max_steps_k, h, J, xs, beta, ref_sigma, max_steps_sq, max_steps_mp):
    q_tilde = d_to_q(d)
    J_var = beta*J
    N=h.shape[0]
    #calc s(d)
    
#    ent_list = []
#    for x in xs:
#        h_var = beta*h + x*ref_sigma
#        J_var = beta*J
#        mia, mbi_tilde = message_passing(h_var, J_var, max_steps_sq)
#        f = free_energy_density(h_var, J_var, mia, mbi_tilde)
#        ent_list.append(f - x*q_tilde)
#    s_q_tilde = min(ent_list)
#    print(s_q_tilde)
    
    #secant method
    xk_1 = x0
    xk = x1
    h_var = beta*h + xk_1*ref_sigma
    mia, mbi_tilde = message_passing(h_var, J_var, max_steps_mp)
#    fx_prev = free_energy(h_var, J_var, mia, mbi_tilde)/float(N)
    s_prev = S(h, J, xk_1, beta, ref_sigma, max_steps_mp)/float(N)
#    q_prev = calc_q(h_var, ref_sigma, mbi_tilde)
#    F_prev = -fx_prev - xk_1*q_tilde #+ s_q_tilde
    F_prev = s_prev
    
    for k in range(max_steps_k):
        h_var = beta*h + xk*ref_sigma
        mia, mbi_tilde = message_passing(h_var, J_var, max_steps_mp)
#        fx_curr = free_energy(h_var, J_var, mia, mbi_tilde)/float(N)
        s_curr = S(h, J, xk, beta, ref_sigma, max_steps_mp)/float(N)
#        q_curr = calc_q(h_var, ref_sigma, mbi_tilde)
#        F_curr = -fx_curr - xk*q_tilde #+ s_q_tilde
        F_curr = s_curr - xk*q_tilde
#        print("fx:", fx_curr - fx_prev)
        
        B_k = (F_curr - F_prev)/(xk - xk_1)
        pk = -F_curr/B_k
        xk_1 = xk
        xk = xk + pk
        
#        print("F:", F_prev - F_curr, "X:", xk - xk_1, "B:", B_k)
        if abs(xk - xk_1) < epsilon or abs(F_curr - F_prev) < epsilon:
            h_var = beta*h + xk*ref_sigma
            mia, mbi_tilde = message_passing(h_var, J_var, max_steps_mp)
#            fx_curr = free_energy_density(h_var, J_var, mia, mbi_tilde)
            break
        
        F_prev = F_curr
#        s = -fx_curr + energy_of_neural_population_density(h_var, J_var, mia, mbi_tilde)
    s = S(h, J, xk, beta, ref_sigma, max_steps_mp)/float(N) #- xk*q_tilde
    return xk, s


def secant3(d, x0, x1, epsilon, max_steps_k, h, J, beta, ref_sigma,
            max_steps_sq, max_steps_mp):
    q_til = d_to_q(d)
    J_var = beta*J
    N=h.shape[0]
    #secant method
    xk_1 = x0
    xk = x1
    h_var = beta*h + xk_1*ref_sigma
    mia = message_passing(h_var, J_var, max_steps_mp)
    mag = model_spiking_rate(h_var, J_var, mia)
    F_prev = np.dot(ref_sigma, mag)/N - q_til
    
    for k in range(max_steps_k):
        h_var = beta*h + xk*ref_sigma
        mia = message_passing(h_var, J_var, max_steps_mp)
        mag = model_spiking_rate(h_var, J_var, mia)
        F_curr = np.dot(ref_sigma, mag)/N - q_til
        
        B_k = (F_curr - F_prev)/(xk - xk_1)
        pk = -F_curr/B_k
        xk_1 = xk
        xk = xk + pk
        
        print("F(x_k):", F_curr, "X_k:", xk)
        if abs(F_curr - 0.) < epsilon:
            h_var = beta*h + xk*ref_sigma
            mia = message_passing(h_var, J_var, max_steps_mp)
            break
        
        F_prev = F_curr
    if k==max_steps_k-1:
        print("Did not converge in %d steps" %max_steps_k)
    f = free_energy(h_var, J_var, mia)/N
    f_h = free_energy(h, J, mia)/N
    e = energy_of_neural_population(h, J, mia)/N
    s = -f + e - xk*q_til
    print(d, q_til, xk, -f, -f_h, e, -f + e, -f + e -xk*q_til, -f_h + e -xk*q_til)
#    s = S(h, J, xk, beta, ref_sigma, max_steps_mp)/float(N) #- xk*q_til
#    s = entropy_bethe(h_var, J_var, 1)/N #- xk*q_til
    return xk, s



##exact
def calc_s(h, ref_sigma, x):
    #h = np.log((1+m)/(1-m))/2.?
    N = h.shape[0]
    return np.sum(np.log(2*np.cosh(h + x*ref_sigma)))/N - np.dot((h + x*ref_sigma), np.tanh(h + x*ref_sigma))/N

def calc_q_h(h, ref_sigma, x):
    N = h.shape[0]
    return np.dot(ref_sigma, np.tanh(h + x*ref_sigma))/N



def secant_h(d, x0, x1, epsilon, max_steps_k, h, beta, ref_sigma):
    q_til = d_to_q(d)
    N=h.shape[0]
    #secant method
    xk_1 = x0
    xk = x1
    F_prev = np.dot(ref_sigma, h+ xk_1*ref_sigma)/N - q_til
    
    for k in range(max_steps_k):
        F_curr = np.dot(ref_sigma, h+ xk*ref_sigma)/N - q_til
        
        B_k = (F_curr - F_prev)/(xk - xk_1)
        pk = -F_curr/B_k
        xk_1 = xk
        xk = xk + pk
        
#        print("F(x_k):", F_curr, "X_k:", xk)
        if abs(F_curr - F_prev) < epsilon:
            break
        
        F_prev = F_curr
    if k==max_steps_k-1:
        print("Did not converge in %d steps" %max_steps_k)
    s = calc_s(h, ref_sigma, xk)
    return xk, s

###Hopfield 

def make_hopfield_weights(pattern_list):
    N = pattern_list[0].shape[0]
    weights = zeros([N, N])
    for pattern in pattern_list:
        weights += np.outer(pattern, pattern)   
        
    np.fill_diagonal(weights, 0)
    return weights/float(N)



