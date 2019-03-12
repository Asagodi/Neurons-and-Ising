"""
Created on Wed Feb 13 12:24:17 2019
@author: abel
"""

from mpl_toolkits.mplot3d import Axes3D
from scipy import *
from scipy import optimize
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import numpy as np
from progress.bar import Bar


def make_connection_matrix(N, inh, R, ell):
    """Constructs a connection matrix for a ring attractor network as in
    Couey et al. """
    theta = zeros([N])
    theta[0:N:2] = 0
    theta[1:N:2] = 1
    theta = 0.5*pi*theta
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
    return W
    

def sim_one_d(N, extinp, inh, R, umax, dtinv, tau, time):
    """Simulates neural behaviour through a Poisson process 
    of a network of size N 
    static bump"""
    xes = zeros(N)
    for x in range(N):
      xes[x] = x
    
    Wgen = zeros([N,N], dtype=bool)
    for x in range(N):
      xdiff = abs(xes-x)
      xdiff = minimum(xdiff, N-xdiff)
      Wgen[xdiff<R,x] = 1
    W = zeros([N,N])
    W[Wgen>0.7] = -inh
    W = sparse.csc_matrix(W)
    
    S = zeros(N)
    ## generate random activity (doesn't matter much)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
    
    times = range(time)
    activities = zeros([N,time])
    Stemp = zeros(N)
    for t in times:
      S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*W))
      S[S<0.00001] = 0.
      activities[:,t] = S
    S = ravel(S)
    return activities

def sim_dyn_one_d(N, extinp, inh, R, umax, dtinv,
              tau, time, ell):
    """Simulates neural behaviour through a Poisson process 
    of a network of size N 
    moving bump"""
    W = make_connection_matrix(N, inh, R, ell)
    W = sparse.csc_matrix(W)
    
    S = zeros(N)
    ## generate random activity (doesn't matter much)
    for i in range(N):
      if(rand()<0.5):
        S[i] = rand()
        
    times = range(time)
    activities = zeros([N,time])
    Stemp = zeros(N)
    for t in times:
      S = S + 1./(dtinv+tau) * (-S + maximum(0., extinp+S*W))
      S[S<0.00001] = 0.
      activities[:,t] = S
    S = ravel(S)
    return activities

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
def detect_spikes(act):
    """Get spikes from Poisson process with threshold 11.8, adapted from Couey"""
    r = 11.8*np.random.rand(act.shape[0], act.shape[1])
    spiked_act = np.greater_equal(act, r)
    return spiked_act.astype(int)

def mean_spike_rate(act):
    # calculates mean spike rate
    return np.sum(act, axis=1)/(2*act.shape[1])+1./2

def spiking_probs(act, delta_t):
    # calculates spking probabilities
    msr = np.average(act, axis=1)
    return -1 + 2*msr*delta_t

#determine state
def determine_states(spike_act):
    states = zeros(spike_act.shape)
    states[spike_act>0] = 1 
    states[spike_act==0] = -1 
    return states


def calc_firing_probs(binned_act, time, bsize):
    # calculates firing probabilities
    nbin = int(time/bsize)
    probs = np.sum(binned_act, axis=1)/nbin
    return probs

def find_codewords(binned_act):
    # determines the existing patterns in binned neural data
    codewords = []
    for i in range(binned_act.shape[1]):
        col = binned_act[:,i]
        #look which neurons are active
        I = np.where(col[:] == 1) ###different value?
        codewords.append(I[0].tolist())
    
    return codewords

def find_frequencies(codewords):
    #determines the frequencies of the patterns in binned neural data
    tuple_codewords = map(tuple, codewords)
    freq_dict = Counter(tuple_codewords)
    freqs = np.array(sorted(list(freq_dict.values()),
                            reverse=True))
    
    return freqs


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

        
def calc_energy(e_funcs, coeff_list, acts):
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
        sum_E += calc_energy(e_funcs, coeff_list, acts[:,i])
        ave_E = sum_E/nbin
    return ave_E

###################Probs + MC
def calc_state_probs(all_states, h, J, beta):
    """Calculates the probabilities of all the possible patterns 
    for exact learning (suitable up to N=20)"""
    p_tot = 0
    p_vec = zeros(all_states.shape[0])
    for n, state in enumerate(all_states):
        e = calc_energy([h1, h2], [h,J], state)
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

def calc_third_order_corr(s_act, mag):
    #calculate third order correlations from neural activity
    Nneur = s_act.shape[0]
    nbin = s_act.shape[1]
    c = zeros([Nneur,Nneur,Nneur])
    
    for t in range(nbin):
        c += np.reshape(np.outer(np.reshape(np.outer(s_act[:,t]-mag,s_act[:,t]-mag), (Nneur, Nneur)), s_act[:,t]-mag), (Nneur, Nneur, Nneur))           
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

def mc_step(N, h, J, current_state, Nflips, e_old, T):
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
        new_state = zeros(N)
        new_state[:] = current_state[:]
        new_state[ijk] =   - current_state[ijk]
        
        #calculate energy of new state
        e_new = calc_energy([h1,h2], [h, J], new_state)
        
        #energy difference
        e_delta = e_new - e_old
        
        #step 4
        r = np.random.rand()
        
#            print(e_delta)
        #step 5
        if r < np.exp(- e_delta/T):
            current_state = new_state
            e_old = e_new
    return current_state

def mc_step_2(N, h, J, current_state, e_old, T):
    """Same as mc_step but with Nflips=1"""
    #step 3
    #calculate energy of old state
#    e_old = -np.sum(.5*np.multiply(J, np.outer(current_state, current_state)))- np.dot(h, current_state)
    ijk = np.random.randint(0,N)
    new_state = zeros(N)
    new_state[:] = current_state[:]
    new_state[ijk] =   - current_state[ijk]
    
    #calculate energy of new state
    e_new = -np.sum(.5*np.multiply(J, np.outer(new_state, new_state)))- np.dot(h, new_state)
    e_delta = e_new - e_old
    #step 4
    r = np.random.rand()
    #step 5
    if r < np.exp(- e_delta/T):
        current_state = new_state
        e_old = e_new
    return current_state, e_old

def metropolis_mc(h, J, Nsamples, Nflips,
                  sample_after_step, sample_per_steps, T):
    """Metropolis Monte Carlo simulation with spin flip
    Nsteps: maximal number of steps
    Nflips: is number of flips before choosing another starting point
    sample_after_step: sample after number of steps
    sample_per_steps sample every sample_per_steps steps"""
    N = h.shape[0]
    initial_state = np.random.rand(N)
    thr = 0.6
    initial_state[initial_state>thr] = 1 
    initial_state[initial_state<=thr] = -1
    mc_samples = zeros([N, Nsamples])
    Nsteps = int(Nsamples * sample_per_steps)
    current_state = initial_state
    e_old = -np.sum(.5*np.multiply(J, np.outer(current_state, current_state)))- np.dot(h, current_state)
    for step in range(sample_after_step):
        #mc_step_2 faster if Nflips=1
        current_state, e_old = mc_step_2(N, h, J, current_state, e_old, T)
    
#    bar = Bar('MC simulation', max=Nsamples)
#    bar.next()
    for step in xrange(0, Nsteps, 1):
        current_state, e_old = mc_step_2(N, h, J, current_state, e_old, T)
        
        if step % sample_per_steps == 0:
#            bar.next()
            mc_samples[:, int(step / sample_per_steps)] = current_state
            
#    bar.finish()    
    return mc_samples



##############LEM + MDS
#greedy descent dynamics
def gdd(coeffs, initial_state):
    """for each neuron, we flip its activity if the flip will decrease the
    energy. If we could not decrease the energy by flipping any
    neuron’s activity, then a local energy minimum is identified"""
    Nneur = initial_state.shape[0]
    current_state = zeros(Nneur)
    current_state[:] = initial_state[:]
    while True:
        e_old = calc_energy([h1, h2], coeffs, current_state)
        
#       attempt to flip spins i~1,N from their current state into {s i , in order of increasing i.
        indices = range(Nneur)
        
        #random order of spin flip
#        indices = np.random.permutation(Nneur)
        stop_ind = 0
        for ind in indices:

            new_state = current_state
            new_state[ind] = -current_state[ind]
            e_new = calc_energy([h1, h2], coeffs, new_state)
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


def gdd_dyn(coeffs, initial_state, reference_state, beta):
    """calculate distance and entropy from reference (final) state
    when doing greedy descent dynamics"""
    beta = 1./T
    d_list = []
    s_list = []
    Nneur = initial_state.shape[0]
    current_state = zeros(Nneur)
    current_state[:] = initial_state[:]
    while True:
        
        e_old = calc_energy([h1, h2], coeffs, current_state)
        d_list.append(hamming_distance(initial_state, reference_state))
        s_list.append(calc_entropy([h1, h2], coeffs, current_state))
        
#       attempt to flip spins i~1,N from their current state into {s i , in order of increasing i.
        indices = range(Nneur)
        
        #random order of spin flip
#        indices = np.random.permutation(Nneur)
        stop_ind = 0
        for ind in indices:

            new_state = current_state
            new_state[ind] = -current_state[ind]
            e_new = calc_energy([h1, h2], coeffs, new_state)
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


##choose randomly a pattern set of size 2000
##identify a LEM set whose size is much smaller than that of the pattern set
def lem(h, J, number_of_initial_patters):
    """Determine local energy minima (for an Ising model)
    by Greedy Descent Dynamics (Huang and Toyoizumi, 2016)"""
    N = h.shape[0]
    patters = []
    for i_p in range(number_of_initial_patters):
        initial_state = np.random.choice([-1,1], N)
        patters.append(gdd([h, J], initial_state))
    return patters

def hamming_distance(sigma_1, sigma_2):
    #Calculates the Hamming distance between two neural patterns
    N = sigma_1.shape[0]
    return (N - np.sum(np.multiply(sigma_1, sigma_2)))/2

def sampled_distance_from_lem(sigma, lem):
    #Calculates the Hamming distance between a reference pattern
    # and another neural pattern
    N = sigma.shape[0]
    return (1- np.sum(np.multiply(sigma, lem)/N))/2

def distance_entropy(sigma, lem):
    #Huang, 2016, Eq 2, epsilon dependency omitted
    "log-number of states per neuron with overlap Nq (d = (1 − q)/2)"
    N = sigma.shape[0]
    
    return (1/N) * 0 # which states to sum over?

def B_matrix(data_points, c):
    #calculate B matrix for multidimensional scaling
    #see Cox-Multidimensional scaling analysis
    n_dp = len(data_points)
    H = np.identity(n_dp) - 1/n_dp*np.outer(ones(n_dp), ones(n_dp))
    A = zeros([n_dp,n_dp])
    for r, dp_r in enumerate(data_points):
        for s, dp_s in enumerate(data_points):
            delta_rs = hamming_distance(np.array(dp_r), np.array(dp_s))
            A[r,s] = -delta_rs**2/2 + c*(1-delta_rs)
#            A[r,s] = A[s,r]
    #B = HAH        
    return np.dot(H, np.dot(A, H))

def mds(data_points, p, option):
    """reduce data to p dimensions with multidimensional scaling
    option 1 ignores the negative values,
    option 2 adds an appropriate constant c to the dissimilarities 
    Cox-Multidimensional scaling analysis 2.2.1"""
    
    #As in Multidimensional scaling analysis, Cox
    B = B_matrix(data_points, c=0)
    eig_val, eig_vec = np.linalg.eig(B)
    np.less_equal(eig_val, 0)
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
    
    X = np.array(eig_vec)
    return X[:, 0:p]
    

####################PLM


def exp_sigma(act_t, rth, h, J, T):
    #calculates the conditional probability of one variable given all the others
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
    
def f_prime_seq(sigmas, h_prime, J_prime, reg_lambda, T):
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
                    f_primes_J[s,r] -= 2*p_s*e_s*sigma[r]*sigma[s]/T 
                 
                    hessian[r,Nneur+ s*Nneur+r] += 4*beta**2*e_s*sigma[s]*(1/(1+e_s) + 1)/(1+e_s)
                    hessian[Nneur+ s*Nneur+r, r] += 4*beta**2*e_s*sigma[s]*(1/(1+e_s) + 1)/(1+e_s)
                    for t in range(sigmas.shape[0]):
                        hessian[Nneur+s*Nneur+r, Nneur+t*Nneur+r] += 4*beta**2*e_s*sigma[s]*sigma[t]*(1/(1+e_s) + 1)/(1+e_s) 
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

def plm_seperated(sigmas, max_steps, h, J, h_lambda, J_lambda,
                  reg_method, reg_lambda, epsilon, T, J_org):
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
    for step in range(max_steps):
        #learning step
        h_primes = zeros(Nneur)
        J_primes = zeros([Nneur,Nneur])
        for r in range(Nneur):
            params = np.append(h[r], J[:,r])
#            sigmas_r = sigmas[r, :]
            h_primes[r], J_primes[:,r] = f_prime_r(sigmas, r, params,
                                                    reg_method, reg_lambda, T)
        np.fill_diagonal(J_primes, 0)
        h -= h_lambda*h_primes        
        J -= J_lambda*J_primes
        rec_err = reconstruction_error(J_org, J)
        min_av_max.append(np.array([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J), rec_err]))
        if step % 10 == 0:
            print("Step",step," Rec. Error:", rec_err)
        
#    A = (np.triu(J, k=1) + np.tril(J, k=-1).T)/2
#    A += A.T
#    J = A
    return h, J, np.array(min_av_max)

def f_prime_r(sigmas, rth, params, reg_method, reg_lambda, T):
    #calculates derivative of f_r (as in Aurell, 2012)
    Nneur = sigmas.shape[0]
    f_primes_J = zeros(Nneur)
#    for sigma in sigmas.T:
    e_s = exp_sigma_r(params, sigmas, rth, T)
    pes = e_s/(1+e_s)
    f_primes_h = -np.sum(2* pes * sigmas[rth,:]/T)

    Jnorm = np.linalg.norm(params[1:])
    for s in range(Nneur):
        Jsr = params[1:]
        f_primes_J[s] -= np.sum(2*pes*sigmas[rth,:]*sigmas[s,:]/T) 
    
    f_primes_J /= sigmas.shape[1]
    f_primes_h /= sigmas.shape[1]
    ###l1-reg
    if reg_method == "sign":
        f_primes_J += reg_lambda*np.sign(params[1:])
        
    if reg_method == "lasso":
        f_primes_h[np.abs(f_primes_h)<=reg_lambda] = 0 
        f_primes_h[f_primes_h>reg_lambda] -= reg_lambda
        f_primes_h[f_primes_h<reg_lambda] += reg_lambda
        
    ###l2-reg
    if reg_method == "l2":
        f_primes_J += reg_lambda*params[1:]
    return f_primes_h, f_primes_J

def exp_sigma_r(params, sigmas, rth, T):
    rth = int(rth)
    act_without_r = np.delete(sigmas, rth, 0)    
    sum_without_r = np.dot(act_without_r.T, np.delete(params[1:], rth, 0))
    return np.exp(-2*sigmas[rth,:]*(params[0]+sum_without_r)/T)

def batch_plm(sigmas, max_steps, batch_size, h, J, h_lambda,
                  J_lambda, reg_lambda, epsilon, T, method):
    "Stochstic Pseudo_likelihood Maximization"
    if method == "NR":
        prime_func = f_prime_NR
        h_lambda = 1.
        J_lambda = 1.
    elif method == "seq":
        prime_func = f_prime_seq
    B_tot = sigmas.shape[1]
    for step in range(max_steps):
        np.random.shuffle(np.transpose(sigmas))
        for b in range(int(B_tot/batch_size)):
            batch = sigmas[:, b*batch_size:(b+1)*batch_size]

            #learning step
            h_primes, J_primes = prime_func(batch, h, J, reg_lambda, T)
#            print(h_primes)
            h -= h_lambda*h_primes        
            J -= J_lambda*J_primes
            
        print("Step", step, "Error:", np.sum(h_primes**2) + np.sum(J_primes**2))
#    A = (np.triu(J_primes, k=1) + np.tril(J_primes, k=-1).T)/2
#    A += A.T
#    J = A
    return h, J

def exp_sigma_r_min(params, sigmas, arth, reg_lambda, T):
    arth = int(arth)
    h_r = params[0]
    J_r = params[1:]
    act_without_r = np.delete(sigmas, arth, 0)
    sum_without_r = np.dot(act_without_r.T, J_r)
    return np.average(np.exp(-2*sigmas[arth,:]*(h_r+sum_without_r)/T)) + reg_lambda*np.linalg.norm(params[1:])

def f_prime_r_min(params, sigmas, arth, reg_lambda, T):
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
    "Pseudo_likelihood Maximization with scipy minimizer"
    ""
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
    "Pseudo_likelihood Maximization with scipy minimizer"
    ""
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
#    print(np.average(h_coeffs), np.average(J_coeffs))
    return h, J, np.array(min_av_max)


def calc_exp_corr(N, h, J, beta):
    #calculates mean magnetixzation (<s_i>, expectation)
    #and mean correlation  <s_is_j>
    p_tot = 0
    model_exps = zeros(N)
    corrs = zeros([N, N])
    for n in xrange(0, 2**N, 1):
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

def boltzmann_learning(N, s_act, max_steps, l_rate, h, J, Nsamples, Nflips,
                  sample_after_step, sample_per_steps, T):
    """Boltzmann Learning with Monte Carlo sampling to determine 
    model expectations and correlations"""
    #stopping condition?
    mag = np.average(s_act, axis=1)
    corrs = zeros([N,N])
    for i in range(s_act.shape[1]):
       corrs += np.outer(s_act[:,i], s_act[:,i])
    corrs /= s_act.shape[1]
    
    min_av_max = []
    for step in range(max_steps):
        model_exps, model_corrs = calc_exps_mc(h, J, Nsamples, Nflips,
                  sample_after_step, sample_per_steps, T)
        h += l_rate*(mag - model_exps)
        J += l_rate*(corrs - model_corrs)
        min_av_max.append([np.min(h), np.average(h), 
                                     np.max(h),
                                     np.min(J), np.average(J), 
                                     np.max(J)])
    
#        if np.min(h) - min_av_max[-1,0] < epsilon and np.min(h) - min_av_max[-1,0]:
#            break
#        print(np.average(h), np.average(J))
    return h, J, np.array(min_av_max)

def calc_exps_mc(h, J, Nsamples, Nflips,
                  sample_after_step, sample_per_steps, T):
    mc_samples = metropolis_mc(h, J, Nsamples, Nflips,
                  sample_after_step, sample_per_steps, T)
    model_exps = np.average(mc_samples, axis=1)
#    print(model_exps.shape)
    
    c = zeros([h.shape[0],h.shape[0]])
    for i in range(Nsamples):
       c += np.outer(mc_samples[:,i], mc_samples[:,i])
    model_corrs =  c/Nsamples
    return model_exps, model_corrs

######Native Mean Field
def nMF(s_act = 0, approx_type = ""):
    """Mean Field approximation of inverse Ising 
    (Hertz Ising Models for Inferring Network Structure from Spike Data)"""
    mag = np.average(s_act, axis=1)
    C = calc_correlations(s_act, mag)
    #include 1/beta?
    J = -np.linalg.inv(C)
#    np.fill_diagonal(J, 0)
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
#        np.fill_diagonal(J, 0)
        h = np.arctanh(mag) - np.dot(J, mag) + np.multiply(mag, np.dot(np.square(J),  1-np.square(mag)))
    
    return h,J

def reconstruction_error(original_J, inferred_J):
    """Calculates reconstruction error as in Aurell 2012 """
    #\Delta = \frac{1}{1/\sqrt{N}}<(J_{ij}^* - J_{ij})^2>^{1/2}.
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