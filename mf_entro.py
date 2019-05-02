#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:53:04 2019

@author: abel
"""
import numpy as np
from scipy import *


def setup_nodes(N):
    a = 0
    V_node = [[] for i in range(N)]
    node = [[] for i in range(N)]
    F_node = [[] for i in range(int((N**2-N)/2))] #(N-1)*N/2;
    
    for i in range(N):
        for j in range(i+1,N):
            V_node[i].append(a)
            V_node[j].append(a)
            node[i].append(j)
            node[j].append(i)
            F_node[a].append(i)
            F_node[a].append(j)
            a += 1
    print(a)
    return V_node, node, F_node
    

def initial_message(N):
    m_ia = np.random.uniform(-1.,1.,[N,N-1])
    return m_ia

def make_flat_hopfield_J(pattern_list):
    N = pattern_list[0].shape[0]
    J = [[] for i in range(int((N**2-N)/2))]
    a = 0
    for i in range(N):
        for j in range(i+1,N):
            weightsum=0.0;
            for k in range(len(pattern_list)):
                weightsum += pattern_list[k][i]*pattern_list[k][j]
            J[a] = weightsum/N
            a+=1
    return J


def make_flat_J(J):
    N = J.shape[0]
    J_ = [[] for i in range(int((N**2-N)/2))]
    a = 0
    for i in range(N):
        for j in range(i+1,N):
            J_[a] = J[i,j]
            a+=1
    return J_
    

def iteration(m_ia, h, J, max_steps, delta=10**-2):
    N = h.shape[0]
    for ite in range(max_steps):
        m_ia_old = np.array(m_ia, copy=True)
        for  it in range(N):
            node_i=it;
            sum_=0.0;
            ll=0;
            maI = zeros(N)
            aa=0;
            for itr in V_node[node_i]:
                Kb= len(F_node[itr])
                l=0;
                m_jb = [0.]*(Kb-1)
#                if(Kb==2)
                l=0;
                for j in range(Kb):
                    ib=F_node[itr][j];
                    if(ib!=node_i):
                        if ib>node_i:
                            b=node_i
                        else:
                            b=node_i-1
#                        print(m_ia[ib][b])
                        m_jb[l]=m_ia[ib][b]
                        l+=1
                product01=1.0;
                for j in range(Kb-1):
                    product01*=m_jb[j];
                product01*=tanh(J[itr]);
#                print(product01)
                maI[ll]=product01;
                l+=1
                sum_+=arctanh(product01);   
            
            for itr in V_node[node_i]:
                Mia=tanh(h[node_i]+sum_-arctanh(maI[ll]));
#                print(node_i,itr)
#                print(Mia, m_ia[node_i][ll])
                m_ia[node_i][ll]=Mia
                ll+=1
            if global_convergence(m_ia, m_ia_old, delta):
                return m_ia
    
    print("Did not converge in %d steps" %max_steps)


def global_convergence(m_ia, m_ia_old, delta):
    s=0
    edges=0
    for i in range(N):
        for j in range(N-1):
            edges+=1
            subs01=abs(m_ia[i][j]-m_ia_old[i][j]);
            if subs01<delta:
                s+=1
    if s==edges:
        return True
    else:
        return False
    
def comput_mag_corre(m_ia, h, J, max_steps):
    N = h.shape[0]
    m_bp = zeros(N)
    
    for  it in range(N):
        node_i=it;
        sum_=0.0;
        ll=0;
        maI = zeros(N)
        aa=0;
        for itr in V_node[node_i]:
            Kb= len(F_node[itr])
            l=0;
            m_jb = [0.]*(Kb-1)
#                if(Kb==2)
            l=0;
            for j in range(Kb):
                ib=F_node[itr][j];
                if(ib!=node_i):
                    if ib>node_i:
                        b=node_i
                    else:
                        b=node_i-1
#                        print(m_ia[ib][b])
                    m_jb[l]=m_ia[ib][b]
                    l+=1
            product01=1.0;
            for j in range(Kb-1):
                product01*=m_jb[j];
            product01*=tanh(J[itr]);
#                print(product01)
            maI[ll]=product01;
            l+=1
            sum_+=arctanh(product01);   
        
        m_bp[node_i]=tanh(h[node_i]+sum_);
        
    ###TO DO calc corrs
    return  m_bp

    
    
def distance_entropy(m_ia, h, J, x, ref_sigma, beta=1.):
    h_p = beta*h + x*ref_sigma
    N = h.shape[0]
#    J = #write func?
    a = 0
    for i in range(N):
        for j in range(i+1,N):
            J[a] *= beta
            a+=1
    N = h.shape[0]
    Delta_Fi=0.0;
    Delta_Fa=0.0;
    E_a=0.0;
    E_i=0.0;
     
    for  it in range(N):
        node_i=it;
        suma=0.0;sumb=0.0;
        for itr in V_node[node_i]:
            Kb= len(F_node[itr])
            l=0;
            m_jb = [0.]*(Kb-1)
#                if(Kb==2)
            l=0;
            for j in range(Kb):
                ib=F_node[itr][j];
                if(ib!=node_i):
                    if ib>node_i:
                        b=node_i
                    else:
                        b=node_i-1
#                        print(m_ia[ib][b])
                    m_jb[l]=m_ia[ib][b]
                    l+=1
            product01=1.0;
            for j in range(Kb-1):
                product01*=m_jb[j];
            product01*=tanh(J[itr]);
            suma+=log(cosh(J[itr])*(1.0+product01));				
            sumb+=log(cosh(J[itr])*(1.0-product01));
        
        sumc=0.0;sumd=0.0;
        for itr in V_node[node_i]:
            Kb= len(F_node[itr])
            l=0;
            m_jb = [0.]*(Kb-1)
            for j in range(Kb):
                ib=F_node[itr][j];
                if(ib!=node_i):
                    if ib>node_i:
                        b=node_i
                    else:
                        b=node_i-1
#                        print(m_ia[ib][b])
                    m_jb[l]=m_ia[ib][b]
                    l+=1
            prod0=1.0;
            for j in range(Kb-1):
                prod0*=m_jb[j];
            prod_ = prod0
            prod0*=tanh(J[itr]);
            Jb=J[itr];
            a1=suma-log(cosh(Jb)*(1.0+prod0));
            a2=sumb-log(cosh(Jb)*(1.0-prod0));
            sumc+=(Jb*sinh(Jb)*(1.0+prod0)+Jb*cosh(Jb)*(1.0-pow(tanh(Jb),2.0))*prod_)*exp(a1+h_p[node_i])
            sumd+=(Jb*sinh(Jb)*(1.0-prod0)-Jb*cosh(Jb)*(1.0-pow(tanh(Jb),2.0))*prod_)*exp(a2-h_p[node_i])
            
        hi=h_p[node_i];
        Zi=exp(hi+suma)+exp(-hi+sumb);
        subs=x*ref_sigma[node_i];
        E_i+=((hi-subs)*(exp(hi+suma)-exp(-hi+sumb))+sumc+sumd)/Zi;
        Delta_Fi+=log(Zi);
    
    for  it in range(int((N**2-N)/2)):
        Ka=len(F_node[it])
        product=1.0e0;
        l=0;
        neib = [0]*Ka
        for itr in F_node[it]:
            neib[l]=itr;
            l+=1
        for j in range(Ka):
            if j==0:
                ib=neib[0];
                node_i=neib[1]
            
            else:
                ib=neib[1];node_i=neib[0];
            if ib>node_i:
                b=node_i
            else:
                b= (node_i-1)
            mia=m_ia[ib][b];
            product*=mia;
        Delta_Fa+=(Ka-1.0)*log(cosh(J[it])*(1.0+tanh(J[it])*product));
        E_a+=(Ka-1.0)*J[it]*(tanh(J[it])+product)/(1.0+tanh(J[it])*product);
        
    fe=(Delta_Fi-Delta_Fa)/float(N);
    eg=-(E_i-E_a)/float(N);
    entropy=fe+eg;
    return entropy


def secant_mp(d, x0, x1, epsilon, h, J, ref_sigma, beta=1., max_steps_k=100,
            max_steps_sq=5, max_steps_mp=5):
    #flat J
    q_til = d_to_q(d)
    J_var = beta*J
    N=h.shape[0]
    #secant method
    xk_1 = x0
    xk = x1
    h_var = beta*h + xk_1*ref_sigma
    mia = iteration(m_ia, h_, J, max_steps, delta=10**-4)
    mag = comput_mag_corre(m_ia, h_, J, max_steps)
    F_prev = np.dot(ref_sigma, mag)/N - q_til
    
    for k in range(max_steps_k):
        h_var = beta*h + xk*ref_sigma
        mia = iteration(m_ia, h_, J, max_steps, delta=10**-4)
        mag = comput_mag_corre(m_ia, h_, J, max_steps)
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
    
    
    s = distance_entropy(m_ia, h, J, x, ref_sigma)
    s -= xk*q_til
    return xk, s



h = np.loadtxt(path_to_docs+"h_N100l2_bsize10_rlam01.csv", delimiter=',')
J = np.loadtxt(path_to_docs+"J_N100l2_bsize10_rlam01.csv", delimiter=',')
s_act = np.loadtxt(path_to_docs + "s_act_head_ang_N100_36.csv", delimiter=',')

max_steps=50
m_ia = initial_message(N)
ref_sigma = -ones(N)
ds = []
fig = plt.figure()
ax = fig.add_subplot(111)
ds = []
sd = []
xs = arange(-3., 3., 0.04)
for x in xs:
    print(x)
    h_ = h + x * ref_sigma
    m_ia = iteration(m_ia, h_, J_, max_steps, delta=10**-2)
    mi = comput_mag_corre(m_ia, h_, J_, max_steps)
    q = np.dot(ref_sigma,mi)/float(N)
    s = distance_entropy(m_ia, h, J_, x, ref_sigma)
    sd.append(s-x*q)
    q = np.dot(ref_sigma, mi)/N
    d = (1-q)/2.
    ds.append(d)
cax = ax.plot(xs, ds, 'k', label='increasing x')
ds = []
sd = []
xs = arange(3., -3., -0.04)
for x in xs:
    print(x)
    h_ = h + x * ref_sigma
    m_ia = iteration(m_ia, h_, J_, max_steps, delta=10**-2)
    mi = comput_mag_corre(m_ia, h_, J_, max_steps)
    q = np.dot(ref_sigma,mi)/float(N)
    s = distance_entropy(m_ia, h, J_, x, ref_sigma)
    sd.append(s-x*q)
    q = np.dot(ref_sigma, mi)/N
    d = (1-q)/2.
    ds.append(d)
cax = ax.plot(xs, ds, '--r', label='decreasing x')
ax.set_ylabel("d")
ax.set_xlabel("x")
ax.set_title(r"$\beta$={0:0.1f}".format(beta))
ax.legend()
plt.show()

#hot to detect jumps?
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(ds, sd, '--r')
ax.set_ylabel("s(d)")
ax.set_xlabel("d")
ax.set_title("Entropy")
plt.show()



diff_list = []
for i,d in enumerate(ds[:-1]):
    diff = ds[i+1] - d
    diff_list.append(diff)
    
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(diff_list)
ax.set_ylabel("Diff")
ax.set_xlabel("d")
ax.set_title("Differences")
ax.legend()
plt.show()