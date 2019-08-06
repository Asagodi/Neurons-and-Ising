#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:00:43 2019

@author: abel
"""
import sys
import numpy as np
from scipy import *
from lec import *
import pickle
import os
import networkx as nx


def main():
    path=os.getcwd()
    ien = sys.argv[1]
    energy_spectrum = arange(0., 10., 1.)
    max_energy = energy_spectrum[ien]
    with open(path+"/data_n%s_%s.txt"%(125,  0), "rb") as fp:  
       data = pickle.load(fp)
 
    h = data['h']
    J = data['J']
    patts = data['ordered_patterns']
    
    all_states =  get_lem_neighbourhood(patts, 1)
    
    all_energies = np.array(calc_energy_list([h,J], all_states))
    all_states = all_states[np.where(np.array(all_energies) < min(all_energies)+max_energy)[0],:]
    all_energies = np.array(calc_energy_list([h,J], all_states))
    
    lems = []
    for state in all_states:
        if state in patts:
            lems.append(state)
    
    lem_energies = calc_energy_list([h,J], np.array(lems))
    op, oe = order_patterns_accto_energies(lems, list(lem_energies), h, J)
    
    lems_, list_of_basins, list_of_lems = determine_basins(h, J, all_states)
    
    spectrum_data = {}
    consistency = lems == lems_
    spectrum_data["consistency"] = consistency
    spectrum_data["lems"] = lems
    spectrum_data["list_of_basins"] = list_of_basins
    spectrum_data["list_of_lems"] = list_of_lems
    spectrum_data["energy_spectrum"] = energy_spectrum
    
    #equivalent to perhombasin graph algorithm, but with array
    c_mat = zeros((lems.shape[0], lems.shape[0]))
    for i, ui in enumerate(list_of_basins):
        for j, uj in enumerate(list_of_basins):
            for patti in ui:
                for pattj in uj:
                    if np.all(patti==pattj) and i!=j:
                        c_mat[str(i+1), str(j+1)] = 1.
    
    spectrum_data["c_mat"] = c_mat
                        
    with open(path+"/sprectrum_data_%s.txt"%i, "wb") as fp:  
       pickle.dump(spectrum_data, fp)
                        
if __name__ == "__main__":
        main()

