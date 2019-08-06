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
    with open(path+"/data_n%s_%s.txt"%(10, 100,  0), "rb") as fp:  
       data = pickle.load(fp)
 
    h = data['h']
    J = data['J']
    patts = data['ordered_patterns']
    
    all_states =  get_lem_neighbourhood(patts, 1)
    #all_energies = np.array(calc_energy_list([h,J], all_states))
    #all_states = all_states[np.where(np.array(all_energies) < min(all_energies)+1.5)[0],:]
    #all_energies = np.array(calc_energy_list([h,J], all_states))
    
    lems, list_of_basins, list_of_lems = determine_basins(h, J, all_states)
    
    G = nx.Graph()
    for i, ui in enumerate(list_of_basins):
        G.add_node(str(i+1))
        for j, uj in enumerate(list_of_basins):
    
            ##check for overlap between clusters (basins)
            for patti in ui:
                for pattj in uj:
                    if np.all(patti==pattj) and i!=j:
                        G.add_edge(str(i+1), str(j+1))
                        
    nx.write_gpickle(G, "N100_dist1.gpickle")
                        
if __name__ == "__main__":
        main()

