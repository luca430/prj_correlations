# Script to compute global measures using graph-tools.
# NB: this script is suppose to run on a virtual machine with a volume attached. In case you are running locally
# pay attention to folders.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import graph_tool.all as gt
from global_funcs import *

# compute global quantities for original graphs and save results in dedicated dictionaries
for n_nodes in [100, 200, 500, 1000]:
    
    print(f"Computing the global variables of N={n_nodes} networks.")
    graphs_folder = os.path.join('..', 'graphs')
    N_dict = {}
    for i in range(25):
        N_dict[f"{i+1}"] = {}

    for file in os.listdir(graphs_folder):
        base_name = os.path.splitext(file)[0]
        _, n_str, i_str = base_name.split('_')

        if int(n_str) == n_nodes:
            file_dict = N_dict[i_str]
            N_dict[i_str] = compute_global_variables(os.path.join(graphs_folder,file), file_dict, load=True)

    #Save the dictionary
    print("\n")
    np.save(f"global_variables2/N_{n_nodes}_global_original.npy", N_dict)

# compute global quantities for reconstructed graphs and save results in dedicated dictionaries
for n_nodes in [100, 200, 500, 1000]:

    print(f"Computing the global variables of N={n_nodes}.")
    path = f"/mnt/corr_data/kuramoto/filtered_corr/N{n_nodes}/Noiseless"

    #N_dict = build_dict()
    N_dict = np.load(f'global_variables2/N_{n_nodes}_global.npy', allow_pickle=True).item()
    ### ---I'M ADDING THE FOLLOWING TO INCLUDE THE NAIVE RMT KEYS IN THE DICTIONARY---
    for i in range(25):
        for k in [0,10,15,25,50]:
            N_dict[f"{i+1}"][f"k{k}"]["NaiveRMT"] = {}
            for p in ["p0.1", "p0.15", "p0.2", "p0.25"]:
                N_dict[f"{i+1}"][f"k{k}"]["NaiveRMT"][p] = {}
    ### --- ###

    for subdir, dirs, files in os.walk(path):
        for file in files:
            print(f"{file}", end='\r')
            idx, k, method, thresh = extract_file_information(subdir, file)
            if method == "NaiveRMT":  ## Only compute the features for missing graphs
                #make graph from correlation matrix
                edge_list = mat2edgelist(os.path.join(subdir,file))
        
                #Fill the appropriate dictionary entry with the gloal variables
                file_dict = N_dict[idx][k][method][thresh]
                N_dict[idx][k][method][thresh] = compute_global_variables(edge_list, file_dict)

    #Save the dictionary
    print("\n")
    np.save(f"global_variables2/N_{n_nodes}_global.npy", N_dict)

