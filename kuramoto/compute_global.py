# Script to compute global measures using graph-tools.
# NB: this script is suppose to run on a virtual machine with a volume attached. In case you are running locally
# pay attention to folders.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from global_funcs import *

output_path = "./kuramoto/data/global_measures"
os.makedirs(output_path, exist_ok=True)

### RECONSTRUCTED NETWORKS ###
for k in [0.0, 1.0, 1.5, 2.5, 5.0]:

    print(f"Computing global measures for K={k}...")
    path = f"./kuramoto/data/filtered_corr/K_{k}"

    N_dict = build_dict()

    for subdir, dirs, files in os.walk(path):
        for file in files:
            idx, N, method, thresh = extract_file_information(os.path.join(subdir,file))

            #make graph from correlation matrix
            edge_list = mat2edgelist(os.path.join(subdir,file))
    
            #Fill the appropriate dictionary entry with the gloal variables
            file_dict = N_dict[f"N_{N}"][int(idx)][method][thresh]
            N_dict[f"N_{N}"][int(idx)][method][thresh] = compute_global_variables(edge_list, file_dict)

    # Save the dictionary
    file_path = os.path.join(output_path, f"K_{k}_global.npy")
    np.save(file_path, N_dict)

### ORIGINAL NETWORKS ###
print(f"Computing global measures for original networks...")
graphs_folder = "./graphs"
N_dict = {}
for N in [100, 200, 500, 1000]:
    N_dict[f"N_{N}"] = {}
    for i in range(1,26):
        N_dict[f"N_{N}"][i] = {}

for file in os.listdir(graphs_folder):
    base_name = os.path.splitext(file)[0]
    _, n_str, i_str = base_name.split('_')

    file_dict = N_dict[f"N_{n_str}"][int(i_str)]
    N_dict[f"N_{n_str}"][int(i_str)] = compute_global_variables(os.path.join(graphs_folder,file), file_dict, load=True)

# Save the dictionary
file_path = os.path.join(output_path, f"original_global.npy")
np.save(file_path, N_dict)

