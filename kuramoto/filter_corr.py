# Script to filter correlation matrices and store results in dedicated folders in 'data/K_{}/filtered_corr/'.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gzip
import numpy as np
import filter_func as ff

corr_path = "./kuramoto/data/corr_matrices"
filtered_path = "./kuramoto/data/filtered_corr"
os.makedirs(filtered_path, exist_ok=True) # Create the output folder if it doesn't exist

for k in [0.0, 1.0, 1.5, 2.5, 5.0]:
    print("Filtering k={}...".format(k), end="\r")
    input_path = os.path.join(corr_path,"K_{}".format(k))
    output_path = os.path.join(filtered_path,"K_{}".format(k))

    for file_name in os.listdir(input_path):
        ff.apply_thresh(os.path.join(input_path,file_name), output_path)

print("Done!")
