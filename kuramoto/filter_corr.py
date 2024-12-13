# Script to filter correlation matrices and store results in dedicated folders in 'data/K_{}/filtered_corr/'.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import filter_func as ff

for k in [0.0, 1.0, 1.5, 2.5, 5.0]:

    input_folder = "./kuramoto/data/corr_matrices\K_{}".format(k)
    output_folder = "./kuramoto/data/filtered_corr\K_{}".format(k)
    os.makedirs(output_folder, exist_ok=True)

    print("Filtering k={}...".format(k), end="\r")

    for file_name in os.listdir(input_folder):
        ff.apply_thresh(os.path.join(input_folder,file_name), output_folder)

print("Done!")
