# Script to create the correlation matrices for white noise and store them in folder `./white_noise/data/corr_matrix`.

import os
import gzip
import numpy as np
np.random.seed(1234)

def standardize_matrix(matrix):
    """
    Standardize each column of the given matrix to have mean 0 and variance 1.

    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    
    return standardized_matrix

source_folder = "./white_noise/data/time_series"
target_folder = "./white_noise/data/corr_matrices"
# Create the output folders if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

file_number = 0
for file_name in os.listdir(source_folder):
    if file_name.startswith("white"):
        file_number += 1
        print("Computing {}/100...".format(file_number), end="\r")
        with gzip.open(os.path.join(source_folder,file_name), "rt") as f:
            x_vals_loaded = np.loadtxt(f, delimiter=",")

        standized_vals = standardize_matrix(x_vals_loaded)
        correlation_matrix = np.corrcoef(standized_vals, rowvar=False)

        # Save the results
        file_path_out = os.path.join(target_folder, file_name)
        with gzip.open(file_path_out, "wt") as f:
                np.savetxt(f, correlation_matrix, delimiter=",")

print("Done!")
