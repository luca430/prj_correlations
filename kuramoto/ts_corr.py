# Script to generate the correlation matrices for kuramoto time series and store them in './kuramoto/data/corr_matrix'.

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

for k in [0.0, 1.0, 1.5, 2.5, 5.0]:
    input_folder = "./kuramoto/data/time_series/K_{}".format(k)
    output_folder = "./kuramoto/data/corr_matrices/K_{}".format(k)

    # Create the output folders if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.startswith("kuramoto"):
            print("Running {}...".format(file_name), end='\r')
            with gzip.open(os.path.join(input_folder,file_name), "rt") as f:
                x_vals_loaded = np.loadtxt(f, delimiter=",")

            standized_vals = standardize_matrix(x_vals_loaded)
            correlation_matrix = np.corrcoef(standized_vals, rowvar=False)
            eigenvalues = np.linalg.eigvals(correlation_matrix)

            # Save the results
            file_path_out = os.path.join(output_folder, file_name)
            with gzip.open(file_path_out, "wt") as f:
                    np.savetxt(f, correlation_matrix, delimiter=",")

            print("{} ...done!".format(file_name))
