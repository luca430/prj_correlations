"Script to create the correlation matrices for white noise and store them in folder `white_noise/data/corr_matrix`."

import os
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

source_folder = "white_noise/data/time_series"
target_folder = "white_noise/data/corr_matrix"
# Create the output folders if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

for file_name in os.listdir(source_folder):
    if file_name.startswith("white"):
        print("Running {}...".format(file_name), end='\r')
        x_vals_loaded = np.loadtxt(os.path.join(source_folder,file_name), delimiter=",")
        standized_vals = standardize_matrix(x_vals_loaded)
        correlation_matrix = np.corrcoef(standized_vals, rowvar=False)

        # Save the correlations
        file_path_out = os.path.join(target_folder,file_name)
        np.savetxt(file_path_out, correlation_matrix, delimiter=",")

        print("{} ...done!".format(file_name))
