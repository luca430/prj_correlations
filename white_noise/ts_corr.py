# Script to create the correlation matrices for white noise and store them in folder `./white_noise/data/corr_matrix`.

import os
import gzip
import numpy as np
import multiprocessing
np.random.seed(1234)

def standardize_matrix(matrix):
    """
    Standardize each column of the given matrix to have mean 0 and variance 1.

    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    
    return standardized_matrix

def ts_corr(params):
     
     # Extract input/output folder paths
    input_, output_ = params

    with gzip.open(input_, "rt") as f:
        x_vals_loaded = np.loadtxt(f, delimiter=",")

    standized_vals = standardize_matrix(x_vals_loaded)
    correlation_matrix = np.corrcoef(standized_vals, rowvar=False)

    # Save the results
    with gzip.open(output_, "wt") as f:
            np.savetxt(f, correlation_matrix, delimiter=",")

def main():
     
    input_folder = "./white_noise/data/time_series"
    output_folder = "./white_noise/data/corr_matrices"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the 'graphs' folder
    params = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".gml"):

            # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
            base_name = os.path.splitext(file_name)[0]  # remove .gml extension
            _, n_str, i_str = base_name.split('_')
            n = int(n_str)
            i = int(i_str)

            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, "white_{}_{}.csv.gz".format(n, i))
            params.append([input_file_path, output_file_path])

    # Parallel processing
    num_cores = 10  # Use physical cores
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(ts_corr, params)

if __name__ == "__main__":
    main()
