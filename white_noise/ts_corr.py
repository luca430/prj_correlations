# Script to create the correlation matrices for white noise and store them in desired folder.

import os
import sys
import gzip
import numpy as np
import multiprocessing
from multiprocessing import Manager
np.random.seed(1234)

def standardize_matrix(matrix):
    """
    Standardize each column of the given matrix to have mean 0 and variance 1.

    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    
    return standardized_matrix

def ts_corr(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        print(f"Computing... {counter.value}/{L}", end="\r")
     
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
        if file_name.endswith(".csv.gz"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            params.append([input_file_path, output_file_path])
    L = len(params)

    # Create a shared counter and lock using Manager
    with Manager() as manager:
        counter = manager.Value('i', 0)  # Shared counter
        lock = manager.Lock()  # Shared lock

        # Parallel processing
        num_cores = 8  # Use physical cores
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(ts_corr, [(param, counter, lock, L) for param in params])

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
    print('Done!')

if __name__ == "__main__":
    main()
