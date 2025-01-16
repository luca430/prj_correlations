# Script to generate white noise time series and store them in desired folder.

import os
import sys
import gzip
import numpy as np
import multiprocessing
from multiprocessing import Manager

np.random.seed(1234)

def ts_generator(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        print(f"Computing... {counter.value}/{L}", end="\r")
    
    # Extract input/output folder paths
    output_, n = params

    # Compute time series
    T, dt = 10, 0.005
    x_vals = np.random.normal(size=(int(T/dt), n))

    # Save the results
    with gzip.open(output_, "wt") as f:
        np.savetxt(f, x_vals, delimiter=",")

def main():
    input_folder = "./graphs"
    output_folder = "/mnt/time_series/white_noise"
    os.makedirs(output_folder, exist_ok=True)

    num_cores = int(os.getenv("NUM_CORES", 8))  # Number of cores used. Default is 8

    # Prepare input parameters
    params = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".gml"):
            # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
            base_name = os.path.splitext(file_name)[0]  # remove .gml extension
            _, n_str, i_str = base_name.split('_')
            n = int(n_str)
            i = int(i_str)

            output_file_path = os.path.join(output_folder, f"white_{n}_{i}.csv.gz")
            params.append([output_file_path, n])
    L = len(params)

    # Create a shared counter and lock using Manager
    with Manager() as manager:
        counter = manager.Value('i', 0)  # Shared counter
        lock = manager.Lock()  # Shared lock

        # Parallel processing
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(ts_generator, [(param, counter, lock, L) for param in params])

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
    print('Done!')

if __name__ == "__main__":
    main()
