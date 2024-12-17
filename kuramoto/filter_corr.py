# Script to filter correlation matrices and store results in the desired folders.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import filter_func as ff
import multiprocessing
from multiprocessing import Manager

def filter_corr(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        print(f"\tComputing... {counter.value}/{L}", end="\r")
     
     # Extract input/output folder paths
    input_, output_ = params

    ff.apply_thresh(input_, output_)

def main():
     
    for k in [0.0, 1.0, 1.5, 2.5, 5.0]:
        print(f"Processing k={k}")
        input_folder = "./kuramoto/data/corr_matrices/K_{}".format(k)
        output_folder = "./kuramoto/data/filtered_matrices/K_{}".format(k)
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through each file in the input folder
        params = []
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".csv.gz"):
                input_file_path = os.path.join(input_folder, file_name)
                params.append([input_file_path, output_folder])
        L = len(params)

        # Create a shared counter and lock using Manager
        with Manager() as manager:
            counter = manager.Value('i', 0)  # Shared counter
            lock = manager.Lock()  # Shared lock

            # Parallel processing
            num_cores = 8  # Use physical cores
            with multiprocessing.Pool(processes=num_cores) as pool:
                pool.starmap(filter_corr, [(param, counter, lock, L) for param in params])

        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
        print('\tDone!')

if __name__ == "__main__":
    main()

