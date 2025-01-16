# Script to compute global measures using graph-tools.
# NB: this script is suppose to run on a virtual machine with a volume attached. In case you are running locally
# pay attention to folders.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from global_funcs import *
import multiprocessing
from multiprocessing import Manager

def compute_global(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        print(f"Computing... {counter.value}/{L}", end="\r")
     
     # Extract input/output folder paths
    input_, output_, k, N_dict = params

    idx, N, method, thresh = extract_file_information(input_)
    edge_list = mat2edgelist(input_)
    file_dict = N_dict[f"N_{N}"][int(idx)][method][thresh]
    N_dict[f"N_{N}"][int(idx)][method][thresh] = compute_global_variables(edge_list, file_dict)
    # Save the dictionary
    file_path = os.path.join(output_, f"K_{k}_global.npy")
    np.save(file_path, N_dict)

def main():

    output_folder1 = "/mnt/global_measures/LIF/shapes"
    os.makedirs(output_folder1, exist_ok=True)
    output_folder2 = "/mnt/global_measures/LIF/spike_trains"
    os.makedirs(output_folder2, exist_ok=True)

    num_cores = int(os.getenv("NUM_CORES", 8))  # Number of cores used. Default is 8

    ### RECONSTRUCTED NETWORKS ###
    for k in [0.0, 1.0, 1.5, 2.5, 5.0]:
        print(f"Computing global measures for k={k}")
        input_folder1 = "/mnt/filtered_matrices/LIF/shapes/K_{}".format(k)
        input_folder2 = "/mnt/filtered_matrices/LIF/spike_trains/K_{}".format(k)

        # Iterate through each file in the input folder
        params = []
        for file_name in os.listdir(input_folder1):
            if file_name.endswith(".npz"):
                N_dict = build_dict()
                input_file_path = os.path.join(input_folder1, file_name)
                params.append([input_file_path, output_folder1, k, N_dict])
        for file_name in os.listdir(input_folder2):
            if file_name.endswith(".npz"):
                N_dict = build_dict()
                input_file_path = os.path.join(input_folder2, file_name)
                params.append([input_file_path, output_folder2, k, N_dict])
        L = len(params)

        # Create a shared counter and lock using Manager
        with Manager() as manager:
            counter = manager.Value('i', 0)  # Shared counter
            lock = manager.Lock()  # Shared lock

            # Parallel processing
            with multiprocessing.Pool(processes=num_cores) as pool:
                pool.starmap(compute_global, [(param, counter, lock, L) for param in params])

        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
        print('\tDone!')

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
    file_path = os.path.join(output_folder, f"original_global.npy")
    np.save(file_path, N_dict)
    print('\tDone!')

if __name__ == "__main__":
    main()


