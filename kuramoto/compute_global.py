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
    
        input_, output_, k, K_dict = params

        # Since python is not good at manage parallelized objects it is necessary to un-nest nested dictionaries (don't really know why but it works)
        if k == "original":
            base_name = os.path.splitext(input_)[0]
            split_base_name = base_name.split('/')
            name = split_base_name[-1]
            _, N, idx = name.split('_')
            nested_dict = K_dict[k]
            nested_dict[f"N_{N}"][int(idx)] = compute_global_variables(input_, load=True)
            K_dict[k] = nested_dict
        else:
            idx, N, method, thresh = extract_file_information(input_)
            edge_list = mat2edgelist(input_)
            nested_dict = K_dict[f"K_{k}"]
            nested_dict2 = nested_dict[f"N_{N}"]
            nested_dict3 = nested_dict2[int(idx)]
            nested_dict3[method][thresh] = compute_global_variables(edge_list)
            nested_dict2[int(idx)] = nested_dict3
            nested_dict[f"N_{N}"] = nested_dict2
            K_dict[f"K_{k}"] = nested_dict

def main():
    output_folder = "/mnt/global_measures"
    os.makedirs(output_folder, exist_ok=True)

    num_cores = int(os.getenv("NUM_CORES", 8))  # Number of cores used. Default is 8

    ### RECONSTRUCTED NETWORKS ###
    for k in [0.0, 1.0, 1.5, 2.5, 5.0, "original"]:
        print(f"Computing global measures for k={k}")
        if k != "original":
            input_folder = f"/mnt/filtered_matrices/kuramoto/K_{k}"
        else:
            input_folder = "./graphs"

        # Gather input file paths
        params = []
        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if file_name.endswith(".npz") or file_name.endswith(".gml"):
                    input_file_path = os.path.join(root, file_name)
                    params.append([input_file_path, output_folder, k])

        L = len(params)

        # Create a shared variables using Manager
        with Manager() as manager:
            K_dict = manager.dict(build_dict())  # Shared dictionary
            counter = manager.Value('i', 0)  # Shared counter
            lock = manager.Lock()  # Shared lock

            # Parallel processing
            with multiprocessing.Pool(processes=num_cores) as pool:
                pool.starmap(compute_global, [(param + [K_dict], counter, lock, L) for param in params])
            standard_dict = dict(K_dict)

        # Save the dictionary
        file_path = os.path.join(output_folder, f'K_{k}.npy')
        np.save(file_path, standard_dict)

    print('Saving files...')
    # Collect all the results in a single dictionary and save it
    total_dict = build_dict()
    for k in ["K_0.0", "K_1.0", "K_1.5", "K_2.5", "K_5.0"]:
        temp = np.load(f"/mnt/global_measures/{k}.npy", allow_pickle=True).item()
        total_dict[k] = temp[k]
        # os.remove(f"/mnt/global_measures/{k}.npy")

    temp = np.load("/mnt/global_measures/K_original.npy", allow_pickle=True).item()
    total_dict['original'] = temp['original']
    # os.remove("/mnt/global_measures/K_original.npy")

    file_path = os.path.join(output_folder, 'kuramoto.npy')
    np.save(file_path, total_dict)

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
    print('\tDone!')

if __name__ == "__main__":
    main()