# Script to generate Kuramoto time series on netwroks contained in './graphs' and store 
# them in the desired folder.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gzip
import numpy as np
import networkx as nx
import runge_kutta as rk
import multiprocessing
from multiprocessing import Manager

np.random.seed(1234)

# Define the kuramoto function (normalized with respect to the mean degree)
def kuramoto(x, t, w, k, A):
    
    phase_diff = x[:, np.newaxis] - x
    sin_phase_diff = np.sin(phase_diff)
    coupling_term = k/np.mean(np.sum(A, axis=1))*np.dot(A, sin_phase_diff)
    
    return w + np.diag(coupling_term)

def ts_generator(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        print(f"Computing... {counter.value}/{L}", end="\r")
    
    # Extract input/output folder paths
    input_, output_, n, i = params

    G = nx.read_gml(input_)
    A = nx.adjacency_matrix(G).toarray()

    # Initial conditions and equation parameters
    T, dt = 10, 0.005
    x0 = np.random.rand(n)*2*np.pi
    mu, sig = 0, 20
    w = np.random.normal(mu, sig, size=n)
    K_c = 30    # rescaling factor
    K = [0.0, K_c, 1.5*K_c, 2.5*K_c, 5*K_c] # Kuramoto coupling regimes

    # Run the Runge-Kutta for different coupling regimes
    for k in K:
        out_folder = os.path.join(output_, "K_{}".format(k/K_c))
        os.makedirs(out_folder, exist_ok=True)
        t_vals, x_vals = rk.runge_kutta(kuramoto, x0, T, dt = dt, w = w, k = k, A = A)

        # Save the results
        file_path_out = os.path.join(out_folder, "kuramoto_{}_{}.csv.gz".format(n, i))
        with gzip.open(file_path_out, "wt") as f:
            np.savetxt(f, x_vals, delimiter=",")

def main():
    input_folder = "./graphs"
    output_folder = "./kuramoto/data/time_series"
    os.makedirs(output_folder, exist_ok=True)

    # Prepare input parameters
    params = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".gml"):
            # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
            base_name = os.path.splitext(file_name)[0]  # remove .gml extension
            _, n_str, i_str = base_name.split('_')
            n = int(n_str)
            i = int(i_str)

            input_file_path = os.path.join(input_folder, file_name)
            params.append([input_file_path, output_folder, n, i])
    L = len(params)

    # Create a shared counter and lock using Manager
    with Manager() as manager:
        counter = manager.Value('i', 0)  # Shared counter
        lock = manager.Lock()  # Shared lock

        # Parallel processing
        num_cores = 8  # Use physical cores
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(ts_generator, [(param, counter, lock, L) for param in params])

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
    print('Done!')

if __name__ == "__main__":
    main()