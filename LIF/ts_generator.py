# Script to generate LIF time series on netwroks contained in './graphs' and store 
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

# Define state variables for the neurons
class LIFState:
    def __init__(self, n):
        self.fire = np.zeros(n)
        self.t_f = np.zeros(n)

# Define the LIF function
def LIF(u, t, k, A, state,
        u_rest=0, u_r=-5, theta=60, 
        R=50, tau=10, I=10, u_syn=65, tau_syn=5):
    
    state.fire[u > theta] = k
    state.t_f[u > theta] = t
    u[u > theta] = u_r
    g = state.fire*np.exp(-(t - state.t_f) / tau_syn)
    g = np.dot(A,g)
    
    return (u_rest - u + R*I)/tau - g*(u - u_syn)

def ts_generator(params, counter, lock, L):
    with lock:  # Use explicit lock for thread safety
        counter.value += 1
        if counter.value % 10 == 0:
            print(f"Computing... {counter.value}/{L}", end="\r")
    
    # Extract input/output folder paths
    input_, output_, n, i = params

    G = nx.read_gml(input_)
    A = nx.adjacency_matrix(G).toarray()

    # Initial conditions and equation parameters
    T, dt = 10, 0.005
    u0 = -20 + np.random.rand(n)*40
    K_c = 0.4    # rescaling factor
    K = [0.0, K_c, 1.5*K_c, 2.5*K_c, 5*K_c] # LIF coupling regimes

    # Run the Runge-Kutta for different coupling regimes
    for k in K:
        state = LIFState(n)
        out_folder = os.path.join(output_, "K_{}".format(np.round(k/K_c,1)))
        os.makedirs(out_folder, exist_ok=True)
        t_vals, x_vals = rk.runge_kutta(LIF, u0, T, A=A, k=k, state=state)

        # Save the results
        file_path_out = os.path.join(out_folder, "LIF_{}_{}.csv.gz".format(n, i))
        with gzip.open(file_path_out, "wt") as f:
            np.savetxt(f, x_vals, delimiter=",")

def main():
    input_folder = "./graphs"
    output_folder = "./LIF/data/time_series"
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

            input_file_path = os.path.join(input_folder, file_name)
            params.append([input_file_path, output_folder, n, i])
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