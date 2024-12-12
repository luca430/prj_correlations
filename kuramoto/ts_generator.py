# Script to generate Kuramoto time series on netwroks contained in './graphs' and store 
# them in 'kuramoto/data/time_series'.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gzip
import numpy as np
import networkx as nx
import runge_kutta as rk
np.random.seed(1234)

# Parameters
ts_folder = "./kuramoto/data/time_series"
graphs_folder = "./graphs"
T = 10
dt = 0.005

# Create the output folder if it doesn't exist
os.makedirs(ts_folder, exist_ok=True)

# Define the kuramoto function (normalized with respect to the mean degree)
def kuramoto(x, t, w, k, A):
    
    phase_diff = x[:, np.newaxis] - x
    sin_phase_diff = np.sin(phase_diff)
    coupling_term = k/np.mean(np.sum(A, axis=1))*np.dot(A, sin_phase_diff)
    
    return w + np.diag(coupling_term)
    
# Iterate through each file in folder 'graphs'
file_number = 0
for file_name in os.listdir(graphs_folder):
    if file_name.endswith(".gml"):
        file_number += 1
        print("Computing {}/100...".format(file_number), end="\r")

        # Extract n and i from the file name
        base_name = os.path.splitext(file_name)[0]  # remove .gml extension
        _, n_str, i_str = base_name.split('_')
        n = int(n_str)
        i = int(i_str)

        # Load the graph
        file_path_in = os.path.join(graphs_folder, file_name)
        G = nx.read_gml(file_path_in)
        A = nx.adjacency_matrix(G).toarray()

        # Initial condition
        x0 = np.random.rand(n)*2*np.pi
        mu, sig = 0, 20
        w = np.random.normal(mu, sig, size=n)
        K_c = 30    # rescaling factor
        K = [0.0, K_c, 1.5*K_c, 2.5*K_c, 5*K_c]

        # Run the Runge-Kutta for different coupling regimes
        
        for k in K:
            out_folder = os.path.join(ts_folder, "K_{}".format(k/K_c))
            os.makedirs(out_folder, exist_ok=True)
            t_vals, x_vals = rk.runge_kutta(kuramoto, x0, T, dt = dt, w = w, k = k, A = A)

            # Save the results
            file_path_out = os.path.join(out_folder, "kuramoto_{}_{}.csv.gz".format(n, i))
            with gzip.open(file_path_out, "wt") as f:
                np.savetxt(f, x_vals, delimiter=",")

print("Done!")