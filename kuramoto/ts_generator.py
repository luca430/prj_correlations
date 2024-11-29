import os
import time
import numpy as np
import networkx as nx
import runge_kutta as rk
np.random.seed(1234)

# Parameters
folder_name = "time_series"
graphs_folder = os.path.join('..', 'graphs2')
T = 10
dt = 0.005

# Create the output folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Define the kuramoto function
def kuramoto(x, t, w, k, A):
    
    phase_diff = x[:, np.newaxis] - x
    sin_phase_diff = np.sin(phase_diff)
    coupling_term = k/np.mean(np.sum(A, axis=1))*np.dot(A, sin_phase_diff)
    
    return w + np.diag(coupling_term)
    
# Iterate through each file in folder 'graphs'
for file_name in os.listdir(graphs_folder):
    if file_name.endswith(".gml"):
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
        K_c = 30
        K = [K_c, 1.5*K_c, 2.5*K_c, 5*K_c]

        # Run the Runge-Kutta for different coupling regimes
        for k in K:
            print("kuramoto_k{}_{}_{}.csv ...".format(int(k/K_c*10), n, i), end="\r")
            ti = time.time()
            t_vals, x_vals = rk.runge_kutta(kuramoto, x0, T, dt = dt, w = w, k = k, A = A)

            # Save the results
            file_path_out = os.path.join(folder_name, "kuramoto_k{}_{}_{}.csv".format(int(k/K_c*10), n, i))
            np.savetxt(file_path_out, x_vals, delimiter=",")

            print("kuramoto_k{}_{}_{}.csv ...done! \t {} s".format(int(k/K_c*10), n, i, np.round(time.time() - ti)))
