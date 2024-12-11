"Script to generate many random graphs and store them in the `graphs` folder."

import os
import numpy as np
import networkx as nx

# parameters
N = [100, 200, 500, 1000]
num = 25
folder_name = "graphs"

# Create the folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

for n in N:
    p = np.sqrt(2)*np.log(n)/n # ensure that the random graph has a GCC
    for i in range(num):
        G = nx.erdos_renyi_graph(n, p, seed = 1234)
        file_path = os.path.join(folder_name, "graph_{}_{}.gml".format(n,i+1))
        nx.write_gml(G, file_path)