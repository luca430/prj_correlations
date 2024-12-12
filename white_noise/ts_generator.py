# Script to generate white noise time series and store them in folder `./white_noise/data/time/series`.

import os
import gzip
import numpy as np
np.random.seed(1234)

# Define the parameters
T = 10
dt = 0.005
folder_name = "./white_noise/data/time_series"
graphs_folder = "./graphs"

# Create the output folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Iterate through each file in the 'graphs' folder
file_number = 0
for file_name in os.listdir(graphs_folder):
    if file_name.endswith(".gml"):
        file_number += 1
        print("Computing {}/100...".format(file_number), end="\r")

        # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
        base_name = os.path.splitext(file_name)[0]  # remove .gml extension
        _, n_str, i_str = base_name.split('_')
        n = int(n_str)
        i = int(i_str)

        x_vals = np.random.normal(size=(int(T/dt), n))

        # Save the results
        file_path_out = os.path.join(folder_name, "white_{}_{}.csv.gz".format(n, i))
        with gzip.open(file_path_out, "wt") as f:
                np.savetxt(f, x_vals, delimiter=",")

print("Done!")
