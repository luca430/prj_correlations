import os
import time
import numpy as np
np.random.seed(1234)

# Define the parameters
T = 10
dt = 0.005
folder_name = "time_series"
graphs_folder = os.path.join('..', 'graphs')

# Create the output folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Iterate through each file in the 'graphs' folder
for file_name in os.listdir(graphs_folder):
    if file_name.endswith(".gml"):
        # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
        base_name = os.path.splitext(file_name)[0]  # remove .gml extension
        _, n_str, i_str = base_name.split('_')
        n = int(n_str)
        i = int(i_str)

        print("white_{}_{}.csv ...".format(n, i), end="\r")
        ti = time.time()

        x_vals = np.random.normal(size=(int(T/dt), n))

        # Save the results
        file_path_out = os.path.join(folder_name, "white_{}_{}.csv".format(n, i))
        np.savetxt(file_path_out, x_vals, delimiter=",")
        print("white_{}_{}.csv ...done! \t {} s".format(n, i, np.round(time.time() - ti)))