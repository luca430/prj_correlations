# Script to generate white noise time series and store them in folder `./white_noise/data/time/series`.

import os
import gzip
import numpy as np
import multiprocessing
np.random.seed(1234)

def ts_generator(params):
    
    # Extract input/output folder paths
    output_, n = params

    # Compute time series
    T, dt = 10, 0.005
    x_vals = np.random.normal(size=(int(T/dt), n))

    # Save the results
    with gzip.open(output_, "wt") as f:
            np.savetxt(f, x_vals, delimiter=",")

def main():
     
    input_folder = "./graphs"
    output_folder = "./white_noise/data/time_series"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the 'graphs' folder
    params = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".gml"):

            # Extract n and i from the file name (assuming the format is "graph_n_i.gml")
            base_name = os.path.splitext(file_name)[0]  # remove .gml extension
            _, n_str, i_str = base_name.split('_')
            n = int(n_str)
            i = int(i_str)

            output_file_path = os.path.join(output_folder, "white_{}_{}.csv.gz".format(n, i))
            params.append([output_file_path, n])

    # Parallel processing
    num_cores = 10  # Use physical cores
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(ts_generator, params)

if __name__ == "__main__":
    main()


