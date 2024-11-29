import os
import numpy as np
np.random.seed(1234)

def standardize_matrix(matrix):
    """
    Standardize each column of the given matrix to have mean 0 and variance 1.

    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized_matrix = (matrix - mean) / std
    
    return standardized_matrix

folder_name = "time_series"
folder_2 = "corr_matrix"
folder_3 = "eigenvalue_distr"
# Create the output folders if it doesn't exist
os.makedirs(folder_2, exist_ok=True)
os.makedirs(folder_3, exist_ok=True)

for file_name in os.listdir(folder_name):
    if file_name.startswith("kuramoto"):
        print("Running {}...".format(file_name), end='\r')
        x_vals_loaded = np.loadtxt(os.path.join(folder_name,file_name), delimiter=",")
        standized_vals = standardize_matrix(x_vals_loaded)
        correlation_matrix = np.corrcoef(standized_vals, rowvar=False)
        eigenvalues = np.linalg.eigvals(correlation_matrix)

        # Save the correlations
        file_path_out = os.path.join(folder_2,file_name)
        np.savetxt(file_path_out, correlation_matrix, delimiter=",")
        # Save the eigenvalues
        file_path_out = os.path.join(folder_3,file_name)
        np.savetxt(file_path_out, eigenvalues, delimiter=",")

        print("{} ...done!".format(file_name))
