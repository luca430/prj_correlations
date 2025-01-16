# Script to generate the eigenvalue distributions and store them in desired folder.
# We want a smooth curve for the pdf so we use the Gaussian Kernel method instead of a histogram.

import os
import sys
import gzip
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    """
    Fit kernel to a series of obs, and derive the prob of obs. x is the array of values
        on which the fit KDE will be evaluated. It is the empirical PDF
    Args:
        obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
        bWidth (float): The bandwidth of the kernel. Default is .25
        kernel (str): The kernel to use. Valid kernels are [‘gaussian’|’tophat’|
            ’epanechnikov’|’exponential’|’linear’|’cosine’] Default is ‘gaussian’.
        x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
    Returns:
        pd.Series: Empirical PDF
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def getPCA(matrix):
    """
    Gets the Eigenvalues and Eigenvector values from a Hermitian Matrix
    Args:
        matrix pd.DataFrame: Correlation matrix
    Returns:
         (tuple): tuple containing:
            np.ndarray: Eigenvalues of correlation matrix
            np.ndarray: Eigenvectors of correlation matrix
    """
    # Get eVal,eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    return eVal, eVec

def eig_distribution(file_list,input_folder):

    zi, zf, dz = 0, 3, 0.01 # we chose `zf=3` because we know that the max eigenvalue possible for us is smaller then 3
    z_range = np.arange(zi, zf, dz)
    pdf_array = np.zeros(int((zf - zi)/0.01))

    for file in file_list:
        with gzip.open(os.path.join(input_folder,file), "rt") as f:
                x = np.loadtxt(f, delimiter=",")
        eVal, eVec = getPCA(x)
        pdf = fitKDE(eVal, bWidth=0.01)
        pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
        pdf_array += np.array([pdf_func(z) for z in z_range])/len(file_list)

    return pd.Series(pdf_array, index=z_range)
     

def main():

    input_folder = "/mnt/corr_matrices/white_noise"
    output_folder = "/mnt/eig_distributions/white_noise"
    os.makedirs(output_folder, exist_ok=True) # Create the output folder if it doesn't exist

    file_list_100 = []
    file_list_200 = []
    file_list_500 = []
    file_list_1000 = []

    print("Extracting files...", end="\r")
    for file_name in os.listdir(input_folder):
        # Extract n and i from the file name
        base_name = os.path.splitext(file_name)[0][:-4]  # remove .csv.gz extension
        _, n_str, i_str = base_name.split('_')
        n = int(n_str)
        i = int(i_str)
        if n == 100: file_list_100.append(file_name)
        elif n == 200: file_list_200.append(file_name)
        elif n == 500: file_list_500.append(file_name)
        elif n == 1000: file_list_1000.append(file_name)

    print('Interpolating n = 100...', end="\r")
    pdf_100_emp = eig_distribution(file_list_100, input_folder)
    pdf_100_emp.to_csv(os.path.join(output_folder,'pdf_100.csv'), header=True)
        
    print('Interpolating n = 200...', end="\r")
    pdf_200_emp = eig_distribution(file_list_200, input_folder)
    pdf_200_emp.to_csv(os.path.join(output_folder,'pdf_200.csv'), header=True)
        
    print('Interpolating n = 500...', end="\r")
    pdf_500_emp = eig_distribution(file_list_500, input_folder)
    pdf_500_emp.to_csv(os.path.join(output_folder,'pdf_500.csv'), header=True)
        
    print('Interpolating n = 1000...', end="\r")
    pdf_1000_emp = eig_distribution(file_list_1000, input_folder)
    pdf_1000_emp.to_csv(os.path.join(output_folder,'pdf_1000.csv'), header=True)
        
    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line by overwriting with spaces
    print('Done!')

if __name__ == "__main__":
    main()