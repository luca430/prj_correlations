import os
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

corr_path = "corr_matrix" #set folder where correlation matrices are found
output_path = "distributions" #set folder where time series are found
os.makedirs(output_path, exist_ok=True) # Create the output folder if it doesn't exist

file_list_100 = []
file_list_200 = []
file_list_500 = []
file_list_1000 = []

for file_name in os.listdir(corr_path):
    # Extract n and i from the file name
    base_name = os.path.splitext(file_name)[0]  # remove .gml extension
    _, n_str, i_str = base_name.split('_')
    n = int(n_str)
    i = int(i_str)
    if n == 100: file_list_100.append(file_name)
    elif n == 200: file_list_200.append(file_name)
    elif n == 500: file_list_500.append(file_name)
    elif n == 1000: file_list_1000.append(file_name)

zi, zf, dz = 0, 3, 0.01
z_range = np.arange(zi, zf, dz)
pdf_100 = np.zeros(int((zf - zi)/0.01))
pdf_200 = np.zeros(int((zf - zi)/0.01))
pdf_500 = np.zeros(int((zf - zi)/0.01))
pdf_1000 = np.zeros(int((zf - zi)/0.01))

print('Interpolating n = 100...')
for file in file_list_100:
    x = np.loadtxt(os.path.join(corr_path,file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_100 += np.array([pdf_func(z) for z in z_range])/len(file_list_100)

pdf_100_emp = pd.Series(pdf_100, index=z_range)
pdf_100_emp.to_csv(os.path.join(output_path,'pdf_100.csv'), header=True)
    
print('Interpolating n = 200...')
for file in file_list_200:
    x = np.loadtxt(os.path.join(corr_path,file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_200 += np.array([pdf_func(z) for z in z_range])/len(file_list_200)

pdf_200_emp = pd.Series(pdf_200, index=z_range)
pdf_200_emp.to_csv(os.path.join(output_path,'pdf_200.csv'), header=True)
    
print('Interpolating n = 500...')
for file in file_list_500:
    x = np.loadtxt(os.path.join(corr_path,file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_500 += np.array([pdf_func(z) for z in z_range])/len(file_list_500)

pdf_500_emp = pd.Series(pdf_500, index=z_range)
pdf_500_emp.to_csv(os.path.join(output_path,'pdf_500.csv'), header=True)
    
print('Interpolating n = 1000...')
for file in file_list_1000:
    x = np.loadtxt(os.path.join(corr_path,file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_1000 += np.array([pdf_func(z) for z in z_range])/len(file_list_1000)

pdf_1000_emp = pd.Series(pdf_1000, index=z_range)
pdf_1000_emp.to_csv(os.path.join(output_path,'pdf_1000.csv'), header=True)
    
print('Done!')