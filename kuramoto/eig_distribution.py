# Script to generate the eigenvalue distribution. We want a smooth curve for the pdf 
# so we use the Gaussian Kernel method instead of a histogram.

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None, fill=False, space=0.05):
    """
    Fit kernel to a series of obs, and derive the prob of obs. x is the array of values
        on which the fit KDE will be evaluated. It is the empirical PDF
    Args:
        obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
        bWidth (float): The bandwidth of the kernel. Default is .25
        kernel (str): The kernel to use. Valid kernels are [‘gaussian’|’tophat’|
            ’epanechnikov’|’exponential’|’linear’|’cosine’] Default is ‘gaussian’.
        x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
        fill (bool): if True, null entries are created to fill eventual gaps in the index
                    larger then 'space'. 
        space (float): distance between subsequent new entries. See 'fill' fo details.
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

    if fill:
        index = np.array(pdf.index)
        values = np.array(pdf.values)
        diff = index[1:] - index[:-1]

        space = 0.05
        positions = np.argwhere(diff > space).flatten()

        new_index = np.copy(index)
        new_values = np.copy(values)

        for pos in positions:
            c = index[pos]
            while c < index[pos + 1]:
                c += space
                new_index = np.append(new_index, c)
                new_values = np.append(new_values, 0)

        sorting = new_index.argsort()  
        new_index = new_index[sorting]
        new_values = new_values[sorting]
        pdf = pd.Series(new_values, index=new_index)

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

corr_path = "./kuramoto/data/corr_matrix" #set folder where correlation matrices are found
output_path = "./kuramoto/data/distributions" #set folder where time series are found
os.makedirs(output_path, exist_ok=True) # Create the output folder if it doesn't exist

file_list_10_100 = []
file_list_15_100 = []
file_list_25_100 = []
file_list_50_100 = []

file_list_10_200 = []
file_list_15_200 = []
file_list_25_200 = []
file_list_50_200 = []

file_list_10_500 = []
file_list_15_500 = []
file_list_25_500 = []
file_list_50_500 = []

file_list_10_1000 = []
file_list_15_1000 = []
file_list_25_1000 = []
file_list_50_1000 = []

for file_name in os.listdir(corr_path):
    # Extract n and i from the file name
    base_name = os.path.splitext(file_name)[0]  # remove .gml extension
    _, k_str, n_str, i_str = base_name.split('_')
    k = int(k_str[1:])
    n = int(n_str)
    i = int(i_str)

    if n == 100:
        if k == 10: file_list_10_100.append(file_name)
        elif k == 15: file_list_15_100.append(file_name)
        elif k == 25: file_list_25_100.append(file_name)
        elif k == 50: file_list_50_100.append(file_name)

    if n == 200:
        if k == 10: file_list_10_200.append(file_name)
        elif k == 15: file_list_15_200.append(file_name)
        elif k == 25: file_list_25_200.append(file_name)
        elif k == 50: file_list_50_200.append(file_name)

    if n == 500:
        if k == 10: file_list_10_500.append(file_name)
        elif k == 15: file_list_15_500.append(file_name)
        elif k == 25: file_list_25_500.append(file_name)
        elif k == 50: file_list_50_500.append(file_name)

    if n == 1000:
        if k == 10: file_list_10_1000.append(file_name)
        elif k == 15: file_list_15_1000.append(file_name)
        elif k == 25: file_list_25_1000.append(file_name)
        elif k == 50: file_list_50_1000.append(file_name)

zi, zf, dz = 0, 200, 0.01
z_range = np.arange(zi, zf, dz)

pdf_10_100 = np.zeros(int((zf - zi)/0.01))
pdf_15_100 = np.zeros(int((zf - zi)/0.01))
pdf_25_100 = np.zeros(int((zf - zi)/0.01))
pdf_50_100 = np.zeros(int((zf - zi)/0.01))

pdf_10_200 = np.zeros(int((zf - zi)/0.01))
pdf_15_200 = np.zeros(int((zf - zi)/0.01))
pdf_25_200 = np.zeros(int((zf - zi)/0.01))
pdf_50_200 = np.zeros(int((zf - zi)/0.01))

pdf_10_500 = np.zeros(int((zf - zi)/0.01))
pdf_15_500 = np.zeros(int((zf - zi)/0.01))
pdf_25_500 = np.zeros(int((zf - zi)/0.01))
pdf_50_500 = np.zeros(int((zf - zi)/0.01))

pdf_10_1000 = np.zeros(int((zf - zi)/0.01))
pdf_15_1000 = np.zeros(int((zf - zi)/0.01))
pdf_25_1000 = np.zeros(int((zf - zi)/0.01))
pdf_50_1000 = np.zeros(int((zf - zi)/0.01))

###### n = 100 ######
print('Interpolating n = 100...')
print('\tInterpolating k = 10...')
for file in file_list_10_100:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_10_100 += np.array([pdf_func(z) for z in z_range])/len(file_list_10_100)

pdf_10_100_emp = pd.Series(pdf_10_100, index=z_range)
pdf_10_100_emp.to_csv(os.path.join(output_path,'pdf_10_100.csv'), header=True)

print('\tInterpolating k = 15...')
for file in file_list_15_100:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_15_100 += np.array([pdf_func(z) for z in z_range])/len(file_list_15_100)

pdf_15_100_emp = pd.Series(pdf_15_100, index=z_range)
pdf_15_100_emp.to_csv(os.path.join(output_path,'pdf_15_100.csv'), header=True)

print('\tInterpolating k = 25...')
for file in file_list_25_100:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_25_100 += np.array([pdf_func(z) for z in z_range])/len(file_list_25_100)

pdf_25_100_emp = pd.Series(pdf_25_100, index=z_range)
pdf_25_100_emp.to_csv(os.path.join(output_path,'pdf_25_100.csv'), header=True)

print('\tInterpolating k = 50...')
for file in file_list_50_100:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_50_100 += np.array([pdf_func(z) for z in z_range])/len(file_list_50_100)

pdf_50_100_emp = pd.Series(pdf_50_100, index=z_range)
pdf_50_100_emp.to_csv(os.path.join(output_path,'pdf_50_100.csv'), header=True)

###### n = 200 ######
print('Interpolating n = 200...')
print('\tInterpolating k = 10...')
for file in file_list_10_200:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_10_200 += np.array([pdf_func(z) for z in z_range])/len(file_list_10_200)

pdf_10_200_emp = pd.Series(pdf_10_100, index=z_range)
pdf_10_200_emp.to_csv(os.path.join(output_path,'pdf_10_200.csv'), header=True)

print('\tInterpolating k = 15...')
for file in file_list_15_200:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_15_200 += np.array([pdf_func(z) for z in z_range])/len(file_list_15_200)

pdf_15_200_emp = pd.Series(pdf_15_100, index=z_range)
pdf_15_200_emp.to_csv(os.path.join(output_path,'pdf_15_200.csv'), header=True)

print('\tInterpolating k = 25...')
for file in file_list_25_200:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_25_200 += np.array([pdf_func(z) for z in z_range])/len(file_list_25_200)

pdf_25_200_emp = pd.Series(pdf_25_200, index=z_range)
pdf_25_200_emp.to_csv(os.path.join(output_path,'pdf_25_200.csv'), header=True)

print('\tInterpolating k = 50...')
for file in file_list_50_200:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_50_200 += np.array([pdf_func(z) for z in z_range])/len(file_list_50_200)

pdf_50_200_emp = pd.Series(pdf_50_200, index=z_range)
pdf_50_200_emp.to_csv(os.path.join(output_path,'pdf_50_200.csv'), header=True)

###### n = 500 ######
print('Interpolating n = 500...')
print('\tInterpolating k = 10...')
for file in file_list_10_500:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_10_500 += np.array([pdf_func(z) for z in z_range])/len(file_list_10_500)

pdf_10_500_emp = pd.Series(pdf_10_500, index=z_range)
pdf_10_500_emp.to_csv(os.path.join(output_path,'pdf_10_500.csv'), header=True)

print('\tInterpolating k = 15...')
for file in file_list_15_500:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_15_500 += np.array([pdf_func(z) for z in z_range])/len(file_list_15_500)

pdf_15_500_emp = pd.Series(pdf_15_500, index=z_range)
pdf_15_500_emp.to_csv(os.path.join(output_path,'pdf_15_500.csv'), header=True)

print('\tInterpolating k = 25...')
for file in file_list_25_500:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_25_500 += np.array([pdf_func(z) for z in z_range])/len(file_list_25_500)

pdf_25_500_emp = pd.Series(pdf_25_100, index=z_range)
pdf_25_500_emp.to_csv(os.path.join(output_path,'pdf_25_500.csv'), header=True)

print('\tInterpolating k = 50...')
for file in file_list_50_500:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_50_500 += np.array([pdf_func(z) for z in z_range])/len(file_list_50_500)

pdf_50_500_emp = pd.Series(pdf_50_500, index=z_range)
pdf_50_500_emp.to_csv(os.path.join(output_path,'pdf_50_500.csv'), header=True)

###### n = 1000 ######
print('Interpolating n = 1000...')
print('\tInterpolating k = 10...')
for file in file_list_10_1000:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_10_1000 += np.array([pdf_func(z) for z in z_range])/len(file_list_10_1000)

pdf_10_1000_emp = pd.Series(pdf_10_1000, index=z_range)
pdf_10_1000_emp.to_csv(os.path.join(output_path,'pdf_10_1000.csv'), header=True)

print('\tInterpolating k = 15...')
for file in file_list_15_1000:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_15_1000 += np.array([pdf_func(z) for z in z_range])/len(file_list_15_1000)

pdf_15_1000_emp = pd.Series(pdf_15_100, index=z_range)
pdf_15_1000_emp.to_csv(os.path.join(output_path,'pdf_15_1000.csv'), header=True)

print('\tInterpolating k = 25...')
for file in file_list_25_1000:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_25_1000 += np.array([pdf_func(z) for z in z_range])/len(file_list_25_1000)

pdf_25_1000_emp = pd.Series(pdf_25_1000, index=z_range)
pdf_25_1000_emp.to_csv(os.path.join(output_path,'pdf_25_1000.csv'), header=True)

print('\tInterpolating k = 50...')
for file in file_list_50_1000:
    x = np.loadtxt(os.path.join(corr_path, file), delimiter=",")
    eVal, eVec = getPCA(x)
    pdf = fitKDE(eVal, bWidth=0.01, fill=True)
    pdf_func = interp1d(pdf.index, pdf.values, kind='cubic', fill_value=0, bounds_error=False)
    pdf_50_1000 += np.array([pdf_func(z) for z in z_range])/len(file_list_50_1000)

pdf_50_1000_emp = pd.Series(pdf_50_1000, index=z_range)
pdf_50_1000_emp.to_csv(os.path.join(output_path,'pdf_50_1000.csv'), header=True)

print('Done!')