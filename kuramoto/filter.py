import os
import numpy as np
import pandas as pd

corr_path = "corr_matrix" #set folder where correlation matrices are found
output_path1 = "filtered_RMT" #set folder where to save RMT results
output_path2 = "filtered_naive" #set folder where to save naive results

os.makedirs(output_path1, exist_ok=True)
os.makedirs(output_path2, exist_ok=True)

file_list_10_100 = []
file_list_15_100 = []
file_list_25_100 = []
file_list_50_100 = []

file_list_10_200 = []
file_list_15_200 = []
file_list_25_200 = []
file_list_50_200 = []

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

p = 0.1 # filter parameter for naive filtering
tau = 2 # filter parameter for RMT filtering
T = 2000 # length of the time series

###### n = 100 ######
print('Filtering n = 100...')
q = T/100
print('\tFiltering k = 10...')
if len(file_list_10_100) != 0:
    for file in file_list_10_100:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)        

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 15...')
if len(file_list_15_100) != 0:
    for file in file_list_15_100:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 25...')
if len(file_list_25_100) != 0:
    for file in file_list_25_100:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 50...')
if len(file_list_50_100) != 0:
    for file in file_list_50_100:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

###### n = 200 ######
print('Filtering n = 200...')
q = T/200
print('\tFiltering k = 10...')
if len(file_list_10_200) != 0:
    for file in file_list_10_200:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 15...')
if len(file_list_15_200) != 0:
    for file in file_list_15_200:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 25...')
if len(file_list_25_200) != 0:
    for file in file_list_25_200:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print('\tFiltering k = 50...')
if len(file_list_50_200) != 0:
    for file in file_list_50_200:

        C = np.loadtxt(os.path.join(corr_path, file), delimiter=",")

        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0

        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0)

        # save results
        file_path_out = os.path.join(output_path1,file)
        np.savetxt(file_path_out, C_RMT, delimiter=",")
        file_path_out = os.path.join(output_path2,file)
        np.savetxt(file_path_out, C_naive, delimiter=",")

print("Done!")
