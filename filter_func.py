# Script that contains functions to filter a correlation matrix according in four ways:
# (1) Naive; (2) RMT + Naive; (3) Fisher; (4) RMT + Fisher

import os
import gzip
import numpy as np
from scipy.sparse import csr_matrix, save_npz

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

def apply_thresh(input_file, output_path):
    p_list = [.1, .15, .2, .25] # filter parameter for naive filtering
    tau_list = [1., 1.5, 2., 2.5]
    T = 2000
    
    # Extract n and i from the file name
    base_name = os.path.splitext(input_file)[0][:-4]  # remove .csv.gz extension
    base_name = base_name.split("/")[-1]
    _, n_str, _ = base_name.split('_')
    n = int(n_str)
    
    q = T/n
    with gzip.open(input_file, "rt") as f:
        C = np.loadtxt(f, delimiter=",")

    #---FISHER THRESHOLDING WITH AND WITHOUT RMT FILTERING---
    for tau in tau_list:
        
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_Fish = np.copy(C)
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)

        C_RMT[C_RMT < C_tau] = 0    # RMT filtering + Fisher
        C_Fish[C_Fish < C_tau] = 0  # Fisher filtering

        # save results
        output_path1 = os.path.join(output_path,"FisherRMT") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"tau{tau}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,base_name)
        save_npz(file_path_out, csr_matrix(C_RMT))

        output_path1 = os.path.join(output_path,"Fisher") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"tau{tau}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,base_name)
        save_npz(file_path_out, csr_matrix(C_Fish))

    #---NAIVE THRESHOLDING OF CORRELATION MATRIX with RMT filtering---
    for p in p_list:
        
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
                
        # RMT filtering + Naive
        C_RMT_flat = np.unique(C_RMT.flatten())
        vals = C_RMT_flat[:int(p*len(C_RMT_flat))]
        C_RMT_naive = np.where(np.isin(C_RMT, vals), C_RMT, 0)

        # Naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0) 
        
        # save results
        output_path1 = os.path.join(output_path,"NaiveRMT") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"p{p}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,base_name)
        save_npz(file_path_out, csr_matrix(C_RMT_naive))
        
        output_path1 = os.path.join(output_path,"Naive") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"p{p}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,base_name)
        save_npz(file_path_out, csr_matrix(C_naive))