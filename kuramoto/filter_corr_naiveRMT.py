import os
import numpy as np
import pandas as pd

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

def apply_thresh2(C, file_name, output_path):
    p_list = [.1, .15, .2, .25] # filter parameter for naive filtering
    T = 2000
    # Extract n and i from the file name
    base_name = os.path.splitext(file_name)[0]  # remove .gml extension
    _, k_str, n_str, i_str = base_name.split('_')
    k = int(k_str[1:])
    n = int(n_str)
    i = int(i_str)

    os.makedirs(output_path, exist_ok=True)
    
    q = T/n
    print(f"Filtering {file_name}", end='\r')

    #---NAIVE THRESHOLDING OF CORRELATION MATRIX with RMT filtering---
    for p in p_list:
        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
                
        # naive filtering
        C_flat = np.unique(C_RMT.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C_RMT, vals), C_RMT, 0) 
        
        # save results
        output_path1 = os.path.join(output_path,"NaiveRMT") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"p{p}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,file_name)
        np.savetxt(file_path_out, C_naive, delimiter=",")