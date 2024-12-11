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

def apply_thresh(C, file_name, output_path):
    p_list = [.1, .15, .2, .25] # filter parameter for naive filtering
    tau_list = [1., 1.5, 2., 2.5]
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

    #---FISHER THRESHOLDING WITH AND WITHOUT RMT FILTERING---
    for tau in tau_list:
        # RMT filtering
        eVal, eVec = getPCA(C)
        eMax = (1 + (1./q)**.5)**2
        C_Fish = np.copy(C)
        C_RMT = np.copy(C)
        for i,eig in enumerate(eVal):
            if eig < eMax:
                v = np.reshape(eVec[:,i],(-1,1))
                C_RMT -= eig*np.dot(v,v.T)
        C_tau = (np.exp(2*tau/np.sqrt(T - 3)) - 1)/(np.exp(2*tau/np.sqrt(T - 3)) + 1)
        C_RMT[C_RMT < C_tau] = 0
        C_Fish[C_Fish < C_tau] = 0
        # save results
        output_path1 = os.path.join(output_path,"FisherRMT") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"tau{tau}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,file_name)
        np.savetxt(file_path_out, C_RMT, delimiter=",")

        output_path1 = os.path.join(output_path,"Fisher") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"tau{tau}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,file_name)
        np.savetxt(file_path_out, C_Fish, delimiter=",")

    #---NAIVE THRESHOLDING OF CORRELATION MATRIX---
    for p in p_list:
        # naive filtering
        C_flat = np.unique(C.flatten())
        vals = C_flat[:int(p*len(C_flat))]
        C_naive = np.where(np.isin(C, vals), C, 0) 
        
        # save results
        output_path1 = os.path.join(output_path,"Naive") #specify thresholding method
        os.makedirs(output_path1, exist_ok=True)
        output_path2 = os.path.join(output_path1,f"p{p}") #specify threshold
        os.makedirs(output_path2, exist_ok=True)
        file_path_out = os.path.join(output_path2,file_name)
        np.savetxt(file_path_out, C_naive, delimiter=",")