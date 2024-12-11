import os
import numpy as np
import pandas as pd
import networkx as nx

naive_path = "filtered_naive" #set folder where naively filtered correlation matrices are found
RMT_path = "filtered_RMT" #set folder where RMT filtered matrices are found
graphs_folder = os.path.join('..', 'graphs2')

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

for file_name in os.listdir(naive_path):
    if file_name.endswith(".csv"):
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

print('Comparing n = 100...')
print('\Comparing k = 10...')
if len(file_list_10_100) != 0:
    for file in file_list_10_100:

        C_naive = np.loadtxt(os.path.join(naive_path, file), delimiter=",")
        C_RMT = np.loadtxt(os.path.join(RMT_path, file), delimiter=",")

        A_naive = np.zeros(np.shape(C_naive))
        A_naive[np.where(np.abs(C_naive) > 0.)] = 1.
        G_naive = nx.from_numpy_array(A_naive, edge_attr = None)
        A_RMT = np.zeros(np.shape(C_RMT))
        A_RMT[np.where(np.abs(C_RMT) > 0.)] = 1.
        G_RMT = nx.from_numpy_array(A_RMT, edge_attr = None)

        base_name = os.path.splitext(file)[0]  # remove .gml extension
        _, k_str, n_str, i_str = base_name.split('_')
        n = int(n_str)
        i = int(i_str)
        # Load the original graph
        file_path_in = os.path.join(graphs_folder, "graph_{}_{}.gml".format(n,i))
        G = nx.read_gml(file_path_in)
        #A_original = nx.adjacency_matrix(G).toarray()

        #Global clustering
        clus_orig = nx.average_clustering(G)
        clus_naive = nx.average_clustering(G_naive)
        clus_RMT = nx.average_clustering(G_RMT)


        

