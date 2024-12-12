# Module which contains many functions for computing global measures

import os
import numpy as np
import graph_tool.all as gt
from scipy.stats import linregress

def build_dict():
    N_dict = {}
    for i in range(25):
        N_dict[f"{i+1}"] = {}
        for k in [0,10,15,25,50]:
            N_dict[f"{i+1}"][f"k{k}"] = {}
            N_dict[f"{i+1}"][f"k{k}"]["Fisher"] = {}
            N_dict[f"{i+1}"][f"k{k}"]["FisherRMT"] = {}
            N_dict[f"{i+1}"][f"k{k}"]["Naive"] = {}
            N_dict[f"{i+1}"][f"k{k}"]["NaiveRMT"] = {}
            for tau in ["tau1.0", "tau1.5", "tau2.0", "tau2.5"]:
                N_dict[f"{i+1}"][f"k{k}"]["Fisher"][tau] = {}
                N_dict[f"{i+1}"][f"k{k}"]["FisherRMT"][tau] = {}
            for p in ["p0.1", "p0.15", "p0.2", "p0.25"]:
                N_dict[f"{i+1}"][f"k{k}"]["Naive"][p] = {}
                N_dict[f"{i+1}"][f"k{k}"]["NaiveRMT"][p] = {}

    return N_dict

def extract_file_information(subdir, file):
    base_name = os.path.splitext(file)[0]
    _, k_str, n_str, i_str = base_name.split('_')
    thresh = subdir.split('/')[-1]
    method = subdir.split('/')[-2]
    
    return(i_str, k_str, method, thresh)

def explore_dict(global_dict, keys):
    # Function to easily access global variables.
    # 'keys' is an array of strings containing ORDERED keys from bottom to top.
    # Example: keys = ['Neigh_degree','k0'] is ok but the opposite is not.
    numbers = list(global_dict.keys())
    ks = list(global_dict['1'].keys())
    methods = list(global_dict['1']['k0'].keys())
    taus = list(global_dict['1']['k0']['Fisher'].keys())
    ps = list(global_dict['1']['k0']['Naive'].keys())
    measures = list(global_dict['1']['k0']['Fisher']['tau1.0'].keys())

    out_dict = global_dict
    for key in keys:
        if key in measures:
            out_dict = {n:  {k: {m: {t: out_dict[n][k][m][t][key] for t in taus} for m in ['Fisher','FisherRMT']} | {m: {p: out_dict[n][k][m][p][key] for p in ps} for m in ['Naive','NaiveRMT']} for k in ks} for n in numbers}
        elif key in ps:
            out_dict = {n:  {k: {m: out_dict[n][k][m][key] for m in ['Naive','NaiveRMT']} for k in ks} for n in numbers}
        elif key in taus:
            out_dict = {n:  {k: {m: out_dict[n][k][m][key] for m in ['Fisher','FisherRMT']} for k in ks} for n in numbers}
        elif key in methods:
            out_dict = {n:  {k: out_dict[n][k][key] for k in ks} for n in numbers}
        elif key in ks:
            out_dict = {n: out_dict[n][key] for n in numbers}
        elif key in numbers:
            out_dict = out_dict[key]
        else:
            return 'Wrong key input.'
    return out_dict

def mat2edgelist(csv_matrix):
    # function to convert a matrix saved in csv format into an edge list
    A = np.loadtxt(csv_matrix,delimiter=",")
    for i in range(len(A)):
        A[i,i] = 0
    edge_list = []
    for source in range(len(A)):
        for target in np.nonzero(A[source,:])[0]:
            edge_list.append((source,target))
    return edge_list

def average_neighbor_degree(g):
    nodes = g.get_vertices()
    degrees = np.reshape(g.get_out_degrees(nodes),(-1,1))
    A = gt.adjacency(g).todense()
    knn = np.dot(A,degrees)
    for i in range(len(degrees)):
        if degrees[i] != 0:
            knn[i] /= degrees[i]
    knn = np.asarray(knn).reshape(-1)
    return {n: k for n, k in zip(nodes,knn)}

def knn_scaling_exponent(g):

    nodes = g.get_vertices()
    degree_list = g.get_out_degrees(nodes)
    knns = average_neighbor_degree(g)
    knns_vals = np.array(list(knns.values()))

    avg_knns = []
    for k in range(1,int(np.max(degree_list) + 1)):
        N = len(nodes[degree_list == k])
        arr = knns_vals[degree_list == k]
        if N == 0: avg_knns.append(0)
        else: avg_knns.append(np.sum(arr)/N)
    avg_knns = np.array(avg_knns)
    
    # Log-transform the data
    y = avg_knns[avg_knns > 0]
    x = np.array([i for i in range(1,int(np.max(degree_list) + 1))])
    x = x[avg_knns > 0]
    x = np.log(x)
    y = np.log(y)
    if len(y) <= 1: return (0,0)
    
    # Perform linear regression to fit log(knn) = alpha * log(k) + C
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    return (slope, std_err)

def avg_shortest_path(g):
    nodes = g.get_vertices()
    n = len(nodes)
    distances = np.array(gt.shortest_distance(g))
    if np.any(distances >= n): return np.nan
    return np.sum(distances)/(n*(n - 1))
        

def compute_global_variables(obj, glob_dict, load=False):

    # if not load:
    #     g = gt.Graph()
    #     g.add_edge_list(obj)
    # else:
    #     g = gt.load_graph(obj, fmt='gml')

    g = gt.Graph()
    g.add_edge_list(obj)
    
    #Mean degree
    degree_sequence = g.get_out_degrees(g.get_vertices())
    glob_dict['Mean_degree'] = np.mean(degree_sequence)
    
    #Molloy-Reed coefficient
    seq2 = np.array(degree_sequence)**2
    glob_dict['MR_coefficient'] = np.mean(seq2)/np.mean(degree_sequence)
    
    #Avg. neigh. degree
    n_deg = average_neighbor_degree(g)
    glob_dict['Neigh_degree'] = np.mean(list(n_deg.values()))

    #Mixing exponent
    expo = knn_scaling_exponent(g)
    glob_dict['Mixing_exponent'] = expo
    
    #Global clustering
    clust = gt.global_clustering(g)
    glob_dict['Global_clustering'] = clust

    #Assortativity
    assort = gt.assortativity(g,"in")
    glob_dict['Assortativity'] = assort

    #Modularity via minimum description length
    mdl_state = gt.minimize_blockmodel_dl(g)
    b = mdl_state.get_blocks()
    mod = gt.modularity(g,b)
    glob_dict['Modularity'] = mod

    #Avg. path length
    glob_dict['Avg_path_length'] = avg_shortest_path(g)
    
    return glob_dict