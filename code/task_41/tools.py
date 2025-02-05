'''
Here we will define all of the functions and tools that we need in order to read and
analyse the networks generated from the citylines dataset.
'''
# ------ Necessary packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
import time
# -----------------------------------


# ----------- Analysis --------------
# Function to get degree frequencies from data
def get_deg_frequencies(deg_list):
    '''
    Params:
        deg_list : numpy array
            Array containing node degree observations.
    Output:
        Returns a tuple (k_vals, k_freqs) where k_vals
        is an array with the unique degree values found
        in the given data, and k_freqs an array with the
        corresponding frequency of the unique values.
    '''
    # Get unique values
    k_vals = np.unique(deg_list)
    # Get corresponding frequencies
    k_freqs = np.array(
        [np.sum(deg_list == k)/len(deg_list) for k in k_vals])
    return k_vals, k_freqs
# -----------------------------------



# --------- Miscellaneous -----------

# Function to read nodes and edges from a city into a graph
def read_city_network(path):
    '''
    Params:
        path : str
            Path of the files "_nodes.csv" and "_edges.csv"
            with the data of a given city.
    Output:
        Returns a networkx Graph built with the nodes and 
        edges extracted from the data of the given city.
    '''
    # First read nodes and edges as DataFrames
    nodes = pd.read_csv(path + '_nodes.csv')
    edges = pd.read_csv(path + '_edges.csv')

    # Init empty network
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(nodes['nodeID'])

    # Add edges
    for i in range(edges.shape[0]):   
        e = edges.iloc[i]
        u,v = e['nodeID_from'], e['nodeID_to']
        G.add_edge(u,v)

    return G


# -----------------------------------