'''
Here we will define all of the functions and tools that we need in order to reproduce the graphs
and results for the papers about sandpile dynamics on complex networks.
'''
# ------ Necessary packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
# -----------------------------------


# -------- Graph creation -----------
# Function to create a scale-free network
# using the static model
def scale_free_graph(N, gamma, m, block_size=100,
                    seed=None):
    '''
    Params:
        N : int
            Number of nodes in the network.
        gamma : float
            Degree exponent for the degree
            power-law distribution. If the
            value "inf" is given, then the
            resulting graph has a uniform
            degree distribution.
        m : int
            Desired half average degree.
        block_size : int (optional)
            Amount of edges attempted before 
            recalculating the average degree 
            of the network. Default is 100.
        seed : int (optional)
            Seed for the numpy random 
            generator, for consistency. This
            feature is not used by default.
    Output:
        Returns a networkx undirected graph,
        where the degrees are distributed as 
        a power law of exponent gamma. The 
        graph has N nodes, and the average
        degree is 2*m.
    '''
    # Set seed if given
    if seed != None:
        np.random.seed(seed)

    # Get control parameter
    if gamma == 'inf':
        alpha = 0
    else:
        alpha = 1 / (gamma - 1)

    # Initialize the graph and all nodes
    all_nodes = np.arange(N)+1
    g = nx.Graph()

    # Define relative weights
    w = np.power(all_nodes, -alpha)
    w = w / np.sum(w)

    # Add all nodes to the graph
    g.add_nodes_from(all_nodes)

    # Initialize average degree
    avg_k = 0

    # Initialize counters
    count = 0
    node_count = 0

    # Keep adding links until reaching k_avg=2*m
    while avg_k < 2*m:
        # Every block_size steps, let's redraw random nodes
        if count % block_size == 0:
            nodes = np.random.choice(all_nodes, 
                                    size=(block_size,2),
                                    replace=True, p=w)
            # Reset node counter
            node_count = 0

            # Recalculate average degree
            avg_k = average_degree(g)

        # Attempt new link
        u,v = nodes[node_count]
        if not g.has_edge(u,v):
            g.add_edge(u,v)

        # Update counters
        count += 1
        node_count += 1

    return g


def get_lcc(graph):
    '''
    Params:
        graph : networkx Graph
            A network with possibly many connected components.
    Output:
        Returns the largest connected component of the given 
        graph, as a new netowrkx Graph object.
    '''
    # Get LCC nodes
    largest_cc = max(nx.connected_components(graph), key=len)
    
    # Generate subgraph
    lcc = graph.subgraph(largest_cc).copy()
    return lcc

# -----------------------------------


# ---------- Simulation -------------


# -----------------------------------

# ----------- Analysis --------------
def average_degree(graph):
    '''
    Params:
        graph : networkx Graph
            Graph to be analysed.
    Output:
        Returns the average degree
        of the given graph.
    '''
    deg = list(dict(graph.degree).values())
    return np.mean(deg)

# -----------------------------------



# --------- Miscellaneous -----------
# Function to save a graph into a csv file
# as (node_from, node_to, weight)
def save_graph(graph, path):
    '''
    Params:
        graph : networkx Graph
            Graph to save.
        path : str
            Path of the file to save the graph into.
    Output:
        Saves the given graph into a csv file with
        the following row format:
            node_from, node_to, weight
        Where weight is simply set to 1 for each edge,
        but is kept for consistency.
    '''
    # Get edges
    g_arr = np.array(graph.edges)

    # Check if the network has a single node
    if g_arr.shape[0] == 0:
        g_dict = {'node_from':[0], 'node_to':[0],
                'weight':[0]}

    else:
        # Save edges into a dictionary
        g_dict = {'node_from':g_arr[:,0], 'node_to':g_arr[:,1], 
                'weight':[1 for i in range(g_arr.shape[0])]}

    # Turn dictionary into a pandas DataFrame
    g_df = pd.DataFrame(g_dict)
    
    g_df.to_csv(path, index=False) 


# -----------------------------------