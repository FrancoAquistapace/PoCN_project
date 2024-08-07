'''
Here we will define all of the functions and tools that we need in order to reproduce the graphs
and results for the Song-Havlin-Makse paper.
'''
# ------ Necessary packages ---------
import numpy as np 
import networkx as nx
# -----------------------------------


# ---- Graph creation and growth ----
# Function to initialize a star graph
def star_graph(N):
    '''
    Params:
        N : int
            Number of nodes in the graph. Must be
            greater than 1.
    Output:
        Returns a star graph with N nodes as a 
        networkx Graph object.
    '''
    # Check that the graph has at least N=2
    if N < 2:
        print('Error: N must be greater than 1.')
        
    # Build the edges
    edge_list = []
    for i in range(N-1):
        edge_list.append([0,i+1])
        
    # Build graph and return it
    G = nx.from_edgelist(edge_list)
    return G


# Function for Mode I growth: hub-hub attraction
def attraction_growth_step(graph, m):
    '''
    Params:
        graph : networkx Graph
            Graph for which to apply the growth step.
        m : int
            Multiplicative constant for number of 
            new nodes. Must be greater than 0.
    Output:
        Returns a grown version of the given graph, using
        a hub-hub attraction mode. In this case, all of the
        previous edges of the netowrk are kept. Then, m*k 
        new nodes are added connected to each previous node
        with degree k. 
    '''
    # Gather old nodes and get the biggest node label
    old_nodes = list(graph.nodes)
    max_label = np.max(old_nodes)

    # Store current node number
    current_node = max_label + 1
    
    # Generate new nodes and egdes at the same time
    G_new = graph.copy()
    for i in old_nodes:
        # Get degree of old node
        k = graph.degree[i]
        # Add m*k new nodes
        for j in range(m*k):
            # Use current node label
            G_new.add_edge(i, current_node)
            # Update node label
            current_node += 1

    return G_new


# Function for Mode II growth: hub-hub repulsion
def repulsion_growth_step(graph, m, seed=42):
    '''
    Params:
        graph : networkx Graph
            Graph for which to apply the growth step.
        m : int
            Multiplicative constant for number of 
            new nodes. Must be greater than 0.
        seed : int (optional)
            Random seed used during growth process.
    Output:
        Returns a grown version of the given graph, using
        a hub-hub repulsion mode. In this case, all of the
        previous edges of the netowrk are erased. Then, m*k 
        new nodes are added connected to each previous node
        with degree k. Finally the old connections are replaced
        by connections between new neighbors of the old nodes, 
        respectively.
    '''
    # Initialize random generator with the seed
    np.random.seed(seed)
    
    # Gather old nodes and get the biggest node label
    old_nodes = list(graph.nodes)
    max_label = np.max(old_nodes)

    # Store current node number
    current_node = max_label + 1

    # Keep track of new nodes, and old nodes to which 
    # they are connected
    new_nodes = dict()

    # Generate new nodes and egdes at the same time
    G_new = graph.copy()
    for i in old_nodes:
        new_nodes[i] = []
        
        # Get degree of old node
        k = graph.degree[i]
        # Add m*k new nodes
        for j in range(m*k):
            # Use current node label
            G_new.add_edge(i, current_node)

            # Save new node info
            new_nodes[i].append(current_node)
            
            # Update node label
            current_node += 1

    # Now, we need to replace the old edges by 
    # non-hub edges 
    for edge in list(graph.edges):
        u, v = edge[0], edge[1]
        # Find new nodes connected to u and v
        u_new = np.random.choice(new_nodes[u])
        v_new = np.random.choice(new_nodes[v])

        # Replace old edge with new one
        G_new.remove_edge(u, v)
        G_new.add_edge(u_new, v_new)

    return G_new


# Function for a stochastic growth step
def stochastic_growth_step(graph, m, e, seed=42):
    '''
    Params:
        graph : networkx Graph
            Graph for which to apply the growth step.
        m : int
            Multiplicative constant for number of 
            new nodes. Must be greater than 0.
        e : float 
            Determines the fraction of Mode I growth
            (equal to e) and Mode II growth (given by
            1 - e). Must be between 0 and 1.
        seed : int (optional)
            Random seed used during growth process.
            Default is 42.
    Output:
        Returns a grown version of the given graph, using
        a sotchastic growth mode. In this case, the previous 
        edges of the netowrk are erased with probability 1 - e. 
        Then, m*k new nodes are added connected to each previous 
        node with degree k. Finally the deleted connections are 
        replaced by connections between new neighbors of the old 
        nodes, respectively.
    '''
    # Check that e is between 0 and 1
    if e < 0 or e > 1:
        print('Error: e must be between 0 and 1.')
        return 
        
    # Initialize random generator with the seed
    np.random.seed(seed)
    
    # Gather old nodes and get the biggest node label
    old_nodes = list(graph.nodes)
    max_label = np.max(old_nodes)

    # Store current node number
    current_node = max_label + 1

    # Keep track of new nodes, and old nodes to which 
    # they are connected
    new_nodes = dict()

    # Generate new nodes and egdes at the same time
    G_new = graph.copy()
    for i in old_nodes:
        new_nodes[i] = []
        
        # Get degree of old node
        k = graph.degree[i]
        # Add m*k new nodes
        for j in range(m*k):
            # Use current node label
            G_new.add_edge(i, current_node)

            # Save new node info
            new_nodes[i].append(current_node)
            
            # Update node label
            current_node += 1

    # Now, we need to replace the old edges by 
    # non-hub edges, with probability p = 1 - e
    for edge in list(graph.edges):
        # Check if old edge is to be replaced or not
        if np.random.uniform() < (1-e):
            u, v = edge[0], edge[1]
            # Find new nodes connected to u and v
            u_new = np.random.choice(new_nodes[u])
            v_new = np.random.choice(new_nodes[v])
    
            # Replace old edge with new one
            G_new.remove_edge(u, v)
            G_new.add_edge(u_new, v_new)

    return G_new


# Function to generate a minimal-model graph
# according to specifications
def minimal_model_graph(init_N, n_iters, m, e, seed=42):
    '''
    Params:
        init_N : int 
            Number of nodes for the initial star graph.
            Must be greater than 1.
        n_iters : int
            Number of growth iterations to perform. Must
            be greater than 0.
        m : int
            Multiplicative constant for number of new 
            nodes. Must be greater than 0.
        e : float 
            Determines the fraction of Mode I growth
            (equal to e) and Mode II growth (given by
            1 - e). Must be between 0 and 1.
        seed : int (optional)
            Random seed used during growth process. 
            Default is 42.
    Output:
        Returns a graph grown according to the minimal model
        defined in the work by Song-Havlin-Makse. The network
        is initialized as a star graph with init_N nodes. Then,
        n_iters growth iterations are performed, each of them
        with a stochastic combination of Modes I and II. This
        is regulated by the Mode I probability e. Additionally,
        a dictionary with relevant data recorded at each growth
        iteration is returned.
    '''
    # Check for correct parameters
    if init_N <= 1:
        print('Error: init_N must be greater than 1.')
        return
    if n_iters <= 0:
        print('Error: n_iters must be greater than 0.')
        return

    # Initialize graph and data
    G = star_graph(init_N)
    data = {'N': [G.number_of_nodes()],
            'L': [nx.diameter(G)]}

    # Perform growth process
    for i in range(n_iters):
        # Use stochastic growth
        if i == 0:
            G = stochastic_growth_step(G, m=m, e=e, seed=seed)
        else:
            G = stochastic_growth_step(G, m=m, e=e, seed=seed)
        # Save iteration data
        data['N'].append(G.number_of_nodes())
        data['L'].append(nx.diameter(G))

    # Return graph and data
    return G, data
# -----------------------------------



# -------- Graph Analysis -----------
# Function to get the (k_1, k_2) pairs of a graph
def degree_pairs(graph):
    '''
    Params:
        graph : networkx Graph
            Graph for which to apply the growth step.
    Output:
        Returns a list containing the k_1, k_2 degree 
        pairs of the given graph.
    '''
    # Init empty results
    res = []
    
    # Iterate over all edges of the graph
    edges = list(graph.edges)
    for edge in edges:
        u, v = edge[0], edge[1]
        # Get degrees k_u and k_v and save to results
        k_u = graph.degree[u]
        k_v = graph.degree[v]
        res.append([k_u, k_v])

    return res


# Function to get P(k_1, k_2) in a frequentist approach
def P_joint_deg(measures):
    '''
    Params:
        measures : list or array
            Iterable containing the measurements of 
            (k_1, k_2) pairs.
    Output:
        Returns a matrix A of size (k_max, k_max) where
        k_max is the maximum degree identified in the
        measures. Each element A_ij of the matrix is the
        frequency of occurrence of the pair k_1=i+1,
        k_2=j+1.
    '''
    # Turn measures into numpy array
    measures = np.array(measures)

    # Init P joint as count matrix
    k_max = np.max(measures)
    A = np.zeros(shape=(k_max, k_max))
    # Iterate only over unique values
    unique_k_1 = list(np.unique(measures))
    for k_1 in unique_k_1:
        # Get pairs with k_1
        k_1_mask = (measures[:,0] == k_1)
        k_1_pairs = measures[k_1_mask,:]

        # Iterate over unique k_2 values 
        k_2_vals = np.unique(k_1_pairs[:,1])
        for k_2 in k_2_vals:
            # Get counts and save them to matrix
            k_2_counts = np.sum(k_1_pairs[:,1] == k_2)
            A[k_1 - 1, k_2 - 1] *= 0
            A[k_1 - 1, k_2 - 1] += k_2_counts
            A[k_2 - 1, k_1 - 1] *= 0
            A[k_2 - 1, k_1 - 1] += k_2_counts
        
    # Normalize by totality of measures to get frequency
    A = A / np.sum(A)
    return A
# -----------------------------------