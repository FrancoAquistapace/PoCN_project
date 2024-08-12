'''
Here we will define all of the functions and tools that we need in order to reproduce the graphs
and results for the Song-Havlin-Makse paper.
'''
# ------ Necessary packages ---------
import numpy as np 
import networkx as nx
from scipy.interpolate import RectBivariateSpline
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


# Function to randomize a graph via edge swapping
def randomize_graph(graph, n_iters):
    '''
    Params:
        graph : networkx Graph
            Graph for which to apply the randomization.
        n_iters : int
            Number of iterations to use for swapping
            the graph edges.
    Output:
        Returns a randomized version of the graph but
        conserving the original degree distribution. The
        randomization is performed by picking at random
        two edges (A, B) and (C, D), and swapping the
        ends so that the new edges are (A, D) and (C, B).
        If these edges already exist, the operation is 
        cancelled and a new set of pairs is chosen. This
        is done for n_iters.
    '''
    # Initialize new graph
    G_new = graph.copy()
    swap_status = []
    # Iterate and perform the swapping
    for i in range(n_iters):
        # Get current edges
        G_edges = list(G_new.edges)

        # Try until successful 
        swap_done = False
        while not swap_done:
            # Pick two random edges
            r_idx = np.random.randint(low=0, high=len(G_edges), size=2)
            AB, CD = G_edges[r_idx[0]], G_edges[r_idx[1]]
            # Build new edges
            AD, BC = (AB[0], CD[1]), (CD[0], AB[1])

            # Check if new edges already exist
            if (AD in G_edges) or (BC in G_edges):
                pass
    
            else:
                # Perform removal of old edges
                G_new.remove_edge(AB[0], AB[1])
                G_new.remove_edge(CD[0], CD[1])

                # And addition of new edges
                G_new.add_edge(AD[0], AD[1])
                G_new.add_edge(BC[0], BC[1])
                
                swap_done = True

    return G_new


# Function to normalize a graph via box-covering,
# as done by Song-Havlin-Makse
def normalization_step(graph, l_B):
    '''
    Params:
        graph : networkx Graph
            Graph to normalize. 
        l_B : int
            Characteristic length to use for the
            normalization via box-covering.
    Output:
        Returns a normalized version of the given
        graph. The normalization is done via the 
        box-covering algorithm used by Song-Havlin-
        Makse.
    '''
    # Get list of current unexplored nodes
    current_nodes = list(graph.nodes)
    
    # Get all shortest paths with length <= l_B
    s_paths = dict(nx.all_pairs_shortest_path_length(graph, 
                                                     cutoff=l_B))
    
    # Initialize empty boxes
    boxes = []
    current_box = []
    
    # Iterate until finished
    finished = False
    in_box = False # Check variable
    box_node = -1
    i = 0
    while not finished:
        # If we are not working inside a box: 
        if not in_box:
            # Pick a random node
            box_node = np.random.choice(current_nodes)
            # Init new box
            current_box = [box_node]
            # Delete picked node from available list
            current_nodes.remove(box_node)
            in_box = True
            
        else:
            # Check if there are neighbors within l_B - 1
            # of every other node in the box
            new_nodes = current_nodes.copy()
            for n_new in new_nodes:
                add_node = True # Add by default
                for n_box in current_box:
                    # If node distance is >= l_B, cancel add
                    # condition
                    if n_new in s_paths[n_box]:
                        if s_paths[n_box][n_new] >= l_B:
                            add_node = False
                    else:
                        add_node = False
    
                # If all checks have passed, add node to box
                if add_node:
                    current_box.append(n_new)
                    # Remove node from available
                    current_nodes.remove(n_new)
    
            # After all nodes are checked, finish box
            in_box = False
            boxes.append(current_box)
    
        # If there are no more nodes to analyze, finish loop
        if len(current_nodes) < 1:
            finished = True
    
    # Now that the box-covering is finished, we can build the 
    # normalized graph
    new_graph = nx.Graph()
    
    # Add nodes
    for i in range(len(boxes)):
        new_graph.add_node(i)
    
    # Add edges
    for i in range(len(boxes)-1):
        for j in range(i+1, len(boxes)):
            b1, b2 = boxes[i], boxes[j]
            # Do not add edge by default
            add_edge = False
            for n1 in b1:
                for n2 in b2:
                    if n2 in s_paths[n1]:
                        add_edge = True
            if add_edge:
                new_graph.add_edge(i, j)

    return new_graph
# -----------------------------------



# -------- Graph Analysis -----------
# Function the get the observed degree frequencies
def degree_frequencies(graph, norm=True):
    '''
    Params:
        graph : networkx Graph
            Graph for which to get the degree
            frequencies.
        norm : bool (optional)
            Whether to return normalized frequencies
            (True, default) or not (False). This 
            last option corresponds to returning the
            counts for each degree.
    Output:
        Returns the unique degree values found in
        the graph, and their respective frequencies
        or counts.
    '''
    # Get all degree data points
    k_all = np.array(graph.degree)[:,1]
    # Turn data into counts
    k_counts = [[k, np.sum(k_all == k)] for k in np.unique(k_all)]
    k_counts = np.array(k_counts)
    # Get unique k values
    k_values = k_counts[:,0]
    # Get frequencies or counts
    k_freqs = k_counts[:,1] 
    if norm:
        k_freqs = k_freqs / np.sum(k_counts[:,1])

    return k_values, k_freqs



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
# with only observed values
def P_joint_deg_emp(measures, k_limit):
    '''
    Params:
        measures : list or array
            Iterable containing the measurements of 
            (k_1, k_2) pairs.
        k_limit : int
            Maximum degree to consider when gathering
            degree-degree counts.

    Output:
        Returns a matrix A of size (k_unique, k_unique) where
        k_unique is the number of unique degrees identified in 
        the measures up to k_limit. Each element A_ij of the 
        matrix is the frequency of occurrence of the pair k_i,
        k_j. The elements A_ij are calculated in a frequentist 
        manner, only for the observed degree pairs. Also returns
        the corresponding degree for each row/column.
    '''
    # Turn measures into numpy array
    measures = np.array(measures)

    # Init list of values and A, along with k_unique
    k_vals = list(np.unique(measures))
    k_vals = [k for k in k_vals if k <= k_limit] # Filter
    
    k_unique = len(k_vals)
    A = np.zeros(shape=(k_unique, k_unique))
    
    # Iterate only over unique values
    for i, k_1 in enumerate(k_vals):
        # Get pairs with k_1
        k_1_mask = (measures[:,0] == k_1)
        
        # Proceed if pairs were found
        if np.sum(k_1_mask) > 0:
            k_1_pairs = measures[k_1_mask,:]
            # Iterate over unique k_2 values 
            for j, k_2 in enumerate(k_vals):
                # Get counts and save them
                k_2_counts = np.sum(k_1_pairs[:,1] == k_2)
                A[i, j] += k_2_counts
                # If not diagonal, save also the transpose
                if i != j:
                    A[j, i] += k_2_counts
        
    # Normalize by totality of measures to get frequency
    A = A / np.sum(A)
    return A, k_vals


# Function to get P(k_1, k_2) in a frequentist approach
# with interpolation
def P_joint_deg_interp(measures, k_max, grid_mode='log', samples=1000):
    '''
    Params:
        measures : list or array
            Iterable containing the measurements of 
            (k_1, k_2) pairs.
        k_max : int
            Maximum degree to consider.
        grid_mode : str (optional)
            Grid mode to use when building domain
            for interpolation. Default is log.
        samples : int (optional)
            Number of samples to use when building
            interpolation domain.
    Output:
        Returns a matrix A of size (k_max, k_max). 
        Each element A_ij of the matrix is the
        frequency of occurrence of the pair k_1=i,
        k_2=j in the given graph. Elements not 
        observed in the measurements are inferred
        from a bivariate spline interpolation.
    '''
    # Turn measures into numpy array
    measures = np.array(measures)

    # Init list of values and M, along with k_unique
    k_vals = list(np.unique(measures))
    k_vals = [k for k in k_vals if k <= k_max] # Filter
    
    k_unique = len(k_vals)
    M = np.zeros(shape=(k_unique, k_unique))
    
    # Iterate only over unique values
    for i, k_1 in enumerate(k_vals):
        # Get pairs with k_1
        k_1_mask = (measures[:,0] == k_1)
        
        # Proceed if pairs were found
        if np.sum(k_1_mask) > 0:
            k_1_pairs = measures[k_1_mask,:]
            # Iterate over unique k_2 values 
            for j, k_2 in enumerate(k_vals):
                # Get counts and save them
                k_2_counts = np.sum(k_1_pairs[:,1] == k_2)
                M[i, j] += k_2_counts
                # If not diagonal, save also the transpose
                if i != j:
                    M[j, i] += k_2_counts

    # Define a grid
    if grid_mode == 'linear':
        x_dom = np.linspace(start=1, stop=k_max, num=samples)
        y_dom = np.linspace(start=1, stop=k_max, num=samples)
        grid_x, grid_y = np.meshgrid(x_dom, y_dom, indexing='ij')
    elif grid_mode == 'log':
        x_dom = np.logspace(start=0, stop=np.log10(k_max), num=samples)
        y_dom = np.logspace(start=0, stop=np.log10(k_max), num=samples)
        grid_x, grid_y = np.meshgrid(x_dom, y_dom, indexing='ij')
        
    # Build interpolator
    M_dom = np.array(k_vals)
    rbs = RectBivariateSpline(M_dom, M_dom, M, kx=3, ky=3)
    
    # Evaluate at every point in the domain
    A = rbs.ev(grid_x, grid_y)
    
    # Normalize by totality of measures to get frequency
    A = A / np.sum(A)
    return A
# -----------------------------------