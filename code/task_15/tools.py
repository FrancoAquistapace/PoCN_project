'''
Here we will define all of the functions and tools that we need in order to reproduce the graphs
and results for the papers about sandpile dynamics on complex networks.
'''
# ------ Necessary packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
import time
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


# Function to create a joint network of regular graphs, 
# with Bernoulli coupling
def joint_AB_graph_Bernoulli(N_A, N_B, z_A, z_B, p):
    '''
    Params:
        N_A, N_B : int
            Number of nodes in subgraphs A and B.
        z_A, z_B : int
            Degrees for each node in A and B, respectively.
            The values z_A * N_A and z_B * N_B must be even.
        p : int
            Probability of a node in A having an external 
            edge to a node in B.
    Output:
        Returns a tuple (g, A_id, B_id). g is a networkx 
        Graph with a joint graph composed of two subgraphs A
        and B, where A is a z_A-regular graph and B a 
        z_B-regular graph. Furthermore, A and B are linked 
        by Bernoulli couplings, where each node in A has a 
        probability p of having one external edge to a node 
        in B. Additionally, A_id contains the id of the nodes
        that belong to A, and B_id contains the id of the 
        nodes that belong to B.
    '''
    # Generate A and B regular graphs
    A = nx.random_regular_graph(z_A, N_A)
    B = nx.random_regular_graph(z_B, N_B)

    # Shift node labels for B
    nx.relabel_nodes(B, mapping={i:i+N_A for i in range(N_B)}, 
                     copy=False)

    # Generate joint network
    g = nx.compose(A, B)

    # Get ids from A and B
    A_id = np.arange(N_A)
    B_id = np.arange(N_A, N_A+N_B)

    # Generate Bernoulli links:
    # Keep track of available nodes in B
    avail_B = [n_b for n_b in B_id] 
    for n_a in A_id:
        # Generate an external link with prob. p
        if np.random.uniform() < p:
            # Choose a random node in B from 
            # available
            n_b = np.random.choice(avail_B)

            # Add new link
            g.add_edge(n_a, n_b)

            # Remove used node in B from available
            avail_B.remove(n_b)
    
    # Return joint graph and ids
    return g, A_id, B_id


# Function to turn A, B ids into a single dataframe
def get_AB_dataframe(A_id, B_id):
    '''
    Params:
        A_id, B_id : array
            Arrays containing the node labels of all the
            nodes in subgraphs A and B of a joint network.
    Output:
        Returns a pandas DataFrame with columns "node" and 
        "ID", where "node" corresponds to the label of a 
        given node in the joint network, and "ID" is either
        "A" or "B" depending on which subgraph the node 
        belongs to.
    '''
    # Generate list of node labels and IDs
    labels = [n_a for n_a in A_id]
    labels.extend([n_b for n_b in B_id])
    
    IDs = ["A" for n_a in A_id]
    IDs.extend(["B" for n_b in B_id])

    # Turn into dataframe
    data = {"node": labels, "ID": IDs}
    data = pd.DataFrame(data)
    return data

# -----------------------------------


# ---------- Simulation -------------

class SandpileSim(object):
    def __init__(self, graph, f):
        '''
        Params:
            graph : networkx Graph
                Network to use as substrate for the simulation. The
                network should not have any node with degree 0 or 1.
            f : float
                Probability of losing a grain on a given transfer 
                between nodes.
        '''
        # System attributes
        # Network
        self.graph = graph
        self.nodes = np.array(list(graph.nodes()))
        self.nodes_idx = np.arange(len(self.nodes))
        self.node_map = {n:i for i,n in enumerate(self.nodes)}
        # Threshold values
        self.thresh = np.zeros((len(self.nodes,)))
        # Fraction of grains lost per transfer
        self.f = f

        # Initialize simulation information:
        # Number of avalanche events
        self.n_events = 0
        # Avalanche area
        self.A = []
        # Avalanche size
        self.S = []
        # Number of toppled grains
        self.G = []
        # Duration of the avalanche
        self.T = []
        # Whether the avalanche is "bulk" or not
        self.B = []

        # Current load of the system
        self.grains = np.zeros((len(self.nodes),))

    # Function to initialize threshold values
    def init_thresholds(self, mode='degree'):
        '''
        Params:
            mode : str or int (optional)
                Mode of assigning the threshold value to each node in the 
                network. Default is "degree", which sets the threshold of 
                each node as its degree. If an int value is given, then
                the thresholds are set uniformly to that value.
        Output:
            Initializes the self.thresh attribute of the system, which 
            contains the threshold value for each node in the network.
        '''
        # If default, then use network degrees as threshold:
        if mode == 'degree':
            for i, n in enumerate(self.nodes):
                self.thresh[i] += self.graph.degree(n)

        elif type(mode) == type(1):
            for i, n in enumerate(self.nodes):
                self.thresh[i] += mode

        else:
            print('Warning: the threshold mode given is not accepted.')


    # Function that adds a grain to a random node of the system
    def add_grain(self):
        '''
        Output:
            Adds a grain to a random node of the system. Each node
            is selected with uniform probability.
        '''
        # Select random node
        i_node = np.random.randint(0, len(self.nodes))

        # Add corresponding grain to current load
        self.grains[i_node] += 1


    # Function to check for avalanche threshold
    def avalanche_check(self):
        '''
        Output:
            Checks if the system is currently in an avalanche 
            event. Returns a bool accordingly.
        '''
        # Check for avalanche
        threshold_check = np.sum(self.grains >= self.thresh)
        if threshold_check > 0:
            return True
        return False

    
    # Function to run the simulation
    def run(self, steps, verbose=True, every=1000, events=None):
        '''
        Params:
            steps : int
                Number of maximum simulation steps to perform.
            verbose : bool (optional)
                Whether to print simulation information. Set 
                to True by default.
            every : int (optional)
                Amount of simulation steps in between information
                updates printed. Default is 1000 iterations.
            events : int (optional)
                Number of avalanche events at which to stop the
                simulation early. Not used by default.
        Output:
            Runs the Bak-Tang-Wiesenfeld sandpile model on the 
            given network. After it is finished, the following
            avalanche metrics are stored:
                - Number of events
                - Per event:
                    - Avalanche area (A)
                    - Avalanche size (S)
                    - Number of toppled grains (G)
                    - Duration of the avalanche (T)
        '''
        # Start timing
        t1 = time.time()
        
        # Centinel variables
        finished = False
        in_avalanche = False
        i_step = 0

        # Run loop
        while not finished:
            # Increase step
            i_step += 1

            # Check for printing frequency
            if verbose and i_step % every == 0:
                print('Step %s of %s. Observed avalanches: %s (max. %s).' % \
                      (i_step, steps, self.n_events, events))

            in_avalanche = self.avalanche_check()

            # If not in avalanche, increase grain load
            if not in_avalanche:
                self.add_grain()
            
            # If in avalanche, enter avalanche process until
            # completion
            else:
                # Init avalanche metrics
                part_nodes = [] # Nodes participating
                S = 0
                G = 0
                T = 0
                B = True
                while in_avalanche:
                    # Increase avalanche duration on each iteration
                    T += 1
                    
                    # Iterate over toppled nodes to generate a new
                    # state
                    grains_delta = np.zeros((len(self.nodes),))

                    # Get toppled nodes
                    t_mask = self.grains >= self.thresh
                    toppled = self.nodes[t_mask]
                    toppled_idx = self.nodes_idx[t_mask]
                    part_nodes.extend(list(toppled_idx))
                    S += len(toppled)
                    for i, nt in enumerate(toppled):
                        t_idx = toppled_idx[i]

                        # Get degree and neighbors of toppled node
                        deg = self.graph.degree(nt)
                        neigh_idx = [self.node_map[n]\
                                     for n in self.graph.neighbors(nt)]

                        # Remove k_i grains from toppled node
                        grains_delta[t_idx] -= deg
                        G += deg

                        # Add one grain to each neighbor with prob 1-f
                        probs = np.random.uniform(size=deg)
                        p_mask = (probs > self.f).astype('int')
                        for j, neigh in enumerate(neigh_idx):
                            grains_delta[neigh] += p_mask[j]
                        
                        # Check if a grain has been lost, and change
                        # bulk status accordingly
                        if 0 in p_mask:
                            B = False

                    # Add delta to current state to get new state
                    new_grains = self.grains + grains_delta
                    self.grains = new_grains
                
                    # Check if avalanche is finished
                    in_avalanche = self.avalanche_check()

                # Increase avalanche count and save results
                self.n_events += 1
                self.A.append(np.unique(part_nodes).shape[0])
                self.S.append(S)
                self.G.append(G)
                self.T.append(T)
                self.B.append(B)
            
            # Check for termination
            if i_step >= steps:
                finished = True
            if events != None:
                if self.n_events >= events:
                    finished = True

        # Finish timing
        t2 = time.time()

        if verbose:
            print('Simulation finished in: %.1f minutes' % ((t2 - t1)/60))

        # Return relevant esults
        res = {'A': self.A, 'S': self.S, 
               'G': self.G, 'T': self.T, 
               'B': self.B}
        return res



class SandpileSimJoint(SandpileSim):
    def __init__(self, graph, graph_data, f):
        '''
        Params:
            graph : networkx Graph
                Network to use as substrate for the simulation. The
                network should not have any node with degree 0 or 1.
            graph_data : pandas DataFrame
                Graph metadata, containing columns "node" and "ID", 
                where "node" is the label of the node in the network,
                and "ID" is either "A" or "B" depending on the 
                subgraph to which the node belongs.
            f : float
                Probability of losing a grain on a given transfer 
                between nodes.
        '''
        super().__init__(graph, f)
        # Joint network attributes
        self.meta = graph_data

        # Additional simulation attributes
        # Local and inflicted avalanche sizes
        self.S_local = []
        self.S_inflicted = []

        # Sub graph where the avalanche begins
        self.Init_graph = []

    
    # Function to run the simulation
    def run(self, steps, verbose=True, every=1000, events=None,
           grains=None):
        '''
        Params:
            steps : int
                Number of maximum simulation steps to perform.
            verbose : bool (optional)
                Whether to print simulation information. Set 
                to True by default.
            every : int (optional)
                Amount of simulation steps in between information
                updates printed. Default is 1000 iterations.
            events : int (optional)
                Number of avalanche events at which to stop the
                simulation early. Not used by default.
            grains : int (optional)
                Number of dropped grains on the system at which to
                stop the simulation early. Not used by default.
        Output:
            Runs the Bak-Tang-Wiesenfeld sandpile model on the 
            given joint network. After it is finished, the 
            following avalanche metrics are stored:
                - Number of events
                - Per event:
                    - Avalanche area (A)
                    - Avalanche size (S)
                    - Local avalanche size (S_local)
                    - Inflicted avalanche size (S_inf)
                    - Subgraph where the avalanche starts (I_g)
                    - Number of toppled grains (G)
                    - Duration of the avalanche (T)
        '''
        # Start timing
        t1 = time.time()
        
        # Centinel variables
        finished = False
        in_avalanche = False
        i_step = 0
        i_grains = 0

        # Run loop
        while not finished:
            # Increase step
            i_step += 1

            # Check for printing frequency
            if verbose and i_step % every == 0:
                print('Step %s of %s. Observed avalanches: %s (max. %s).' % \
                      (i_step, steps, self.n_events, events))

            in_avalanche = self.avalanche_check()

            # If not in avalanche, increase grain load
            if not in_avalanche:
                self.add_grain()
                i_grains += 1
            
            # If in avalanche, enter avalanche process until
            # completion
            else:
                # Get place of origin of avalanche
                av_node = self.nodes[self.grains >= self.thresh][0]
                I_g = self.meta.loc[self.meta.node == av_node, 'ID'].iloc[0]
                # Init avalanche metrics
                part_nodes = [] # Nodes participating
                S = 0
                S_local = 0
                S_inflicted = 0
                G = 0
                T = 0
                B = True
                while in_avalanche:
                    # Increase avalanche duration on each iteration
                    T += 1
                    
                    # Iterate over toppled nodes to generate a new
                    # state
                    grains_delta = np.zeros((len(self.nodes),))

                    # Get toppled nodes
                    t_mask = self.grains >= self.thresh
                    toppled = self.nodes[t_mask]
                    toppled_idx = self.nodes_idx[t_mask]
                    toppled_g = self.meta.loc[self.meta.node.isin(toppled), 'ID']
                    part_nodes.extend(list(toppled_idx))
                    S += len(toppled)
                    S_local += np.sum(toppled_g == I_g)
                    S_inflicted += np.sum(toppled_g != I_g)
                    for i, nt in enumerate(toppled):
                        t_idx = toppled_idx[i]

                        # Get degree and neighbors of toppled node
                        deg = self.graph.degree(nt)
                        neigh_idx = [self.node_map[n]\
                                     for n in self.graph.neighbors(nt)]

                        # Remove k_i grains from toppled node
                        grains_delta[t_idx] -= deg
                        G += deg

                        # Add one grain to each neighbor with prob 1-f
                        probs = np.random.uniform(size=deg)
                        p_mask = (probs > self.f).astype('int')
                        for j, neigh in enumerate(neigh_idx):
                            grains_delta[neigh] += p_mask[j]
                        
                        # Check if a grain has been lost, and change
                        # bulk status accordingly
                        if 0 in p_mask:
                            B = False

                    # Add delta to current state to get new state
                    new_grains = self.grains + grains_delta
                    self.grains = new_grains
                
                    # Check if avalanche is finished
                    in_avalanche = self.avalanche_check()

                # Increase avalanche count and save results
                self.n_events += 1
                self.A.append(np.unique(part_nodes).shape[0])
                self.S.append(S)
                self.G.append(G)
                self.T.append(T)
                self.B.append(B)

                # Additional metrics
                self.S_local.append(S_local)
                self.S_inflicted.append(S_inflicted)
                self.Init_graph.append(I_g)
            
            # Check for termination
            if i_step >= steps:
                finished = True
            if events != None:
                if self.n_events >= events:
                    finished = True
            if grains != None:
                if i_grains >= grains:
                    finished = True

        # Finish timing
        t2 = time.time()

        if verbose:
            print('Simulation finished in: %.1f minutes' % ((t2 - t1)/60))

        # Return relevant esults
        res = {'A': self.A, 'S': self.S, 
               'G': self.G, 'T': self.T, 
               'B': self.B, 'I_g': self.Init_graph,
               'S_local': self.S_local,
               'S_inf': self.S_inflicted}
        return res

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


# Function to get the unique values and respective
# frequencies for a given dataset
def get_frequency(data):
    '''
    Params:
        data : array
            Array of shape (N,) containing N measurements of
            a given observable.
    Output:
        Returns a tuple (vals, freqs) where vals is an array
        that contains the unique values found in the data, and
        freqs is an array containing the respective frequency 
        of the unique values.
    '''
    # Turn data into numpy array
    obs = np.array(data)
    vals = np.unique(obs)
    freq = np.array([np.sum(obs==k) for k in vals])
    freq = freq / obs.shape[0]
    return vals, freq


# Function to get local, inflicted and total avalanche
# probabilities
def get_avalanche_prob(data, cut):
    '''
    Params:
        data : pandas DataFrame
            Data containing avalanche instances with all the
            respective metrics.
        cut : int 
            Cutoff value that defines large avalanches in the
            amount of nodes topppled S, S_loc or S_inf.
    Output:
        Returns a tuple (P_loc, P_inf, P_tot) where P_loc is
        the probability of a large avalanche that started 
        locally, P_inf is the probability of a large avalanche
        that was inflicted externally and P_tot the probability 
        of a large avalanche regardless of the origin.
    '''
    # Separate data into A and B origins
    data_A = data.loc[data.I_g == 'A']
    data_B = data.loc[data.I_g == 'B']

    # Get number of avalanches in each system
    N_tot_A = data_A.shape[0] + np.sum(data_B.S_inf > 0)
    N_tot_B = data_B.shape[0] + np.sum(data_A.S_inf > 0)

    # Get local large avalanche probability
    A_loc = np.sum(data_A.S_local > cut) / N_tot_A
    B_loc = np.sum(data_B.S_local > cut) / N_tot_B
    P_loc = (A_loc + B_loc) / 2

    # Get inflicted large avalanche probability
    A_inf = np.sum(data_A.S_inf > cut) / N_tot_B
    B_inf = np.sum(data_B.S_inf > cut) / N_tot_A
    P_inf = (A_inf + B_inf) / 2

    # Get total large avalanche probability
    P_tot = (P_loc + P_inf) / 2

    return P_loc, P_inf, P_tot 

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