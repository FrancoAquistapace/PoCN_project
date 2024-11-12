'''
Here, we will normalize the previously generated networks.
'''
# ------ Import scripts and packages ---------
import numpy as np 
import networkx as nx
from tools import *
# --------------------------------------------

# Retrieve global parameters
E = [1, 0.8] # Growth mode
M = [2] # Growth factor
N_init = [5] # Number of initial nodes in the star graph
N_samples = 10 # Number of samples
seed = 42 # Initial random seed

L_B = [i+1 for i in range(32)] # List of normalization lengths to use

# Normalize the graphs and save them
folder_path = '../../data/task_6/'
n_graph = 1
tot_graph = len(N_init) * len(E) * len(M) * N_samples
is_single_node = False # To avoid unnecessary calculation
for n_init in N_init:
    for e in E:
        for m in M:
            for i in range(N_samples):
                # Reset check
                is_single_node = False

                print('Normalizing graph %d of %d' % (n_graph, tot_graph))
                # Get minimal model graph
                g_path = 'minimal_model/' + 'Ni_' + str(n_init) +\
                         '_e_' + str(e).replace('.','_') +\
                         '_m_' + str(m) + '_Nst_4_' + str(i) + '.csv'
                g_df = pd.read_csv(folder_path + g_path)
                g_edges = np.array(g_df[['node_from', 'node_to']])
                g = nx.from_edgelist(g_edges)
                
                # Get randomized version
                g_rand_path = g_path.replace('minimal_model', 'random_model')
                g_r_df = pd.read_csv(folder_path + g_rand_path)
                g_r_edges = np.array(g_r_df[['node_from', 'node_to']])
                g_r = nx.from_edgelist(g_r_edges)

                for l_B in L_B:
                    # Normalize minimal model graph
                    if not is_single_node:
                        # Only recalculate if the previous l_B was
                        # not a single node
                        g_norm = normalization_step(g, l_B=l_B)

                    # Check if normalized graph has a single node
                    if len(list(g_norm.nodes)) == 1:
                        is_single_node = True

                    # Normalize randomized version
                    g_r_norm = normalization_step(g_r, l_B=l_B)
                    
                    # Change seed
                    seed += 1

                    # Save graphs
                    g_norm_path = 'minimal_model_norm/' + 'Ni_' + str(n_init) +\
                            '_e_' + str(e).replace('.','_') +\
                            '_m_' + str(m) + '_Nst_4_' + str(i) +\
                            '_lB_' + str(l_B) + '.csv'
                    save_graph(g_norm, folder_path + g_norm_path)

                    g_r_norm_path = g_norm_path.replace('minimal_model_norm', 
                                                        'random_model_norm')
                    save_graph(g_r_norm, folder_path + g_r_norm_path)

                # Update graph number
                n_graph += 1



# Exit program
exit()