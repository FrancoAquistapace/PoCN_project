'''
Here we will generate the data that is needed to reproduce the results for the 
Song-Havlin-Makse paper.
'''
# ------ Import scripts and packages ---------
import numpy as np 
import networkx as nx
from tools import *
# --------------------------------------------

# We want to reproduce the minimal model graphs reported in the original 
# work by Song, Havlin and Makse.

# Define global parameters
E = [1, 0.8] # Controls the growth mode
M = [2] # Growth factor, M*k new nodes are generated for each node with degree k
N_init = [5] # Number of initial nodes in the star graph
N_steps = 4 # Number of growth steps to perform
N_samples = 1 # Number of samples
seed = 42 # Initial random seed


# Generate the graphs and save them
folder_path = '../data/task_6/'
n_graph = 1
tot_graph = len(N_init) * len(E) * len(M) * len(N_samples)
for n_init in N_init:
    for e in E:
        for m in M:
            for i in range(N_samples):
                print('Generating graph %d of %d' % (n_graph, tot_graph))
                # Build minimal model graph
                g, _ = minimal_model_graph(
                            init_N=n_init, n_iters=N_steps, 
                            m=m, e=e, seed=seed)

                print('Randomizing graph\n')
                # Get randomized version
                g_r = randomize_graph(g, 2*g.number_of_edges())
                
                # Change seed
                seed += 1

                # Save graphs
                g_path = 'minimal_model/' + 'Ni_' + str(n_init) +\
                         '_e_' + str(e).replace('.','_') +\
                         '_m_' + str(m) + '_Nst_4_' + str(i) + '.csv'
                save_graph(g, folder_path + g_path)

                g_rand_path = g_path.replace('minimal_model', 'random_model')
                save_graph(g_r, folder_path + g_rand_path)

                # Update graph number
                n_graph += 1