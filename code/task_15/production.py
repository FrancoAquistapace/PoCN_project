'''
Here we will generate the networks that are then going to be used
for the Bak-Tang-Wiesenfeld sandpile dynamics
'''
# ------ Import scripts and packages ---------
import numpy as np 
import networkx as nx
from tools import *
# --------------------------------------------

# Switch between possibly different production runs
PROD = 'scale_free'

# ---------- Scale-free networks --------------------
# Define global parameters for scale-free production:
folder_path_sf = '../../data/task_15/scale_free/'
# Approximate size of scale free networks
N_sf = int(2e5) 
# Degree distribution exponents
gamma_sf = [2.01, 2.2, 2.4, 2.6, 2.8, 3.0, 5, "inf"]
# Half average degree
m_sf = 2

if PROD == 'scale_free':
    # Produce graphs
    for i, gamma in enumerate(gamma_sf):
        print('Producing network with gamma =', gamma)

        # Build network
        g = scale_free_graph(N=N_sf, gamma=gamma, m=m_sf)

        # Save to file
        gamma_label = str(gamma).replace('.','_')
        g_path = 'sf_N_2e5_m_2_gamma_'+gamma_label+'.csv'
        save_graph(g, folder_path_sf + g_path)

print('Process completed')

exit()
