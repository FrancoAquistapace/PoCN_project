'''
Here we will generate the networks that are then going to be used
for the Bak-Tang-Wiesenfeld sandpile dynamics
'''
# ------ Import scripts and packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
from tools import *
# --------------------------------------------

# Switch between possibly different production runs
PROD = 'joint'

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



# ---------- Interconnected networks ----------------
# Define global parameters for joint network production:
folder_path_joint = '../../data/task_15/joint_AB/'

# Size of A, B networks
N_AB = int(2e3)
# Degree of A, B networks
z_AB = 3
# Bernoulli coupling probability values
p_joint = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075,
           0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

if PROD == 'joint':
    # Produce graphs
    for i, p in enumerate(p_joint):
        print('Producing network with p =', p)

        # Build network
        g, A_id, B_id = joint_AB_graph_Bernoulli(
            N_A=N_AB, N_B=N_AB, 
            z_A=z_AB, z_B=z_AB,
            p=p
        )

        # Build node id data
        data = get_AB_dataframe(A_id, B_id)

        # Save graph to file
        p_label = str(p).replace('.','_')
        g_path = 'joint_A3_B3_p_'+p_label+'.csv'
        save_graph(g, folder_path_joint + g_path)

        # Save node data to file
        data_path = g_path.replace('.csv', '_meta.csv')
        data.to_csv(folder_path_joint + data_path, 
                    index=False)


print('Process completed')

exit()
