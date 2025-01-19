'''
Here we will perform Bak-Tang-Wiesenfeld sandpile dynamics
simulations on the previously generated networks.
'''
# ------ Import scripts and packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
from tools import *
# --------------------------------------------

# Switch between possibly different simulation runs
PROD = 'scale_free'

# ---------- Scale-free networks --------------------
if PROD == 'scale_free':
    # Networks folder path
    folder_path_sf = '../../data/task_15/scale_free/'
    # Approximate size of scale free networks
    N_sf = int(2e5) 
    # Degree distribution exponents
    gamma_sf = [2.01, 2.2, 2.4, 2.6, 2.8, 3.0, 5, "inf"]
    # Half average degree
    m_sf = 2

    # Read scale-free networks
    sf_g = []
    for gamma in gamma_sf:
        # Read edge list
        g_path = 'sf_N_2e5_m_2_gamma_'+str(gamma).replace('.', '_')+'.csv'
        g_df = pd.read_csv(folder_path_sf + g_path)
        g_edges = np.array(g_df[['node_from', 'node_to']])
        # Build graph
        g = nx.from_edgelist(g_edges)
        # Get only the LCC 
        sf_g.append(get_lcc(g))

    # Output location for simulation results
    output_path_sf = '../../data/task_15/scale_free_sim/'

    # Grain loss probability
    f = 1e-3

    # Perform simulations
    for i, g in enumerate(sf_g):
        print('Running simulation with gamma =', gamma_sf[i])
        # Init simulation
        sim = SandpileSim(g, f=f)
        sim.init_thresholds()
        # Run simulation
        df = pd.DataFrame(sim.run(int(1e8), events=int(1e6), every=int(5e5)))
        
        # Save results
        gamma_label = str(gamma_sf[i]).replace('.','_')
        df_path = 'sf_N_2e5_m_2_gamma_'+gamma_label+'_sim.csv'
        df.to_csv(output_path_sf + df_path, index=False)


print('Process complete')

exit()
