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
PROD = 'joint'

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



# ---------- Interconnected networks ----------------
if PROD == 'joint':
    # Define global parameters for joint network simulation:
    folder_path_joint = '../../data/task_15/joint_AB/'

    # Size of A, B networks
    N_AB = int(2e3)
    # Degree of A, B networks
    z_AB = 3
    # Bernoulli coupling probability values
    p_joint = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075,
            0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    joint_g = []
    joint_data = []
    for p in p_joint:
        # Read edge list
        g_path = 'joint_A3_B3_p_'+str(p).replace('.', '_')+'.csv'
        g_df = pd.read_csv(folder_path_joint + g_path)
        g_edges = np.array(g_df[['node_from', 'node_to']])
        g = nx.from_edgelist(g_edges)
        joint_g.append(g)

        # Read metadata
        data_path = g_path.replace('.csv', '_meta.csv')
        data_df = pd.read_csv(folder_path_joint + data_path)
        joint_data.append(data_df)

    # Output location for simulation results
    output_path_joint = '../../data/task_15/joint_AB_sim/'

    # Run simulations
    f = 0.01
    N_grains = int(2e5)
    for i in range(len(p_joint)):
        print('Running simulation %d of %d' % (i+1, len(p_joint)))
        # Init simulation
        sim = SandpileSimJoint(joint_g[i], joint_data[i], f=f)
        sim.init_thresholds()
        # Run simulation
        df = pd.DataFrame(sim.run(int(1e8), every=int(1e5), grains=N_grains))
        
        # Save results
        p_label = str(p_joint[i]).replace('.','_')
        df_path = 'joint_A3_B3_p_'+p_label+'_sim.csv'
        df.to_csv(output_path_joint + df_path, index=False)

print('Process complete')

exit()
