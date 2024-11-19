'''
Here, we will analyse the results and produce the necessary plots.
'''
# ------ Import scripts and packages ---------
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from tools import *
# --------------------------------------------

# Parameters
E = [1, 0.8]
M = [2]
N_init = [5]
N_steps = 4
N_samples = 10
L_B = [i+1 for i in range(32)]

# Fetch data
fg_path = '../../data/task_6/'
G_minimal = dict()
G_minimal_r = dict()
G_minimal_norm = dict()
G_minimal_r_norm = dict()
for n_init in N_init:
    dict_n = dict()
    dict_n_r = dict()
    dict_n_nrm = dict()
    dict_n_r_nrm = dict()
    for e in E:
        dict_e = dict()
        dict_e_r = dict()
        dict_e_nrm = dict()
        dict_e_r_nrm = dict()
        for m in M:
            dict_m = []
            dict_m_r = []
            dict_m_nrm = []
            dict_m_r_nrm = []
            for i in range(N_samples):
                # Get minimal model graph
                g_path = 'minimal_model/' + 'Ni_' + str(n_init) +\
                         '_e_' + str(e).replace('.','_') +\
                         '_m_' + str(m) + '_Nst_4_' + str(i) + '.csv'
                g_df = pd.read_csv(fg_path + g_path)
                g_edges = np.array(g_df[['node_from', 'node_to']])
                g = nx.from_edgelist(g_edges)
                
                # Get randomized version
                g_rand_path = g_path.replace('minimal_model', 'random_model')
                g_r_df = pd.read_csv(fg_path + g_rand_path)
                g_r_edges = np.array(g_r_df[['node_from', 'node_to']])
                g_r = nx.from_edgelist(g_r_edges)

                # Save graphs
                dict_m.append(g)
                dict_m_r.append(g_r)

                dict_l = []
                dict_l_r = []
                for l_B in L_B:
                    # Get normalized minimal model
                    g_norm_path = 'minimal_model_norm/' + 'Ni_' + str(n_init) +\
                            '_e_' + str(e).replace('.','_') +\
                            '_m_' + str(m) + '_Nst_4_' + str(i) +\
                            '_lB_' + str(l_B) + '.csv'
                    g_nrm_df = pd.read_csv(fg_path + g_norm_path)
                    g_nrm_edges = np.array(g_nrm_df[['node_from', 'node_to']])
                    g_nrm = nx.from_edgelist(g_nrm_edges)

                    # Get normalized randomized version
                    g_r_norm_path = g_norm_path.replace('minimal_model_norm', 
                                                        'random_model_norm')
                    g_r_nrm_df = pd.read_csv(fg_path + g_r_norm_path)
                    g_r_nrm_edges = np.array(g_r_nrm_df[['node_from', 'node_to']])
                    g_r_nrm = nx.from_edgelist(g_r_nrm_edges)

                    # Save graphs
                    dict_l.append(g_nrm)
                    dict_l_r.append(g_r_nrm)

                # Save l_B graphs
                dict_m_nrm.append(dict_l)
                dict_m_r_nrm.append(dict_l_r)

            dict_e[str(m)] = dict_m.copy()
            dict_e_r[str(m)] = dict_m_r.copy()
            dict_e_nrm[str(m)] = dict_m_nrm.copy()
            dict_e_r_nrm[str(m)] = dict_m_r_nrm.copy()
        dict_n[str(e)] = dict_e.copy()
        dict_n_r[str(e)] = dict_e_r.copy()
        dict_n_nrm[str(e)] = dict_e_nrm.copy()
        dict_n_r_nrm[str(e)] = dict_e_r_nrm.copy()
    G_minimal[str(n_init)] = dict_n.copy()
    G_minimal_r[str(n_init)] = dict_n_r.copy()
    G_minimal_norm[str(n_init)] = dict_n_nrm.copy()
    G_minimal_r_norm[str(n_init)] = dict_n_r_nrm.copy()



# ------- N_B / N results -----------------------------------

# Now, we need to get the number of nodes in the original and
# renormalized networks
g_N = dict()
g_N_B = dict()
for e in G_minimal['5']:
    g_N[e] = [g.number_of_nodes() for g in G_minimal['5'][e]['2']]
    g_N_B[e] = [
        [g.number_of_nodes() for g in G_minimal_norm['5'][e]['2'][i]]\
             for i in range(len(G_minimal_norm['5'][e]['2']))]


# Separate data by e
data_0_8 = np.array(g_N['0.8']), np.array(g_N_B['0.8'])
data_1_0 = np.array(g_N['1']), np.array(g_N_B['1'])

# Get average N ratio measurements
N_ratio_0_8 = np.mean(
    np.expand_dims(np.power(data_0_8[0].astype('float'), -1), -1) * data_0_8[1], 
                      axis=0)
N_ratio_0_8_err = np.std(
    np.expand_dims(np.power(data_0_8[0].astype('float'), -1), -1) * data_0_8[1], 
                      axis=0)
N_ratio_1_0 = np.mean(
    np.expand_dims(np.power(data_1_0[0].astype('float'), -1), -1) * data_1_0[1], 
                      axis=0)
N_ratio_1_0_err = np.std(
    np.expand_dims(np.power(data_1_0[0].astype('float'), -1), -1) * data_1_0[1], 
                      axis=0)

# Make predictions for N ratio
L_0 = 0
n = 5
a = 1.4
d_B = np.log(n) / np.log(a)
def f_pred(x, A, d_B):
    return A * np.power(x + L_0, -d_B) 
    
popt, pcov = curve_fit(f_pred, np.array(L_B[8:]), N_ratio_0_8[8:])
preds_0_8 = f_pred(np.array(L_B), popt[0], popt[1])

def f_pred_2(x, A):
    return A * np.exp2(-x)
popt2, pcov2 = curve_fit(f_pred_2, np.array(L_B[0:12]), N_ratio_1_0[0:12])
preds_1_0 = f_pred_2(np.array(L_B), popt2[0])

# Plot results:
FONTSIZE = 15
CS = 2
fig, ax = plt.subplots()
ax.errorbar(L_B, N_ratio_1_0, yerr=N_ratio_1_0_err, 
            label=r'$e=1.0$', c='red', marker='o', 
            ls='', capsize=CS)
ax.plot(L_B, preds_1_0, c='red')
ax.errorbar(L_B, N_ratio_0_8, yerr=N_ratio_0_8_err, 
            label=r'$e=0.8$', c='k', marker='^', 
           ls='', capsize=CS)
ax.plot(np.array(L_B), preds_0_8, c='k')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xlabel(r'$L_B$', fontsize=FONTSIZE)
ax.set_ylabel(r'$N_B(L_B)/N$', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE-2)
plt.legend(fontsize=FONTSIZE-2)
plt.ylim(bottom=0.0002, top=1.2)

plt.savefig('../../latex/images/task_6/N_B_N_vs_L_B.png', 
            dpi=200)



exit()