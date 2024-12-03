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


# ------ FIRST FIGURE ---------------------------------------
# Init figure
FONTSIZE = 20
MS = 7
CS = 2
LW = 2
fig, ax = plt.subplots(figsize=(13, 5), 
                       nrows=1, ncols=2, dpi=200)

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
d_B = np.log(n) / np.log(a) # Expected slope

# Power function
def f_pred(x, A, d_B):
    return A * np.power(x + L_0, -d_B) 
popt, pcov = curve_fit(f_pred, np.array(L_B[8:]), N_ratio_0_8[8:])
preds_0_8 = f_pred(np.array(L_B), popt[0], popt[1])

# Exponential function
def f_pred_2(x, A):
    return A * np.exp2(-x)
popt2, pcov2 = curve_fit(f_pred_2, np.array(L_B[0:12]), N_ratio_1_0[0:12])
preds_1_0 = f_pred_2(np.array(L_B), popt2[0])

# Plot results:
# 1.0 data
ax[0].errorbar(L_B, N_ratio_1_0, yerr=N_ratio_1_0_err, 
            label=r'$e=1.0$', c='red', marker='o', 
            ls='', capsize=CS, markersize=MS)
ax[0].plot(L_B, preds_1_0, c='red', lw=LW)
# 0.8 data
ax[0].errorbar(L_B, N_ratio_0_8, yerr=N_ratio_0_8_err, 
            label=r'$e=0.8$', c='k', marker='^', 
           ls='', capsize=CS, markersize=MS)
ax[0].plot(np.array(L_B), preds_0_8, c='k', lw=LW)
# Axes scales
ax[0].set_xscale('log', base=2)
ax[0].set_yscale('log', base=2)
# Label params
ax[0].set_xlabel(r'$L_B$', fontsize=FONTSIZE)
ax[0].set_ylabel(r'$N_B(L_B)/N$', fontsize=FONTSIZE)
ax[0].tick_params(labelsize=FONTSIZE-2)
# Legend
ax[0].legend(fontsize=FONTSIZE-2)
# Limits
ax[0].set_ylim(bottom=0.0002, top=1.2)


# ------- S(L_B) results -----------------------------------

# Now let's get the degree data
g_k = dict()
g_k_B = dict()
for e in G_minimal['5']:
    g_k[e] = [np.max(np.array(g.degree)[:,1]) for g in G_minimal['5'][e]['2']]
    g_k_B[e] = [
        [np.max(np.array(g.degree)[:,1]) for g in G_minimal_norm['5'][e]['2'][i]]\
             for i in range(len(G_minimal_norm['5'][e]['2']))]

# Separate data by e
k_data_0_8 = np.array(g_k['0.8']), np.array(g_k_B['0.8'])
k_data_1_0 = np.array(g_k['1']), np.array(g_k_B['1'])

# Get average ratio measurements
k_ratio_0_8 = np.mean(
    np.expand_dims(np.power(k_data_0_8[0].astype('float'), -1), -1) * k_data_0_8[1], 
                      axis=0)
k_ratio_0_8_err = np.std(
    np.expand_dims(np.power(k_data_0_8[0].astype('float'), -1), -1) * k_data_0_8[1], 
                      axis=0)
k_ratio_1_0 = np.mean(
    np.expand_dims(np.power(k_data_1_0[0].astype('float'), -1), -1) * k_data_1_0[1], 
                      axis=0)
k_ratio_1_0_err = np.std(
    np.expand_dims(np.power(k_data_1_0[0].astype('float'), -1), -1) * k_data_1_0[1], 
                      axis=0)

# Make predictions
L_0 = 0
s = 3
a = 1.4
d_k = np.log(s) / np.log(a) # Expected slope

# Power function
def f_pred(x, A, d_k):
    return A * np.power(x + L_0, -d_k) 
    
popt, pcov = curve_fit(f_pred, np.array(L_B[12:]), k_ratio_0_8[12:])
preds_0_8 = f_pred(np.array(L_B), popt[0], popt[1])

# Exponential functions
def f_pred_2(x, A):
    return A * np.exp2(-0.5*np.log(s)*(x + L_0))
popt2, pcov2 = curve_fit(f_pred_2, np.array(L_B[2:12]), k_ratio_1_0[2:12])
preds_1_0 = f_pred_2(np.array(L_B), popt2[0])

def f_pred_3(x, A, l0):
    return A * np.exp(-x / l0)
popt3, pcov3 = curve_fit(f_pred_3, np.array(L_B[1:12]), k_ratio_1_0[1:12])
preds_1_0 = f_pred_3(np.array(L_B), popt3[0], popt3[1])

# Plot results
# 1.0 data
ax[1].errorbar(L_B, k_ratio_1_0, yerr=k_ratio_1_0_err, 
            label=r'$e=1.0$', c='red', marker='o',
           ls='', capsize=CS, markersize=MS)
ax[1].plot(L_B, preds_1_0, c='red', lw=LW)
# 0.8 data
ax[1].errorbar(L_B, k_ratio_0_8, yerr=k_ratio_0_8_err,
            label=r'$e=0.8$', c='k', marker='^', 
           ls='', capsize=CS, markersize=MS)
ax[1].plot(np.array(L_B), preds_0_8, c='k', lw=LW)
# Axes scales
ax[1].set_xscale('log', base=2)
ax[1].set_yscale('log', base=2)
# Label params
ax[1].set_xlabel(r'$L_B$', fontsize=FONTSIZE)
ax[1].set_ylabel(r'$\mathcal{S}(L_B)$', fontsize=FONTSIZE)
ax[1].tick_params(labelsize=FONTSIZE-2)
# Legend
ax[1].legend(fontsize=FONTSIZE-2)
# Limits
ax[1].set_ylim(bottom=0.004, top=4)

plt.tight_layout()

# Save figure
plt.savefig('../../latex/images/task_6/N_B_and_S_vs_L_B.png', 
            dpi=200)



# ------ SECOND FIGURE: DEGREE MIXING RATIOS ----------------
# Get degree pair observations
measures_1_0 = []
measures_1_0_r = []
measures_0_8 = []
measures_0_8_r = []
for i in range(len(G_minimal['5']['1']['2'])):
    # e=1.0 case
    measures_1_0.extend(degree_pairs(G_minimal['5']['1']['2'][i]))
    measures_1_0_r.extend(degree_pairs(G_minimal_r['5']['1']['2'][i]))
    
    # e=0.8 case
    measures_0_8.extend(degree_pairs(G_minimal['5']['0.8']['2'][i]))
    measures_0_8_r.extend(degree_pairs(G_minimal_r['5']['0.8']['2'][i]))

# ---- e=1.0 case ----
# Get joint degree-degree probability
P_1_0 = P_joint_deg_emp(measures_1_0, k_limit=400)
P_1_0_r = P_joint_deg_emp(measures_1_0_r, k_limit=400)
# Make P minimal same shape as random model
P_1_0_new = np.zeros(shape=P_1_0_r[0].shape)
for i in P_1_0[1]:
    for j in P_1_0[1]:
        old_col = P_1_0[1].index(i)
        old_row = P_1_0[1].index(j)
        # Get corresponding indexes from random
        new_row = P_1_0_r[1].index(i)
        new_col = P_1_0_r[1].index(j)
        P_1_0_new[new_row,new_col] += P_1_0[0][old_row, old_col]

# Get degree mixing ratio (avoiding division by zero)
R_1_0 = np.divide(P_1_0_new, P_1_0_r[0], 
                  out=np.ones_like(P_1_0_new), 
                  where=P_1_0_r[0]!=0)

# ---- e=0.8 case ----
# Get joint degree-degree probability
P_0_8 = P_joint_deg_emp(measures_0_8, k_limit=400)
P_0_8_r = P_joint_deg_emp(measures_0_8_r, k_limit=400)
# Get degree mixing ratio (avoiding division by zero)
R_0_8 = np.divide(P_0_8[0], P_0_8_r[0], 
                  out=np.ones_like(P_0_8[0]), 
                  where=P_0_8_r[0]!=0)

# Init figure
FONTSIZE = 18
STEP = 9
fig, axes = plt.subplots(figsize=(13, 5), nrows=1, ncols=2)

# 1.0 data:
img_0 = axes[0].imshow(R_1_0, origin='lower', vmin=0, vmax=18)
# Title
axes[0].set_title(r'$e=1.0$', fontsize=FONTSIZE)
# Axes params (in this case we can use all observations)
axes[0].set_xticks(range(R_1_0.shape[0]), P_1_0_r[1], size=FONTSIZE-2)
axes[0].set_yticks(range(R_1_0.shape[0]), P_1_0_r[1], size=FONTSIZE-2)
# Labels
axes[0].set_xlabel(r'$k_1$', fontsize=FONTSIZE)
axes[0].set_ylabel(r'$k_2$', fontsize=FONTSIZE)
# Colorbar
cbar_0 = plt.colorbar(img_0)
cbar_0.set_label(r'$R(k_1, k_2)$', fontsize=FONTSIZE)
cbar_0.ax.tick_params(labelsize=FONTSIZE-2)

# 0.8 data
img_1 = axes[1].imshow(R_0_8, origin='lower', vmin=0, vmax=10)
# Title
axes[1].set_title(r'$e=0.8$', fontsize=FONTSIZE)
# Axes params (in this case we must use a step)
axes[1].set_xticks(np.arange(0, R_0_8.shape[0], step=STEP), 
                   P_0_8[1][::STEP], size=FONTSIZE-2, rotation=0)
axes[1].set_yticks(np.arange(0, R_0_8.shape[0], step=STEP), 
                   P_0_8[1][::STEP], size=FONTSIZE-2)
# Labels
axes[1].set_xlabel(r'$k_1$', fontsize=FONTSIZE)
axes[1].set_ylabel(r'$k_2$', fontsize=FONTSIZE)
# Colorbar
cbar_1 = plt.colorbar(img_1)
cbar_1.set_label(r'$R(k_1, k_2)$', fontsize=FONTSIZE)
cbar_1.ax.tick_params(labelsize=FONTSIZE-2)

plt.tight_layout()

# Save figure
plt.savefig('../../latex/images/task_6/R_k1_k2_comparison.png',
            dpi=200)

exit()