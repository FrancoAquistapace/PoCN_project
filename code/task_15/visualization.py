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


# ---------- Scale-free networks --------------------
# Read data
folder_path = '../../data/task_15/scale_free_sim/'
# Approximate size of scale free networks
N_sf = int(2e5) 
# Degree distribution exponents
gamma_sf = [2.01, 2.2, 2.4, 2.6, 2.8, 3.0, 5, "inf"]
# Half average degree
m_sf = 2

sf_data = []
for gamma in gamma_sf:
    df_path = 'sf_N_2e5_m_2_gamma_'+str(gamma).replace('.', '_')+'_sim.csv'
    df = pd.read_csv(folder_path + df_path)
    sf_data.append(df)


# Pre-processing: Keep only bulk avalanches
sf_data_final = []
for df in sf_data:
    df_new = df.loc[df.B == True]
    sf_data_final.append(df_new)

# Define theoretical function for tau with respect to gamma
def tau_func(g):
    return np.clip(g/(g-1), a_min=3/2, a_max=5)

# Get optimal parameters for avalanche size distribution
tau_opt = dict()
tau_sd_opt = dict()
C_opt = dict()

# Fixed s_c
m = 2
f = 1e-3
s_c = 1 / (2*m*f)

# Number of log bins
n_bins = 13

# Amount of initial values to skip
n_skip = 3

# Define model function
def t_func(s, tau, C):
    return C * np.power(s, -tau)*np.exp(-s / s_c)

for gamma in gamma_sf:
    # Perform logarithmic binning
    data = sf_data_final[gamma_sf.index(gamma)]
    vals, bins = np.histogram(data['A'], 
                    bins=np.logspace(start=0, stop=3, num=n_bins), 
                    density=True)

    # Fit function to data
    opt = curve_fit(t_func, bins[n_skip:-1], vals[n_skip:], 
                    p0=(3/2, 1), 
                    bounds=([1, 0], 
                            [4, 1.1]))

    tau_opt[gamma] = opt[0][0]
    tau_sd_opt[gamma] = np.sqrt(opt[1][0,0])
    C_opt[gamma] = opt[0][1]


# Define theoretical function for z with respect to gamma
def z_theory(g):
    return np.clip((g-1)/(g-2), a_min=2, a_max=120)

# Get dynamic coefficients from the data
z_opt = dict()
z_sd_opt = dict()
C_z_opt = dict()

# Define t bounds
t_low = [5, 5, 5, 5, 10, 10, 10, 10]
t_high = [11, 11, 11, 11, 50, 50, 50, 50]

# Define the model function
def z_model(t, z, C):
    return C * np.power(t, z)

for i, gamma in enumerate(gamma_sf):
    # Get data, exclude avalanches of duration 1
    df = sf_data_final[gamma_sf.index(gamma)]
    df = df.loc[(df['T'] > t_low[i]) & (df['T'] < t_high[i])]

    # Get mean avalanche size for each observed duration
    t_unique = np.unique(df['T'])
    s_mean = [np.mean(df.loc[df['T'] == t, 'S']) for t in t_unique]

    # Fit function to data
    opt = curve_fit(z_model, t_unique, s_mean, 
                    p0=(2, 1), 
                    bounds=([1.5, 0.], 
                            [110, 1.1]))

    z_opt[gamma] = opt[0][0]
    z_sd_opt[gamma] = np.sqrt(opt[1][0,0])
    C_z_opt[gamma] = opt[0][1]



# -------- Figure for Main Text - Exponents -----------------
# Figure parameters
FONTSIZE = 13
CS = 2.5

# Init figure
fig, ax = plt.subplots(figsize=(8,3), dpi=200, nrows=1, ncols=2)

# Plot (A): Avalanche size distribution exponents
# Plot theoretical curve for tau and values obtained from fits
gamma_vals = [g if g != 'inf' else 1000 for g in gamma_sf]
ax[0].plot(gamma_vals, list(tau_opt.values()),
             label='Simulation', marker='o')

# Plot theoretical curve
gamma_dom = np.linspace(2, 7, num=100)
ax[0].plot(gamma_dom, tau_func(gamma_dom), label='Theory')

# Axes labels
ax[0].set_xlabel(r'$\gamma$', fontsize=FONTSIZE)
ax[0].set_ylabel(r'$\tau$', fontsize=FONTSIZE)

# Limits
ax[0].set_xlim(left=1.9, right=5.2)
ax[0].set_ylim(top=2.6, bottom=1.4)


# Plot (B): Dynamic exponents
# Plot theoretical curve for tau and values obtained from fits
gamma_vals = [g if g != 'inf' else 1000 for g in gamma_sf]
ax[1].errorbar(gamma_vals, list(z_opt.values()), list(z_sd_opt.values()),
             label='Simulation', marker='o', capsize=CS)

# Plot theoretical curve
gamma_dom = np.linspace(2.001, 7, num=100)
ax[1].plot(gamma_dom, z_theory(gamma_dom), label='Theory')

# Axes labels
ax[1].set_xlabel(r'$\gamma$', fontsize=FONTSIZE)
ax[1].set_ylabel(r'$z$', fontsize=FONTSIZE)

# Limits
ax[1].set_xlim(left=1.9, right=5.2)
ax[1].set_ylim(top=4.5, bottom=1.5)

# Legend 
ax[1].legend(fontsize=FONTSIZE-2)

plt.tight_layout()
# Save figure
plt.savefig('../../latex/images/task_15/tau_and_z_vs_gamma.png',
            dpi=200)



# -------- Figure for SM - Avalanche distributions ----------
# Figure parameters
FONTSIZE = 13
colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:pink']
markers = ['o', 'D', '^', 's']
ls = ['-', '--', '-.', ':']

# Init figure
fig, ax = plt.subplots(figsize=(8,3), dpi=200, ncols=2, nrows=1)

# Plot (A): Avalanche size distribution
for i, gamma in enumerate([2.01, 2.2, 3.0, 'inf']):
    # Get data
    df_test = sf_data_final[gamma_sf.index(gamma)]

    # Plot hist values
    vals, bins = np.histogram(df_test['A'], 
                    bins=np.logspace(start=0, stop=3, num=n_bins), 
                    density=True)
    gamma_label = str(gamma).replace('inf', r'$\infty$')
    ax[0].scatter(bins[:-1], vals, c=colors[i], marker=markers[i],
                label=r'$\gamma =$%s' % gamma_label)

    # Get optimal parameters
    tau, C = tau_opt[gamma], C_opt[gamma]
    
    # Plot function
    dom = np.logspace(start=0, stop=4, num=100)
    y_vals = t_func(dom, tau, C)
    ax[0].plot(dom, y_vals, c=colors[i], ls=ls[i])

# Axes labels
ax[0].set_xlabel(r'$A$', fontsize=FONTSIZE)
ax[0].set_ylabel(r'$p_a(A)$', fontsize=FONTSIZE)
ax[0].tick_params(labelsize=FONTSIZE-2)

# Limits
ax[0].set_ylim(top=1, bottom=1e-10)

# Axes scale
ax[0].set_xscale('log')
ax[0].set_yscale('log')


# Plot (B): S vs T, dynamic exponents
for i, gamma in enumerate([2.01, 2.2, 3.0, 'inf']):
    # Get data
    df = sf_data_final[gamma_sf.index(gamma)]

    # Get mean avalanche size for each observed duration
    t_unique = np.unique(df['T'])
    s_mean = np.array(
        [np.mean(df.loc[df['T'] == t, 'S']) for t in t_unique])
    
    # Plot data
    gamma_label = str(gamma).replace('inf', r'$\infty$')
    ax[1].scatter(t_unique, s_mean, c=colors[i], 
                marker=markers[i], s=5)
    ax[1].scatter([], [], c=colors[i], marker=markers[i],
                label=r'$\gamma =$%s' % gamma_label)

    # Plot model
    t_dom = np.linspace(1, 200, num=50)
    s_vals = z_model(t_dom, z_opt[gamma], C_z_opt[gamma])
    ax[1].plot(t_dom, s_vals, c=colors[i], ls=ls[i])

# Labels
ax[1].set_xlabel(r'$T$', fontsize=FONTSIZE)
ax[1].set_ylabel(r'$S$', fontsize=FONTSIZE)
ax[1].tick_params(labelsize=FONTSIZE-2)

# Axes scale
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# Legend
ax[1].legend(fontsize=FONTSIZE-2)

# Limits
ax[1].set_ylim(bottom=1, top=1e4)
ax[1].set_xlim(left=1, right=150)

plt.tight_layout()

# Save figure
plt.savefig('../../latex/images/task_15/SM_scale_free_distributions.png',
            dpi=200)





# ---------- Interconnected networks ----------------
# Read data
folder_path = '../../data/task_15/joint_AB_sim/'

# Bernoulli coupling probability values
p_joint = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075,
           0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Size of A, B networks
N_AB = int(2e3)
# Degree of A, B networks
z_AB = 3

joint_df = []
for i, p in enumerate(p_joint):
    file_path = 'joint_A3_B3_p_'+str(p).replace('.', '_')+'_sim.csv'
    joint_df.append(pd.read_csv(folder_path+file_path))

# Get results for large avalanche probability
cut = N_AB / 2
P_loc_raw = []
P_inf_raw = []
P_tot_raw = []
for i in range(len(p_joint)):
    P_loc, P_inf, P_tot = get_avalanche_prob(joint_df[i], cut)
    P_loc_raw.append(P_loc)
    P_inf_raw.append(P_inf)
    P_tot_raw.append(P_tot)

# Estimate critical interconnection probability
p_crit = p_joint[np.argmin(P_tot_raw)]
print("Critical p for joint AB networks:", p_crit)


# ------- Figure for Main Text - Large avalanche probability --------
# Figure params
FONTSIZE = 13
COLORS = {'loc': 'mediumblue',
          'inf': 'firebrick', 
          'tot': 'goldenrod'}
MARKERS = {'loc': 'o',
           'inf': 's',
           'tot': 'D'}

# Init figure
fig, ax = plt.subplots(figsize=(5,3), dpi=200, nrows=1, ncols=1)

# Plot results
plt.plot(p_joint, P_loc_raw, c=COLORS['loc'], 
         marker=MARKERS['loc'], label='Local')
plt.plot(p_joint, P_inf_raw, c=COLORS['inf'], 
         marker=MARKERS['inf'], label='Inflicted')
plt.plot(p_joint, P_tot_raw, c=COLORS['tot'],
         marker=MARKERS['tot'], label='Global')

# Vertical line to indicate p critical
plt.vlines(p_crit, ymin=0, ymax=0.0045, 
           linestyles='--', colors='grey')
plt.text(x=p_crit+0.005, y=0.002, s=r'$p^*$', 
         fontsize=FONTSIZE-2)

# Labels
plt.ylabel(r'$P(S_{large})$', fontsize=FONTSIZE)
plt.xlabel(r'$p$', fontsize=FONTSIZE)
plt.tick_params(labelsize=FONTSIZE-2)

# Limits
plt.xlim(left=0, right=0.5)
plt.ylim(bottom=0, top=0.0045)

# Legend
plt.legend(fontsize=FONTSIZE-2)

plt.tight_layout()
# Save figure
plt.savefig('../../latex/images/task_15/large_S_vs_p_joint_AB.png',
            dpi=200)


# -------- Figure for SM - Largest avalanche rank plot --------
# Figure params
FONTSIZE = 13

# Init figure
fig, ax = plt.subplots(figsize=(5,3), dpi=200, nrows=1, ncols=1)

# Plot data
n_avalanches = int(1e3)
dom = 1+np.arange(n_avalanches)
for p in [0.001, 0.01, 0.1]:
    # Get data
    data = joint_df[p_joint.index(p)]

    # Get largest cascades rank plot
    largest = data.sort_values(by='S', ascending=False).iloc[:n_avalanches]['S']
    
    plt.plot(dom, largest, label=r'$p = %s$' % p)

# Scale
plt.xscale('log')
plt.yscale('log')

# Ticks
ax.set_yticks([1000, 2000, 3000], 
              labels=[1000, 2000, 3000])

# Labels
plt.xlabel('Rank', fontsize=FONTSIZE)
plt.ylabel(r'$S$', fontsize=FONTSIZE)
plt.tick_params(labelsize=FONTSIZE-2)

# Limits
plt.xlim(left=1, right=1e3)

# Legend
plt.legend(fontsize=FONTSIZE-2)

plt.tight_layout()
# Save figure
plt.savefig('../../latex/images/task_15/SM_large_avalanche_rank_plot_joint_AB.png',
            dpi=200)



print('Process completed')
exit()