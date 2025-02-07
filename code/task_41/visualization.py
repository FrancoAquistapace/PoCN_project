'''
Here, we will analyse the results and produce the necessary plots.
'''
# ------ Import scripts and packages ---------
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import networkx as nx
from tools import *
# --------------------------------------------


# ----------- Data input ----------------
# Find unique city names
data_path = '../../data/task_41/'
city_names = [p[-1] for p in os.walk(data_path)][0]
city_names = [s.replace('_nodes.csv', '') for s in city_names if '_nodes' in s]

print('Found %d unique cities' % len(city_names))

# Read all the city networks, but only keep those with data
city_nets = {}
cities_with_data = 0
for city in city_names:
    g = read_city_network(data_path + city)
    if len(g.nodes()) > 0:
        city_nets[city] = g
        cities_with_data += 1
print('Cities with data: %d' % (cities_with_data))
# ---------------------------------------



# ------------ FIGURES ------------------

# MAIN TEXT Figure: Degree distribution of small, medium and large cities
# Get the degree distribution of the small, medium and large cities
deg_small = []
deg_medium = []
deg_large = []
for city in city_nets:
    g = city_nets[city]
    size = len(g.nodes)
    deg = list(dict(g.degree()).values())
    if size < 100:
        deg_small.extend(deg)
    elif size >= 100 and size < 500:
        deg_medium.extend(deg)
    else:
        deg_large.extend(deg)

# Turn into numpy arrays
deg_small = np.array(deg_small)
deg_medium = np.array(deg_medium)
deg_large = np.array(deg_large)

# Figure params
FONTSIZE = 13

# Init figure
fig, ax = plt.subplots(figsize=(5,3), dpi=200, nrows=1, ncols=1)

# Get degree freqs and plot
k_vals_small, k_freqs_small = get_deg_frequencies(deg_small)
k_vals_medium, k_freqs_medium = get_deg_frequencies(deg_medium)
k_vals_large, k_freqs_large = get_deg_frequencies(deg_large)

# Plot degree frequencies
plt.plot(k_vals_small, k_freqs_small, label='Small')
plt.plot(k_vals_medium, k_freqs_medium, label='Medium')
plt.plot(k_vals_large, k_freqs_large, label='Large')

# Labels
plt.xlabel(r'$k$', fontsize=FONTSIZE)
plt.ylabel(r'$P(k)$', fontsize=FONTSIZE)
plt.tick_params(labelsize=FONTSIZE-2)

# Limits
plt.ylim(bottom=0, top=0.8)
plt.xlim(left=0, right=4)

# Legend
plt.legend(fontsize=FONTSIZE-2)

plt.tight_layout()
plt.savefig('../../latex/images/task_41/degree_dist_cities.png', dpi=200)



# MAIN TEXT Figure: Average shortest path length
# Get average shortest path length as a function of city size
N_vals = []
spath_large_vals = []
spath_mean_vals = []
for city in city_nets:
    g = city_nets[city]
    N_vals.append(len(g.nodes()))

    # Get largest connected component and its avg path
    g_lcc = max(nx.connected_components(g), key=len) 
    g_lcc = g.subgraph(g_lcc).copy()
    spath_large_vals.append(nx.average_shortest_path_length(g_lcc))

    # Get average from all connected components
    path_cc = []
    for C in (g.subgraph(c).copy() for c in nx.connected_components(g)):
        path_cc.append(nx.average_shortest_path_length(C))
    spath_mean_vals.append(np.mean(path_cc))

# Turn into numpy arrays
N_vals = np.array(N_vals)
spath_large_vals = np.array(spath_large_vals)
spath_mean_vals = np.array(spath_mean_vals)

# Now, get average value among all cities in a size bin (logarithmic)
n_bins = 6
N_bins = np.logspace(start=0, stop=np.log10(2200), num=n_bins)
spath_large = []
spath_large_err = []
spath_mean = []
spath_mean_err = []
for i in range(n_bins):
    if i == n_bins-1:
        N_mask = N_vals >= N_bins[i]
    else:
        N_mask = (N_vals >= N_bins[i]) * (N_vals < N_bins[i+1])
        
    # Get values for LCC 
    spath_large.append(np.mean(spath_large_vals[N_mask]))
    spath_large_err.append(np.std(spath_large_vals[N_mask]))
                           
    # Get mean values
    spath_mean.append(np.mean(spath_mean_vals[N_mask]))
    spath_mean_err.append(np.std(spath_mean_vals[N_mask]))

# Figure params
FONTSIZE = 13
CAPSIZE = 4

# Init figure
fig, ax = plt.subplots(figsize=(5,3), dpi=200, nrows=1, ncols=1)

# Plot results
plt.errorbar(N_bins, spath_large, yerr=spath_large_err, label='LCC', 
            capsize=CAPSIZE, marker='o')
plt.errorbar(N_bins, spath_mean, yerr=spath_mean_err, label='Average', 
            capsize=CAPSIZE, marker='o')

# Reference curves
#plt.plot(N_bins, 6*np.log10(N_bins))
#plt.plot(N_bins, 3*np.sqrt(np.log10(N_bins)))

# Labels
plt.ylabel(r'$l_G$', fontsize=FONTSIZE)
plt.xlabel(r'$N$', fontsize=FONTSIZE)

# Legend
plt.legend(fontsize=FONTSIZE-2)

# Scale
plt.xscale('log')

plt.tight_layout()
plt.savefig('../../latex/images/task_41/avg_path_length_cities.png', dpi=200)


# SM Figure: Size distribution of the networks
# Figure params
FONTSIZE = 13

# Init figure
fig, ax = plt.subplots(figsize=(5,3), dpi=200, nrows=1, ncols=1)

# Get city sizes 
city_sizes = []
for city in city_nets:
    g = city_nets[city]
    city_sizes.append(len(g.nodes()))

# Plot results
plt.hist(city_sizes, bins='sturges')

# Scale
plt.yscale('log')

# Labels
plt.ylabel('Count', fontsize=FONTSIZE)
plt.xlabel(r'$N$', fontsize=FONTSIZE)
plt.tick_params(labelsize=FONTSIZE-2)

# Limits
plt.xlim(left=0, right=2500)

plt.tight_layout()
plt.savefig('../../latex/images/task_41/networks_size_dist.png', dpi=200)



# SM Figure - Cities:
# Plot of two given cities
city_plot_1 = 'Tokyo'
city_plot_2 = 'Seoul'

# Retrieve raw data from the cities
city_plot_nodes_1 = pd.read_csv(data_path + city_plot_1 + '_nodes.csv')
city_plot_edges_1 = pd.read_csv(data_path + city_plot_1 + '_edges.csv')

city_plot_nodes_2 = pd.read_csv(data_path + city_plot_2 + '_nodes.csv')
city_plot_edges_2 = pd.read_csv(data_path + city_plot_2 + '_edges.csv')

# Figure params
FONTSIZE = 13
NODE_SIZE = 3.2
LINE_WIDTH = 1.8

# Init figure
fig, ax = plt.subplots(figsize=(10,5), dpi=200, nrows=1, ncols=2)

# Mode colors
mode_colors = ['tab:blue','tab:orange','tab:green','tab:cyan', 
               'tab:purple','tab:brown', 'tab:gray', 'tab:olive',
               'tab:pink','tab:red','indigo']

# Get unique modes
unique_modes = list(city_plot_edges_1['mode'].unique())
unique_modes.extend(list(city_plot_edges_2['mode'].unique()))
unique_modes = np.unique(unique_modes)

# Plot the stations and connections in space:
# Operate per mode to plot links
for n, m in enumerate(unique_modes):
    # City 1:
    mode_data = city_plot_edges_1.loc[city_plot_edges_1['mode'] == m]
    for i in range(mode_data.shape[0]):
        # Get node ids for the link
        u = city_plot_nodes_1.loc[
            city_plot_nodes_1['nodeID'] == mode_data.iloc[i]['nodeID_to']]
        v = city_plot_nodes_1.loc[
            city_plot_nodes_1['nodeID'] == mode_data.iloc[i]['nodeID_from']]
        
        # Get latitudes and longitudes of the two nodes
        lat = [u['latitude'], v['latitude']]
        long = [u['longitude'], v['longitude']]

        # Plot link
        ax[0].plot(lat, long, c=mode_colors[n], lw=LINE_WIDTH)

    # City 2:
    mode_data = city_plot_edges_2.loc[city_plot_edges_2['mode'] == m]
    for i in range(mode_data.shape[0]):
        # Get node ids for the link
        u = city_plot_nodes_2.loc[
            city_plot_nodes_2['nodeID'] == mode_data.iloc[i]['nodeID_to']]
        v = city_plot_nodes_2.loc[
            city_plot_nodes_2['nodeID'] == mode_data.iloc[i]['nodeID_from']]
        
        # Get latitudes and longitudes of the two nodes
        lat = [u['latitude'], v['latitude']]
        long = [u['longitude'], v['longitude']]

        # Plot link
        ax[1].plot(lat, long, c=mode_colors[n], lw=LINE_WIDTH)

    # Add mode label
    mode_name = m.replace('_', ' ').capitalize()
    ax[0].plot([],[], c=mode_colors[n], label=mode_name)

# Plot stations as points
for i, city_plot_nodes in enumerate([city_plot_nodes_1, city_plot_nodes_2]):
    node_lat = city_plot_nodes['latitude']
    node_long = city_plot_nodes['longitude']
    ax[i].scatter(node_lat, node_long, c='k', s=NODE_SIZE)

# Turn of axes
ax[0].axis('off')
ax[1].axis('off')

# Legend
fig.legend(fontsize=FONTSIZE-2, loc=[0.41, 0.65])

plt.tight_layout()
plt.savefig('../../latex/images/task_41/tokyo_seul_networks.png',dpi=200)

# ---------------------------------------



print('Process completed')
exit()