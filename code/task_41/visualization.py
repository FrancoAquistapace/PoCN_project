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

# SM Figure:
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