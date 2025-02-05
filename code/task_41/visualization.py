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




print('Process completed')
exit()