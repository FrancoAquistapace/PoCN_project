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

print('Found %d unique cities in data' % len(city_names))
# ---------------------------------------