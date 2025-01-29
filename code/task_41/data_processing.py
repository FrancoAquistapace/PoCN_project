'''
Here we will define all of the functions and tools that we need in order to build the graphs
from the citylines dataset, and perform the data processing and graph generation.
'''
# ------ Necessary packages ---------
import numpy as np 
import pandas as pd
import networkx as nx
import time
# -----------------------------------


# ------------ Data functions ---------------
# Function to retrieve all the data relative to a city, given its id
def data_from_city_id(city_id, datasets=all_data):
    '''
    Params:
        city_id : int
            Identifier of a city present in the citylines dataset.
    Output:
        Returns a dictionary with the relevant data from the given
        city.
    '''
    # Init results
    res = dict()
    
    # Get city data
    city_data = datasets['cities'].loc[datasets['cities']['id'] == city_id].iloc[0]
    res['city'] = city_data

    # Get systems data
    sys_df = datasets['systems']
    sys_data = sys_df.loc[sys_df.city_id == city_id]
    res['systems'] = sys_data

    # Get lines data
    lines_data = datasets['lines'].loc[datasets['lines'].city_id == city_id]
    res['lines'] = lines_data

    # Get sections data
    sect_df = datasets['sections']
    sect_data = sect_df.loc[sect_df.city_id == city_id]
    res['sections'] = sect_data

    # Get section lines data
    sect_lines_df = datasets['section_lines']
    sect_lines_data = sect_lines_df.loc[sect_lines_df.city_id == city_id]
    res['section_lines'] = sect_lines_data

    # Get stations data
    stat_df = datasets['stations']
    stat_data = stat_df.loc[stat_df.city_id == city_id]
    res['stations'] = stat_data

    # Get station lines data
    stat_lines_df = datasets['station_lines']
    stat_lines_data = stat_lines_df.loc[stat_lines_df.city_id == city_id]
    res['station_lines'] = stat_lines_data

    # Get all transport modes
    modes_df = datasets['transport_modes']
    res['modes'] = modes_df
    
    return res


# Function that checks if either station data or section data is missing
def check_missing_data(city_data):
    '''
    Params:
        city_data : dict
            Data of a given city, with at least "stations" and
            "sections" keys.
    Output:
        Returns True if either the stations data or the sections
        data is missing for the given city, and returns False
        otherwise.
    '''
    no_stations = city_data['stations'].shape[0] == 0
    no_sections = city_data['sections'].shape[0] == 0
    if no_stations or no_sections:
        return True
        
    return False


# Function to extract latitude and longitude from geometry data
def get_point_location(point):
    '''
    Params:
        point : str
            Point location of a station as defined in the 
            geometry feature of the citylines dataset.
    Output:
        Returns a tuple (latitude, longitude) with the 
        location of the given point.
    '''
    # Remove geometry notation
    lat_long = point.replace('POINT(', '')
    lat_long = lat_long.replace(')', '')
    # Split data
    lat_long = lat_long.split()
    # Convert to numerical and return 
    lat = float(lat_long[0])
    long = float(lat_long[1])
    return lat, long
# -------------------------------------------





# -------- Data import and processing --------
# We need to read all the csv files:
print('Reading data')
# Folder path
folder_path = '../../../citylines_data/'

# Cities data
cities_path = 'dataclips_fbkrwnzullhoggqxrynyqpmerrdm.csv'
cities_df = pd.read_csv(folder_path + cities_path)

# Systems data
systems_path = 'dataclips_ktjczpiaitvdflctatuhrbyymwzb.csv'
systems_df = pd.read_csv(folder_path + systems_path)

# Lines data
lines_path = 'dataclips_phsrswqjdraavfhlllnkjgfjgngg.csv'
lines_df = pd.read_csv(folder_path + lines_path)

# Sections data
sections_path = 'dataclips_irtgepjgovwtgvykpfyqxmoqhbcr.csv'
sections_df = pd.read_csv(folder_path + sections_path)

# Section lines data
section_lines_path = 'dataclips_ixybcilzttdifyvkhfriembrusxa.csv'
section_lines_df = pd.read_csv(folder_path + section_lines_path)

# Stations data
stations_path = 'dataclips_ssxbtmzqfqzgdsxhlibtetfutilz.csv'
stations_df = pd.read_csv(folder_path + stations_path)

# Station lines
station_lines_path = 'dataclips_vezkvrglavsdfvsxpuydclocyfsk.csv'
station_lines_df = pd.read_csv(folder_path + station_lines_path)

# Transport modes
transport_modes_path = 'dataclips_ictamqsydokxnpweeoujgtphosfn.csv'
transport_modes_df = pd.read_csv(folder_path + transport_modes_path)

# Define a single dictionary with access to all the data
all_data = {'cities': cities_df, 
            'systems': systems_df, 'lines': lines_df,
            'sections': sections_df, 'section_lines': section_lines_df,
            'stations': stations_df, 'station_lines': station_lines_df,
            'transport_modes': transport_modes_df}

# -------------------------------------------


print('Process completed')
exit()