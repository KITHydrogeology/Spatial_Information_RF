# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:57:30 2023

@author: Marc
"""

import pandas as pd
import geopandas as gpd

from spatial_feature_functions import * # functions to include the spatial information

#%%

def read_data(url):
    # Mapping of URL to data sources
    url_mapping = {
        'point': 'https://raw.githubusercontent.com/marcohmer/Spatial_Information_RF/main/data/NO3_BW_data.csv',
        'grid': 'https://raw.githubusercontent.com/marcohmer/Spatial_Information_RF/main/data/NO3_BW_Grid.csv'
    }
    if url not in url_mapping:
        raise ValueError('Invalid URL!')  
    # Read data from the specified URL
    df = pd.read_csv(url_mapping[url], index_col="GW-Nummer" if url == 'point' else None)    
    # Rename the 'Nitrat' column to 'target'
    df.rename(columns={'Nitrat': 'target'}, inplace=True)    
    # Create a GeoDataFrame with geometry column from x and y coordinates
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs='EPSG:32632')
    df = df.drop(columns=['geometry'])
    return df, gdf


def preprocess(input_data, variables, parameters, regio=None):
    # Iterate over the variables and apply the respective functions
    for variable, status in variables.items():
        if status and variable in functions:
            print(variable)
            input_data = functions[variable](input_data, parameters)    
    # Drop x and y columns if GC is False
    if not variables['GC']:
        input_data.drop(columns=['x', 'y'], inplace=True)
    return input_data


def generate_variable_string(variables, parameters):
    result = ""    
    # Generate a string based on the selected variables and their parameters
    for key, value in variables.items():
        if value:
            if key in parameters:
                param_value = parameters[key]
                if isinstance(param_value, tuple):
                    param_str = ','.join(str(val) for val in param_value)
                    result += f'{key}{param_str}_'
                else:
                    result += f'{key}{param_value}_'
            else:
                result += f'{key}_'    
    # Remove the trailing underscore
    result = result.rstrip('_')    
    return result