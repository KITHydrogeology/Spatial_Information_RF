# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:59:15 2023

@author: Marc
"""


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

#%%



def PolynomialGeographicCoordinates(data, parameters, regio=None):
    '''
    The function converts the 'x' and 'y' columns of the 'data' DataFrame into a numpy array 
    and applies a 4th degree polynomial transformation on it. The result of the transformation is added 
    as new columns to the original 'data' DataFrame, and the '1st', 'x', and 'y' columns are dropped. 
    '''
    # Convert 'x' and 'y' columns to a numpy array
    points_poly = data[['x', 'y']].to_numpy()
    
    # Apply 4th degree polynomial transformation
    poly = PolynomialFeatures(degree=4)
    data_poly = poly.fit_transform(points_poly)
    
    # Create a DataFrame with the transformed data
    data_poly = pd.DataFrame(data_poly, columns=['1','x','y','x^2','xy','y^2','x^3','x^2y','xy^2','y^3','x^4','x^3y','x^2y^2','xy^3','y^4'])
    
    # Drop unnecessary columns
    data_poly.drop(columns=['1', 'x','y'], inplace=True)
    
    # Join the transformed data with the original data
    data_poly = data.join(data_poly)
    
    # Convert column names to string type
    data_poly.columns = data_poly.columns.astype(str)
    
    return data_poly

def ObliqueGeographicCoordinates(data, parameters, regio=None):
    '''
    The function adds oblique coordinates (0-180 degrees, with intervals of 5 degrees) to a data set.
    It takes the input data and parameters, converts the 'x' and 'y' coordinates to numpy arrays, 
    calculates the oblique coordinates using trigonometry, and returns the data with the added 
    oblique coordinates.
    '''
    # Get parameters for oblique coordinates
    start, step, stop = parameters['OGC']
    
    # Calculate angles in degrees and radians
    angles_deg = np.arange(start=start, step=step, stop=stop)
    angles_rad = np.radians(angles_deg)
    
    # Convert 'x' and 'y' columns to numpy arrays
    x = data[['x']].to_numpy()
    y = data[['y']].to_numpy()
    
    # Calculate oblique coordinates for each angle
    for i in angles_rad:
        OGC = np.sqrt(x**2 + y**2) * np.cos(angles_rad - np.arctan(x/y))
    
    # Create a DataFrame with the oblique coordinates
    OGC = pd.DataFrame(OGC)
    
    # Reset the index of the original data
    data.reset_index(inplace=True)
    
    # Join the oblique coordinates with the original data
    data_OGC = data.join(OGC)
    
    # Convert column names to string type
    data_OGC.columns = data_OGC.columns.astype(str)
    
    return data_OGC

def WTC_kernel(normalized_lon, normalized_lat, n):
    # Define the number of basis functions
    num_basis = [n ** 2, (2*n) ** 2, (4*n) ** 2]
    
    # Define the knots for each basis function
    knots_1dx = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
    
    # Initialize the phi matrix
    basis_size = 0
    phi = np.zeros((normalized_lon.shape[0], sum(num_basis)))
    
    # Compute the weight functions for each basis function
    for res in range(len(num_basis)):
        theta = 1 / np.sqrt(num_basis[res]) * 2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res], knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(), knots_y.flatten()))
        
        for i in range(num_basis[res]):
            d = np.linalg.norm(np.vstack((normalized_lon, normalized_lat)).T - knots[i, :], axis=1) / theta
            
            for j in range(len(d)):
                if 0 <= d[j] <= 1:
                    phi[j, i + basis_size] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3
                else:
                    phi[j, i + basis_size] = 0
        
        basis_size = basis_size + num_basis[res]
    
    return phi

def WendlandTransformedCoordinates(data, parameters, regio=None):
    '''
    The function applies Wendland Transform to the input data set.
    It takes the data, parameters, and regio as input.
    It calculates the normalized coordinates based on 'x' and 'y' columns.
    It calls the WTC_kernel function to calculate the kernel values.
    It returns the data set with the added transformed coordinates.
    '''
    # Get the value of 'n' parameter for WTC
    n = parameters['WTC']
    
    # Normalize 'x' and 'y' coordinates
    Rechts = (data['x'] - np.min(data['x'])) / (np.max(data['x']) - np.min(data['x']))
    Hoch = (data['y'] - np.min(data['y'])) / (np.max(data['y']) - np.min(data['y']))
    
    # Calculate WTC kernel
    WTC = WTC_kernel(Rechts, Hoch, n)
    
    # Create a modified data set by joining the WTC matrix with the original data
    data_mod = data
    data_mod_1 = pd.concat([data_mod.reset_index(), pd.DataFrame(WTC, columns=np.arange(WTC.shape[1]).astype('str'))], axis=1)
    
    # Convert column names to string type
    data_mod_1.columns = data_mod_1.columns.astype(str)
    
    return data_mod_1

def EuclideanDistanceFields(data, parameters, regio=None):
    '''
    Calculates the Euclidean distances between each point in the 'x' and 'y' columns of the input DataFrame
    and the four corners (upper left, upper right, bottom left, and bottom right) and the center 
    of the study area. The distances are added as new columns 'NW', 'NE', 'SW', 'SE', and 'C' to the input data DataFrame, and the resulting DataFrame is returned. The function is optimized by using vectorized operations from NumPy and avoiding redundant calculations.
    '''
    # Calculate minimum and maximum coordinates
    xmin, ymin = data[['x', 'y']].min()
    xmax, ymax = data[['x', 'y']].max()
    
    # Calculate the mean coordinates
    xmid = data.x.mean()
    ymid = data.y.mean()
    
    # Convert 'x' and 'y' columns to numpy array
    points = data[['x', 'y']].values
    
    # Define the four corners and center of the study area
    SW = np.array([xmin, ymin])
    SE = np.array([xmax, ymin])
    NE = np.array([xmax, ymax])
    NW = np.array([xmin, ymax])
    C = np.array([xmid, ymid])
    
    # Calculate Euclidean distances for each point and corner/center
    dm_SW = np.linalg.norm(points - SW, axis=1)
    dm_SE = np.linalg.norm(points - SE, axis=1)
    dm_NE = np.linalg.norm(points - NE, axis=1)
    dm_NW = np.linalg.norm(points - NW, axis=1)
    dm_C = np.linalg.norm(points - C, axis=1)
    
    # Create a DataFrame with the distances
    data_EDF = data.assign(SW=dm_SW, SE=dm_SE, NE=dm_NE, NW=dm_NW, C=dm_C)
    
    # Check if EDF normalization is enabled
    if parameters.get('EDF', False):
        # Normalize each column separately
        for col in ['SW', 'SE', 'NE', 'NW', 'C']:
            max_val = np.max(data_EDF[col])
            if max_val > 0:
                data_EDF[col] = data_EDF[col] / max_val
    
    return data_EDF

def EuclideanDistanceMatrix(data, parameters, regio=None):
    '''
    The function calculates the Euclidean distance between each pair of points in the 'x' 
    and 'y' columns of the input 'data' DataFrame and adds the resulting distance matrix or PCA transformed 
    matrix as new columns to the 'data' DataFrame. If the 'EDM' parameter is True, PCA transformation is 
    applied to the distance matrix with the specified number of components. The final result is a DataFrame with 
    the original columns of 'data' and the distance matrix/PCA transformed matrix.
    '''
    # Convert 'x' and 'y' columns to numpy array
    points_coords = data[['x', 'y']].to_numpy()
    
    # Check if regio is provided for distance calculation
    if regio is not None:
        points1_coords = regio[['x', 'y']].to_numpy()
        dm = cdist(points_coords, points1_coords, metric='euclidean')
    else:
        dm = cdist(points_coords, points_coords, metric='euclidean')
    
    # Check if EDM normalization is enabled
    if parameters.get('EDM', False):
        dm = dm / np.max(dm)
    
    # Create a DataFrame with the distance matrix
    result = np.hstack([data, dm])
    result = pd.DataFrame(result, columns=data.columns.tolist() + ['dist_' + str(i) for i in range(dm.shape[1])])
    
    return result

def PrincipleComponentAnalysis(data, parameters, regio=None):
    '''
    This implementation uses the PCA class from the sklearn.decomposition module to perform PCA on
    the input data. The fit_transform method is used to fit the PCA model to the input data and to
    obtain the transformed data in a lower-dimensional space. The n_components argument specifies
    the number of dimensions in the reduced space, with a default value of 2 for easy visualization.
    '''
    normalize = True
    n_components = int(parameters['PCA'])
    
    # Convert 'x' and 'y' columns to numpy array
    points_coords = data[['x', 'y']].to_numpy()
    
    # Check if regio is provided for distance calculation
    if regio is not None:
        points1_coords = regio[['x', 'y']].to_numpy()
        dm = cdist(points_coords, points1_coords, metric='euclidean')
    else:
        dm = cdist(points_coords, points_coords, metric='euclidean')
    
    # Normalize the distance matrix if enabled
    if normalize:
        dm = StandardScaler().fit_transform(dm)
    
    # Perform PCA transformation
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(dm)
    
    # Create a DataFrame with the transformed data
    PC = pd.DataFrame(transformed_data)
    data_PCA = data.join(PC)
    
    # Convert column names to string type
    data_PCA.columns = data_PCA.columns.astype(str)
    
    return data_PCA

def EigenvectorSpatialFiltering(data, parameters, regio=None):
    '''
    The function applies Eigenvector Spatial Filtering (ESF) to the input data set.
    It takes the data, parameters, and regio as input.
    It calculates the Euclidean distances between each pair of points.
    It performs Singular Value Decomposition (SVD) on the distance matrix.
    It selects the first K eigenvectors.
    It returns the data set with the added eigenvectors.
    '''
    # Get parameters for ESF
    dmax, k = parameters['ESF']
    
    # Convert 'x' and 'y' columns to numpy array
    points_EV = data[['x', 'y']].to_numpy()
    
    # Check if regio is provided for distance calculation
    if regio is not None:
        points1_coords = regio[['x', 'y']].to_numpy()
        dm_EV = cdist(points_EV, points1_coords, metric='euclidean')
    else:
        dm_EV = cdist(points_EV,points_EV, metric='euclidean')
    
    # Set distances greater than dmax to dmax * 4
    dm_EV = np.asarray(dm_EV)
    dm_EV[dm_EV > dmax] = dmax * 4
    
    # Compute SVD of the distance matrix
    U, s, Vt = np.linalg.svd(dm_EV, full_matrices=False)
    
    # Select the first K eigenvectors
    eigvec = U[:, :k]
    
    # Convert eigenvectors array to a DataFrame
    eigvec_df = pd.DataFrame(data=eigvec, index=data.index, columns=[f'eigvec_{i+1}' for i in range(k)])
    
    # Join the eigenvector DataFrame with the original data DataFrame
    data_eigvec = pd.concat([data, eigvec_df], axis=1)
    
    return data_eigvec

# Mapping of variables to their corresponding functions
functions = {
    # Coordinate Based Methods
    'PGC': PolynomialGeographicCoordinates,
    'OGC': ObliqueGeographicCoordinates,
    # Distance Based Methods
    'WTC': WendlandTransformedCoordinates,
    'EDF': EuclideanDistanceFields,
    'EDM': EuclideanDistanceMatrix,
    'PCA': PrincipleComponentAnalysis,
    'ESF': EigenvectorSpatialFiltering,  
}