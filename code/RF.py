# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:46:55 2023
@author: Marc Ohmer
"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
import shap

# Importing function modules
from general_functions import read_data, preprocess, generate_variable_string
from plotting_functions import (
    plot_target_concentration,
    plot_moran_analysis,
    generate_shap_plots,
    regio_plot,
)

#%% Spatial functions setting

# =============================================================================
# 1. Definition of spatial information used for model prediction
# =============================================================================

# Variables indicating whether to load the respective function
# Multiple selection possible True --> We be used as spatial covariate

variables = {
    'GC': False, # Geographic coordinates
    'PGC': True, # Polynomial Geographic coordinates
    'OGC': False, # Oblique Geographic coordinates
    'WTC': False, # Wendand Transformed Coordinates
    'EDF': False, # Euclidean Distance Fields
    'EDM': False, # Euclidean Distance Matrix
    'PCA': False, # Principle Component Analysis
    'ESF': False, # Eigenvector Spatial Filtering
}

# Parameters for the functions
parameters = {
    'OGC': (0, 3, 181),# start, stepsize, stop in degree:(0, 6, 181)
    'WTC': int(12), # number and size of basis functions used in the transformation
    'EDF': (False), #If True EDF values will be normalized
    'EDM': (True), # If True EDM values will be normalized
    'PCA': int(28), #No. of used Principle components (if 3: Only the 3 most important PCA will be used
    'ESF': (48, 17), # Search distance in km, number of eigenvalues used
}

#%% # Read data and plot data 
# =============================================================================
# 2.1. Load Nitrate measurements and covariates at monitoring well locations
# =============================================================================

# Read the Example data 
df, gdf = read_data("point")

# Plot target concentration
plot_target_concentration(gdf, ms=12)

# Perform Moran analysis
plot_moran_analysis(gdf, p=0.05, ms=12)

#%%Create Spatial Information Features

# Start the timer
start_time = time.time()

# =============================================================================
# 2.2. Adding the spatial information selected in 1. to the dataframe
# All spatial information feature functions can be found in 
# spatial_feature_functions.py
# =============================================================================

# Preprocess the data
data = preprocess(df, variables, parameters)

# Generate the variable string
variable_str = generate_variable_string(variables, parameters)

# =============================================================================
# 2.3. Encoding & Scaling
# =============================================================================

# Identify categorical variables (marked with asterisk in column name)
categorical_variables = [col for col in data.columns if '*' in col]

# Convert identified categorical variables to object type
data[categorical_variables] = data[categorical_variables].astype('category')

# Fit Leave-One-Out Encoder on categorical variables
encoder = LeaveOneOutEncoder().fit(data[categorical_variables], data['target'])


# Apply encoder transformation and concatenate with non-categorical columns
data_enc = pd.concat([data.drop(columns=categorical_variables), encoder.transform(data[categorical_variables])], axis=1)\
               .loc[:, data.columns]\
               .reset_index(drop=True)


#%%
# Extract the target variable and the input features
Y = data_enc['target'].values
X = data_enc.drop(columns=['target'])

# Fit StandardScaler on the input features
scaler = StandardScaler().fit(X)

# Scale the input features using the fitted scaler
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# =============================================================================
# 2.4 CV and Model
# =============================================================================

# Set the number of splits for cross-validation
n_splits = 10

# Set CV_LOO to True for Leave-One-Out cross-validation, or False for K-Fold cross-validation
CV_LOO = False

# Create a RandomForestRegressor model with 100 estimators
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define the cross-validation strategy based on CV_LOO
if CV_LOO:
    cv = LeaveOneOut()  # Use Leave-One-Out cross-validation
else:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Use K-Fold cross-validation

# Perform cross-validation and calculate metrics
mae = -cross_val_score(model, X_scaled, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
mse = -cross_val_score(model, X_scaled, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
cv_pred = cross_val_predict(model, X_scaled, Y, cv=cv)
RMSE = np.sqrt(np.mean(mse))
r2 = r2_score(Y, cv_pred)
bias = np.mean(Y - cv_pred)

print(f"Spatial Information: {variable_str}")
print(f"Mean Absolute Error (MAE): {np.mean(mae):.3f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.3f}")
print(f"Bias: {bias:.3f}")
print(f"R²-Score: {r2:.3f}")


#%% SHAP values

# =============================================================================
# 2.5 Computation of the feature importance of the covariates and the 
#     spatial information with shap plots
# =============================================================================

# Fit the model to the scaled input features and target variable
model.fit(X_scaled,Y)
# Create a TreeExplainer object with the trained model
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for the input features using the TreeExplainer
shap_values = shap.TreeExplainer(model).shap_values(X_scaled, check_additivity=True)
# Generate SHAP plots
generate_shap_plots(explainer, shap_values, X_scaled, variable_str)

#%% Regionalization

# =============================================================================
# 3.1 Loading covariates in a grid of a defined solution. In this example, 
# the covariates were read out in a rectangular point grid (point spacing = 300m).
# =============================================================================

# Read data from the "grid" source
df1, gdf1 = read_data("grid")

#%%Create Spatial Information Features

# =============================================================================
# 3.2. Adding the spatial information selected in 1. to the pointgrid dataframe
# =============================================================================

# Preprocess the data
data1 = preprocess(df1, variables, parameters,True)

# =============================================================================
# 3.3. Encoding & Scaling
# =============================================================================

# Target Encoding
data1_enc = data1.copy()

# Perform target encoding for each categorical variable
for k in categorical_variables:
    # Map each category to the mean target value for that category
    data1_enc[k] = data1[k].map(data.groupby(k)['target'].mean())

# Drop rows with missing values
data1_enc.dropna(inplace=True)

# Prepare and scale input data
X_pred = data1_enc.copy()

# Scale the input data using the fitted scaler
X_pred_scaled = scaler.transform(X_pred)

# Create a DataFrame with the scaled data
X_pred_scaled = pd.DataFrame(X_pred_scaled)

# Set the column names and index of the scaled data to match X_pred
X_pred_scaled.columns = X_pred.columns
X_pred_scaled.index = X_pred.index

# =============================================================================
# 3.4 CV and Model
# =============================================================================

# Create a RandomForestRegressor model
model = RandomForestRegressor(verbose=0, n_jobs=20)

# Fit the model to the scaled input features (X_scaled) and target variable (Y)
model.fit(X_scaled, Y)

# Make predictions on the scaled input data (X_pred_scaled)
predicted = pd.DataFrame(model.predict(X_pred_scaled), columns=['Nitrat'])

# Set the index of the predictions DataFrame to match the index of X_pred
predicted.index = X_pred.index

# Merge the predictions with the data1_enc DataFrame based on the index
predicted = data1_enc.merge(predicted, left_index=True, right_index=True, how='right')

# Merge the predictions DataFrame with the geometry column of the gdf1 GeoDataFrame based on the index
predicted = predicted.loc[:, ['Nitrat']].merge(gdf1.geometry, left_index=True, right_index=True, how='left')

# Create a GeoDataFrame using the predicted DataFrame, specifying the geometry column and CRS
predicted = gpd.GeoDataFrame(predicted, geometry='geometry', crs='EPSG:32632')

# Print a message indicating that the regionalization process has finished
print('Finished Regionalization....')

# =============================================================================
# 3.5 Plot the predicted map
# =============================================================================

# Plot the spatial prediction
regio_plot(predicted, variable_str)