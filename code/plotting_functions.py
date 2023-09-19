# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:58:21 2023

@author: Marc
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
from libpysal.weights import KNN, Queen
from esda.moran import Moran, Moran_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster
from splot.esda import moran_scatterplot, lisa_cluster
import shap

#%%

#get_cmap, plot_target_concentration, moran_i_analysis, plot_moran_analysis, plot_moran_residual_analysis, generate_shap_plots, regio_plot


def get_cmap():
    # Define a list of colors
    colors = ['#292C5B', '#4575b4', '#74add1', '#90D8F0', '#B1F2E7',
              '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026', '#800000',
              '#4d0000', '#330000', '#55013C', '#5C0377']  #, '#6705CF']
    # Create a colormap from the list of colors
    return LinearSegmentedColormap.from_list('my_cmap', colors, N=160)


def plot_target_concentration(gdf, ms):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    # Set the title and column label
    title = "Nitrate observations in Baden-Württemberg"
    column = "Nitrate concentration [mg/L]"
    # Convert the GeoDataFrame to EPSG 3857 projection
    gdf = gdf.to_crs(epsg=3857)
    # Load the boundary shapefile for Baden-Württemberg
    border_BW = gpd.read_file('/vsicurl/https://github.com/stepheneb/jsxgraph/raw/master/examples/Cartography/vg2500_bld.shp')
    border_BW = border_BW.to_crs(epsg=32632)
    poly = border_BW[border_BW['GEN'] == 'Baden-Württemberg']
    cmap = get_cmap()

    # Convert the boundary shapefile to EPSG 3857 projection
    poly = poly.to_crs(epsg=3857)
    poly.boundary.plot(ax=ax, color="black")
    # Set the buffer and bounds
    buffer = 3000  # m
    xmin, ymin, xmax, ymax = poly.total_bounds
    # Convert the GeoDataFrame to EPSG 3857 projection
    gdf = gdf.to_crs(epsg=3857)
    # Plot the target column with the specified colormap and markersize
    gdf.plot(column='target', cmap=cmap, markersize=ms, edgecolor='k',
             ax=ax, legend=True, legend_kwds={'label': column, 'orientation': "vertical"})
    # Set the title, x-label, and y-label
    ax.set_title(title)
    ax.set_xlabel('Easting (X) [m]')
    ax.set_ylabel('Northing (Y) [m]')
    # Annotate the north direction
    ax.annotate('N', xy=(0.97, 0.97), xytext=(0.97, 0.97 - 0.05),
                arrowprops=dict(facecolor='black', width=3, headwidth=10),
                ha='center', va='center', fontsize=15,
                xycoords=ax.transAxes)
    # Set the x and y limits with buffer
    ax.set_xlim(xmin - buffer, xmax + buffer)
    ax.set_ylim(ymin - buffer, ymax + buffer)
    # Add basemaps
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerBackground, alpha=0.5)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TerrainBackground, alpha=0.45)
    # Show the plot
    plt.show()


def moran_i_analysis(df, k, point_size):
    # Create spatial weights using k-nearest neighbors
    w_kNN = KNN.from_dataframe(df, k=k, distance_metric='euclidean')
    w_kNN.transform = 'r'
    # Calculate Moran's I
    moran = Moran(df['target'], w_kNN)
    p_value = moran.p_sim
    # Plot Moran scatterplot
    plot_moran(moran, scatter_kwds={'s': point_size})
    plt.show()
    return p_value


def plot_moran_analysis(gdf, p, ms):
    # Extract target values from the GeoDataFrame
    y = gdf['target'].values
    # Create spatial weights using Queen contiguity
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Load boundary shapefile for Baden-Württemberg
    border_BW = gpd.read_file('/vsicurl/https://github.com/stepheneb/jsxgraph/raw/master/examples/Cartography/vg2500_bld.shp')
    border_BW = border_BW.to_crs(epsg=32632)
    poly = border_BW[border_BW['GEN'] == 'Baden-Württemberg']
    # Compute local Moran's I
    moran_loc = Moran_Local(y, w)
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    # Figure 1: Scatterplot of Local Moran's I
    moran_scatterplot(moran_loc, p=p, ax=axes[0], aspect_equal=True, scatter_kwds={'s': ms, 'alpha': 0.7})
    axes[0].set_xlabel("Nitrate")
    axes[0].set_ylabel('Spatial Lag of Nitrate')
    # Figure 2: LISA Cluster
    lisa_cluster(moran_loc, gdf, p=p, ax=axes[1], legend=True, markersize=ms, alpha=0.7)
    poly.boundary.plot(ax=axes[1], color="black")
    # Display the subplots
    plt.tight_layout()
    plt.show()


def plot_moran_residual_analysis(gdf, CV_LOO, p, ms):
    # Create a GeoDataFrame with the geometry column
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:32632')

    # Extract target values from the GeoDataFrame
    y = gdf["Resid"].astype(float).values
    # Create spatial weights using Queen contiguity
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'
    # Load boundary shapefile for Baden-Württemberg
    border_BW = gpd.read_file('/vsicurl/https://github.com/stepheneb/jsxgraph/raw/master/examples/Cartography/vg2500_bld.shp').to_crs(epsg=32632)
    border_BW = border_BW.to_crs(epsg=32632)
    poly = border_BW[border_BW['GEN'] == 'Baden-Württemberg']
    # Compute local Moran's I
    moran_loc = Moran_Local(y, w)
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    # Figure 1: Scatterplot of Local Moran's I
    moran_scatterplot(moran_loc, p=p, ax=axes[0], aspect_equal=False, scatter_kwds={'s': ms, 'alpha': 0.7})
    axes[0].set_xlabel("Nitrate")
    axes[0].set_ylabel('Spatial Lag of  Nitrate')
    # Figure 2: LISA Cluster
    lisa_cluster(moran_loc, gdf, p=p, ax=axes[1], legend=True, markersize=ms, alpha=0.7)
    poly.boundary.plot(ax=axes[1], color="black")
    # Compute global Moran's I
    moran_global = Moran(y, w)
    # Display the subplots
    plt.tight_layout()
    plt.show()
    return moran_loc.Is, moran_global


def generate_shap_plots(explainer, shap_values, X_scaled, variable_str):
    # Violinplot
    shap.summary_plot(shap_values, X_scaled, max_display=14, show=False)
    plt.show()
    plt.close()

    # Barplot
    shap.summary_plot(shap_values, X_scaled, plot_type='bar', max_display=14, show=False)
    plt.show()
    plt.close()


def regio_plot(gdf, variable_str):
    sns.set_style("white")
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(14, 10))
    border_BW = gpd.read_file('/vsicurl/https://github.com/stepheneb/jsxgraph/raw/master/examples/Cartography/vg2500_bld.shp')
    border_BW = border_BW.to_crs(epsg=32632)
    poly = border_BW[border_BW['GEN'] == 'Baden-Württemberg']

    # Convert the boundary shapefile to EPSG 3857 projection
    poly = poly.to_crs(epsg=3857)
    poly.boundary.plot(ax=ax, color="black")
    # Convert the GeoDataFrame to EPSG 3857 projection
    gdf_wm = gdf.to_crs(epsg=3857)

    # Set the colormap
    cmap = get_cmap()
    norm = colors.Normalize(vmin=0, vmax=70)
    nitrat_plot = gdf_wm.plot(ax=ax, column='Nitrat', marker='s',
                              edgecolors='none', cmap=cmap, norm=norm, markersize=1, alpha=0.85, legend=True)

    # North arrow
    x1, y1, arrow_length = 0.05, 0.97, 0.065
    ax.annotate('N', xy=(x1, y1), xytext=(x1, y1 - arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)

    # Basemap
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerBackground, alpha=0.5)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TerrainBackground, alpha=0.45)

    vstr = variable_str.replace("False", "").replace("True", "")

    ax.set_title(f"Modelled Nitrate-concentration [mg/l], {vstr}", fontsize=13)
    plt.show()