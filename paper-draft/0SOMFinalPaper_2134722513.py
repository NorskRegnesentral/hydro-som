import itertools
import re
from pathlib import Path

import minisom
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shapely

import plotly.colors as pc
import matplotlib.pyplot as plt

from plotly.colors import sample_colorscale, DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from IPython.display import Image
from scipy.interpolate import griddata


# --------------------------------------------------------
# Input and settings
# --------------------------------------------------------

# data_path = Path('H:\\Python\\Self_Org_Map\\As_Progr_for-Som\\A0_Es_Final_2\\a_data_Paper.csv')
data_path = Path(__file__).parent / 'a_data_Paper.csv'
input_columns = ['F4','F5','F7','F8','F9', 'pH_L']

som_nx = None  # Set to None to determine these automatically (PCA)
som_ny = None
#________________________________________________

# Exp 3 clusters with QE = 0.13 and TE = 0.0
som_sigma = 2.4
som_learning_rate = 0.37
som_random_seed = 10
som_max_iterations = 6000

som_activation_distance = 'euclidean'
som_topology = 'hexagonal'
som_neighborhood_function = 'gaussian'

max_clusters = 10  # Maximum number of clusters to test
number_of_clusters = None  # Set to None to pick the optimal number automatically



# Visualization
#features_colorscale = 'spectral'
features_colorscale = 'RdBu'
pio.templates.default = 'plotly_white'
pio.renderers.default = 'browser' # Set to browser if not already
plotly_interactive = True

colorbar_tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
colorbar_tickformat = ".2f"

show_cluster_sample_names = True

# Isoline configuration  ─────────────────────────────────────────────────────
# Background contour levels drawn per feature (normalised weight-space values)
isoline_config = {
    'F4': [0.2, 0.25, 0.36, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
    'F5': [0.1, 0.2, 0.3,  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    'F7': [0.1, 0.2, 0.25, 0.3, 0.32, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9],
    'F8': [0.1, 0.2, 0.3, 0.35, 0.44, 0.5, 0.55, 0.6, 0.65, 0.7],
    'F9': [0.1, 0.2, 0.25, 0.3, 0.4, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75],
}

# Highlighted (coloured) isoline values – drawn on top of the green background
isoline_colors = {
    #'F3': {0.38: 'darkorange'},
    'F4': {0.31: 'darkorange'},
    'F5': {0.59: 'darkorange', 0.72: 'darkorange'},
    'F7': {0.14: 'darkorange'},
    'F8': {0.37: 'darkorange'},
    'F9': {0.48: 'darkorange'},
}

# Dash styles for the highlighted isolines
isoline_styles = {
    #'F3': {0.38: 'solid'},
    'F4': {0.31: 'solid'},
    'F5': {0.59: 'solid', 0.72: 'dashdot'},
    'F7': {0.14: 'solid'},
    'F8': {0.37: 'solid'},
    'F9': {0.48: 'solid'},
}


# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------


# Convenience function to allow changing between interactive and static images
def show_figure(fig: go.Figure):
    if plotly_interactive:
        fig.show()
    else:
        return Image(data=fig.to_image(format='png', scale=2))


def apply_normalization(arr):
    data_min = arr.min(axis=0)
    data_range = arr.max(axis=0) - data_min

    return (arr - data_min) / data_range


# --------------------------------------------------------
# Data processing and SOM training
# --------------------------------------------------------

# Read input data
df = pd.read_csv(data_path)
input_array = df[input_columns].values
transformed_data = apply_normalization(input_array)

# Derive SOM dimensions
if som_nx is None or som_ny is None:
    recommended_nodes = 5 * np.sqrt(len(df))
    print(f"Recommended number of nodes: {recommended_nodes:.0f}")
   
    # "Naive" approach: square root of the number of samples (with round-off)
    # final_nx = np.round(np.sqrt(som_m)).astype(int)
    # final_ny = np.round(som_m / som_nx).astype(int)

    # Perform PCA analysis
    pca = PCA(n_components=2)
    pca.fit(transformed_data)

    # Calculate the ratio between the first two principal components
    pca_ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
    final_nx = np.round(np.sqrt(recommended_nodes * pca_ratio)).astype(int)
    final_ny = np.round(recommended_nodes / final_nx).astype(int)
    print(f"Using PCA to determine SOM dimensions: {final_nx} x {final_ny} (PCA ratio: {pca_ratio:.2f})")
else:
    final_nx, final_ny = som_nx, som_ny
    print(f"Using fixed SOM dimensions: {final_nx} x {final_ny}")

# Print all principal components values
print(f"Principal component values: {pca.explained_variance_}")

# Set up model
som = minisom.MiniSom(
    final_nx,
    final_ny,
    len(input_columns),
    sigma=som_sigma,
    learning_rate=som_learning_rate,
    activation_distance=som_activation_distance,
    topology=som_topology,
    neighborhood_function=som_neighborhood_function,
    random_seed=som_random_seed,
)

# Train
som.train(transformed_data, som_max_iterations)

# Extract data
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()
win_map = som.win_map(transformed_data, return_indices = True)

# Per-feature normalisation of weights to the 0-1 interval.
# Many downstream operations (isolines, colour mapping, PC-neuron maps,
# statistics) work in this normalised space.  The original `weights` array
# is kept unchanged for tasks that need the transformed-data scale (e.g.
# hierarchical clustering).
weights_normalized = np.empty_like(weights)
for _k in range(weights.shape[2]):
    _wk = weights[:, :, _k]
    _wk_min, _wk_max = _wk.min(), _wk.max()
    weights_normalized[:, :, _k] = (_wk - _wk_min) / (_wk_max - _wk_min)

# Print quantization and topographic errors for the trained model
print(f"Quantization error: {som.quantization_error(transformed_data)}")
print(f"Topographic error : {som.topographic_error(transformed_data)}")

# Range of SOM grid sizes to test
grid_sizes = range(2, 14)

# Setup input to clustering and dendrogram plot
mij = np.meshgrid(np.arange(final_nx), np.arange(final_ny), indexing='ij')
cells_ij = np.column_stack((mij[0].flat, mij[1].flat))
labels = [(i, j) for i, j in cells_ij]
flat_weights = weights.reshape(-1, len(input_columns)) 

# --------------------------------------------------------
# Clustering and dendrogram
# --------------------------------------------------------

# Calculate link
links = hierarchy.linkage(flat_weights, method='complete', metric='euclidean')

# Clustering
cluster_results = []
for i in range(2, max_clusters):
    flat_labels = hierarchy.fcluster(links, i, 'maxclust')
    sil = silhouette_score(flat_weights, flat_labels) if len(set(flat_labels)) == i else None
    cluster_results.append({
        "Number of clusters": i,
        "Silhouette": sil,
        "labels": flat_labels,
    })

cluster_df = pd.DataFrame.from_dict(cluster_results)

sil_optimal_cluster = cluster_df.iloc[cluster_df['Silhouette'].idxmax()]

if number_of_clusters is None:
    selected_cluster = sil_optimal_cluster
else:
    selected_cluster = cluster_df[cluster_df['Number of clusters'] == number_of_clusters].iloc[0]

print(f"Optimal number of clusters based on Silhouette : {sil_optimal_cluster['Number of clusters']}")

print()

if number_of_clusters is None:
    print(f"Number of clusters picked based on Silhouette     : {selected_cluster['Number of clusters']}")
    final_n_clusters = selected_cluster['Number of clusters']
else:
    print(f"Number of clusters fixed to                    : {number_of_clusters}")
    final_n_clusters = number_of_clusters

fig = px.line(cluster_df, x='Number of clusters', y='Silhouette', markers=True)
fig.layout.title = "Silhouette score (higher is better)"
fig.layout.yaxis.spikesnap = "hovered data"
fig.layout.yaxis.spikemode = "across"
fig.layout.yaxis.spikethickness = 2
show_figure(fig)

# Optimal number of lables
node_labels = selected_cluster['labels'].reshape(final_nx, final_ny)

# Sample labels
sample_ij = [None] * len(df)
sample_label = [-1] * len(df)
for key, value in win_map.items():
    for ix in value:
        sample_ij[ix] = key
        sample_label[ix] = node_labels[key]

df['ij'] = sample_ij
df['cluster_label'] = sample_label

# Print samples in each cluster
n_cols = 8
for lab, sub_df in df.groupby('cluster_label'):
    print(f"Cluster {lab}: {len(sub_df)} samples")
    for i, s in enumerate(sub_df['Sample']):
        print(f"  {s:19}", end="")
        if i % n_cols == n_cols - 1:
            print()
    print("\n")

# --------------------------------------------------------
# Utility functions for visualization
# --------------------------------------------------------

def hexagon(_xx, _yy, radius):
    # Returns a NX x NY x 6 x 2 array of hexagon coordinates
    hex_xy = np.zeros((_xx.shape[0], _xx.shape[1], 7, 2))
    for i in range(7):
        theta = np.pi / 6 + 2 * np.pi * i / 6
        hex_xy[:, :, i, 0] = _xx + radius * np.cos(theta)
        hex_xy[:, :, i, 1] = _yy + radius * np.sin(theta)
    return hex_xy


def add_cluster_boundaries(fig: go.Figure, row_index, col_index):
    for k, boundary in cluster_boundaries.items():
        
        if boundary.geom_type == 'Polygon':
            bnds = [boundary]
        elif boundary.geom_type == 'MultiPolygon':
            bnds = list(boundary.geoms)
        else:
            raise NotImplementedError(boundary.geom_type)
        
        for poly in bnds:
            if poly.boundary.geom_type == 'LineString':
                crds = np.array(poly.boundary.coords)
            else:
                crds = np.array(poly.boundary.geoms[0].coords)

            fig.add_scatter(
                x=crds[:, 0],
                y=crds[:, 1],
                # fill='toself',
                mode="lines",
                showlegend=False,
                line=dict(color='black', width=2),
                row=row_index,
                col=col_index,
            )


def add_som_features(fig: go.Figure, row, col, feature_index, colorbar_settings):
    if feature_index >= weights_normalized.shape[2]:
        w = umatrix.copy()
        # Normalise u-matrix to 0-1 for colour mapping
        w_min, w_max = w.min(), w.max()
        w_01 = (w - w_min) / (w_max - w_min)
    else:
        # Use pre-computed normalised weights (0-1 per feature)
        w_01 = weights_normalized[:, :, feature_index]
        w = weights[:, :, feature_index]
        w_min, w_max = w.min(), w.max()

    colors = np.array(
        sample_colorscale(features_colorscale, w_01.flatten())
    ).reshape(final_nx, final_ny)


    # Add colored hexagons
    for i in range(final_nx):
        for j in range(final_ny):
            text = str(sorted(df.iloc[win_map[(i, j)]].index.values.tolist()))

            fig.add_scatter(
                x=hexagons[i, j, :, 0],
                y=hexagons[i, j, :, 1],
                fill='toself',
                mode='lines',
                fillcolor=colors[i, j],
                line=dict(color='black', width=0),
                text=text,
                showlegend=False,
                hoverinfo="text",
                hoveron="fills",
                row=row,
                col=col,
            )

    # Add invisible scatter for colourbar (with original weights for correct tick values)
    fig.add_scatter(
        x=[0, 0],
        y=[0, 0],
        showlegend=False,
        marker=dict(
            color=[w_min, w_max],
            colorscale=features_colorscale,
            size=0.001,
            colorbar=colorbar_settings
        ),
        row=row,
        col=col,
    )


def add_training_cells(fig: go.Figure, row, col, show_sum = True):
    # Add colored hexagons
    label_colors = DEFAULT_PLOTLY_COLORS
    for i in range(final_nx):
        for j in range(final_ny):
            text = ""
            if len(win_map[(i, j)]) > 0:
                if show_sum:
                    text = f"{len(win_map[(i, j)])}"
                else:
                    if "Sample" in df.columns and not show_cluster_sample_names:
                        names = df.iloc[win_map[(i, j)]]["Sample"]
                    else:
                        names = df.iloc[win_map[(i, j)]].index
                    samples = [str(s) for s in sorted(names.values.tolist())]
                    text = f"{'<br>'.join(samples)}"

            fig.add_scatter(
                x=hexagons[i, j, :, 0],
                y=hexagons[i, j, :, 1],
                fill='toself',
                mode='lines',
                fillcolor=label_colors[node_labels[i, j]],
                line=dict(width=0),
                text=text,
                showlegend=False,
                hoverinfo="text",
                hoveron="fills",
                row=row,
                col=col,
            )

            mid = (hexagons[i, j, 0, :] + hexagons[i, j, 3, :]) / 2
            fig.add_scatter(
                x=[mid[0]],
                y=[mid[1]],
                mode="text",
                text=text,
                textfont=dict(color="black", size=10),
                marker=dict(color='black'),
                showlegend=False,
                hoverinfo="none",
                row=row,
                col=col,
            )


def add_isolines_to_som(fig, som_weights, coords, feature_name, isoline_values, row, col,
                        color_map=None, style_map=None):
    """
    Add contour isolines to a SOM feature plane.

    Parameters
    ----------
    fig            : plotly Figure
    som_weights    : 2-D array  weight matrix for one feature
    coords         : (xx, yy) euclidean coordinate arrays
    feature_name   : str (informational only)
    isoline_values : list of float  contour levels to draw
    row, col       : subplot position
    color_map      : dict {value: color}  per-value colour (default: 'darkgreen')
    style_map      : dict {value: dash}   per-value dash   (default: 'dash')
    """
    xx, yy = coords
    points = np.column_stack([xx.flatten(), yy.flatten()])
    values = som_weights.flatten()
    grid_x = np.linspace(xx.min(), xx.max(), 100)
    grid_y = np.linspace(yy.min(), yy.max(), 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_z = griddata(points, values, (grid_xx, grid_yy), method='cubic')

    for isoline_val in isoline_values:
        color = (color_map or {}).get(isoline_val, 'darkgreen')
        dash  = (style_map or {}).get(isoline_val, 'solid')
        width = 4.0 if color_map and isoline_val in color_map else 1.0

        fig_temp = plt.figure()
        ax_temp  = fig_temp.add_subplot(111)
        contours = ax_temp.contour(grid_xx, grid_yy, grid_z, levels=[isoline_val])
        plt.close(fig_temp)

        if not contours.allsegs:
            continue
        for seg in contours.allsegs[0]:
            fig.add_scatter(
                x=seg[:, 0], y=seg[:, 1],
                mode='lines',
                line=dict(color=color, width=width, dash=dash),
                showlegend=False,
                hoverinfo='skip',
                row=row, col=col,
            )


def add_pc1_isolines(fig, coords, pc1_neuron_map, pc1_isoline_value, row, col):
    """
    Draw the mean-PC1 isoline (red dotted) on one SOM subplot.

    Parameters
    ----------
    fig               : plotly Figure
    coords            : (xx, yy) euclidean coordinate arrays
    pc1_neuron_map    : 2-D array  per-neuron PC1 scores
    pc1_isoline_value : float     contour level (typically the map mean)
    row, col          : subplot position
    """
    xx, yy = coords
    points  = np.column_stack([xx.flatten(), yy.flatten()])
    values  = pc1_neuron_map.flatten()
    grid_x  = np.linspace(xx.min(), xx.max(), 100)
    grid_y  = np.linspace(yy.min(), yy.max(), 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_z  = griddata(points, values, (grid_xx, grid_yy), method='cubic')

    fig_temp = plt.figure()
    ax_temp  = fig_temp.add_subplot(111)
    contours = ax_temp.contour(grid_xx, grid_yy, grid_z, levels=[pc1_isoline_value])
    plt.close(fig_temp)

    if len(contours.allsegs) > 0:
        for path in contours.allsegs[0]:
            fig.add_scatter(
                x=path[:, 0], y=path[:, 1],
                mode='lines',
                line=dict(color='red', width=4.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
                row=row, col=col,
            )


def add_pc2_isolines(fig, coords, hexagons, feature_name, row, col):
    """
    Add PC2 isoline to a SOM plane based on PC2 neuron values.
    This ensures the same geometric line across all feature plots.
    """
    xx, yy = coords
    
    # Create interpolated grid for smoother contours using PC2 neuron map
    points = np.column_stack([xx.flatten(), yy.flatten()])
    values = pc2_neuron_map.flatten()
    
    # Create fine grid
    grid_x = np.linspace(xx.min(), xx.max(), 100)
    grid_y = np.linspace(yy.min(), yy.max(), 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
    # Interpolate PC2 values
    grid_z = griddata(points, values, (grid_xx, grid_yy), method='cubic')
    
    # Generate contour for PC2 isoline value
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    contours = ax_temp.contour(grid_xx, grid_yy, grid_z, levels=[pc2_isoline_value], colors='black', linewidths=2)
    plt.close(fig_temp)
    
    # Extract contour paths and add to plotly figure
    if len(contours.allsegs) > 0:
        for path in contours.allsegs[0]:
            fig.add_scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode='lines',
                line=dict(color='darkblue', width=5.0, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
                row=row,
                col=col
            )

# -------------------------------------------------------
# Generate plots
# -------------------------------------------------------

hexagons = hexagon(xx, yy, 0.5)

# Cluster polygons
node_hexagons = hexagon(xx, yy, np.sqrt(3) / 3 + 0.001).reshape(-1, 7, 2)
cluster_boundaries = {}
for c in np.unique(node_labels):
    matches = node_labels.flatten() == c
    joined_polygon = shapely.union_all([shapely.Polygon(p) for p in node_hexagons[matches]])
    cluster_boundaries[c] = joined_polygon


n_plots = len(input_columns) + 2
subplot_rows = (n_plots + 1) // 2
subplot_columns = 2
features_fig = make_subplots(
    rows=subplot_rows,
    cols=subplot_columns,
    subplot_titles=input_columns + ["Umatrix"] + ["Clusters and winning cells"],
    horizontal_spacing= 0.05,
    vertical_spacing=0.05,

)
features_fig.layout.height = 50 + 250 * subplot_rows
#features_fig.layout.width = 1200


subplot_ix = list(
    itertools.product(range(1, subplot_rows + 1), range(1, subplot_columns + 1))
)[:n_plots]

# Feature plots
for i, (row_index, col_index) in enumerate(subplot_ix[:len(input_columns)+1]):
    # add_cluster_boundaries(features_fig, row_index, col_index)
    colorbar_settings = dict(
        thickness = 20,
        len = 0.5 * (1 / subplot_rows),
        y = 1 - (row_index - 0.5) / subplot_rows,
        x = 0.38 if col_index == 1 else 0.90,
        tickfont=dict(size=18),
        tickvals=colorbar_tickvals,
        tickformat=colorbar_tickformat,
    )
    add_som_features(features_fig, row_index, col_index, i, colorbar_settings)

add_training_cells(features_fig, subplot_ix[-1][0], subplot_ix[-1][1])
add_cluster_boundaries(features_fig, subplot_ix[-1][0], subplot_ix[-1][1])
# Add cluster boundaries to U-matrix plot
add_cluster_boundaries(features_fig, subplot_ix[-2][0], subplot_ix[-2][1])


for i, (row_index, col_index) in enumerate(subplot_ix):
    sp = features_fig.get_subplot(row_index, col_index)
    sp.yaxis.scaleanchor = sp.yaxis.anchor
    sp.yaxis.showticklabels = False
    sp.xaxis.showticklabels = False

## -------------------------------------------------------
# Large cluster plot
## -------------------------------------------------------

large_cluster_fig = go.Figure()
add_training_cells(large_cluster_fig, None, None, show_sum=False)
add_cluster_boundaries(large_cluster_fig, None, None)
large_cluster_fig.layout.yaxis.scaleanchor = "x"
large_cluster_fig.layout.yaxis.showticklabels = False
large_cluster_fig.layout.xaxis.showticklabels = False

show_figure(large_cluster_fig)

# --------------------------------------------------------
# Draw isolines for SOM feature planes
# --------------------------------------------------------

# Compute per-neuron PC1 score (vectorised: shape [nx, ny])
n_features = len(input_columns)
pc1_neuron_map = (
    weights_normalized.reshape(-1, n_features) @ pca.components_[0]
).reshape(final_nx, final_ny)
pc1_isoline_value = pc1_neuron_map.mean()

print(f"\nPC1 Isoline Value (mean): {pc1_isoline_value:.6f}")
print(f"PC1 range: [{pc1_neuron_map.min():.6f}, {pc1_neuron_map.max():.6f}]")
print("="*50)

print("\n" + "="*50)
print("PC1 Value for Each Neuron")
print("="*50)
print(f"\nPC1 neuron values - Min: {pc1_neuron_map.min():.6f}, Max: {pc1_neuron_map.max():.6f}, Mean: {pc1_neuron_map.mean():.6f}")
print("="*50 + "\n")

for feat_name, isoline_vals in isoline_config.items():
    if feat_name not in input_columns:
        continue
    feat_idx = input_columns.index(feat_name)
    row_idx, col_idx = subplot_ix[feat_idx]

    # Green dashed background isolines
    add_isolines_to_som(
        features_fig,
        weights_normalized[:, :, feat_idx],
        (xx, yy),
        feat_name,
        isoline_vals,
        row_idx, col_idx,
    )

    # Highlighted (coloured) isolines for key threshold values
    if feat_name in isoline_colors:
        add_isolines_to_som(
            features_fig,
            weights_normalized[:, :, feat_idx],
            (xx, yy),
            feat_name,
            list(isoline_colors[feat_name].keys()),
            row_idx, col_idx,
            color_map=isoline_colors[feat_name],
            style_map=isoline_styles.get(feat_name, {}),
        )

# PC1 isoline on every feature subplot
for feat_name in input_columns:
    feat_idx = input_columns.index(feat_name)
    row_idx, col_idx = subplot_ix[feat_idx]
    add_pc1_isolines(features_fig, (xx, yy), pc1_neuron_map, pc1_isoline_value, row_idx, col_idx)

# Calculate and print Ward's distance for each feature
print("\n" + "="*50)
print("Ward's Distance for Each Feature")
print("="*50)

for feature_name in ['F3', 'F4', 'F5', 'F7', 'F8', 'F9']:
    if feature_name in input_columns:
        feat_idx = input_columns.index(feature_name)
        feature_weights = weights_normalized[:, :, feat_idx].reshape(-1, 1)
        
        # Calculate Ward's linkage for this feature
        ward_linkage = hierarchy.linkage(feature_weights, method='ward', metric='euclidean')
        
        # The maximum distance in the linkage matrix represents Ward's distance
        max_ward_distance = ward_linkage[:, 2].max()
        
        print(f"{feature_name}: {max_ward_distance:.4f}")

print("="*50 + "\n")

# Calculate and print variance values for each feature
print("\n" + "="*50)
print("Variance Values for Each Feature")
print("="*50)

for feature_name in ['F3', 'F4', 'F5', 'F7', 'F8', 'F9']:
    if feature_name in input_columns:
        feat_idx = input_columns.index(feature_name)
        # Get the feature data from the transformed data
        #feature_data = transformed_data[:, feat_idx]
        feature_data = weights_normalized[:, :, feat_idx].reshape(-1)
        
        # Calculate variance
        #variance = np.var(feature_data)
        variance = np.var(feature_data)
        
        print(f"{feature_name}: {variance:.6f}")

print("="*50 + "\n")



# Calculate and print weight ranges for each feature
print("\n" + "="*50)
print("Weight Ranges for Each Feature")
print("="*50)

for feature_name in ['F3', 'F4', 'F5', 'F7', 'F8', 'F9']:
    if feature_name in input_columns:
        feat_idx = input_columns.index(feature_name)
        # Get the feature weights from the SOM
        feature_weights = weights_normalized[:, :, feat_idx].flatten()
        
        # Calculate min, max, and mean
        mean_weight = feature_weights.mean()
        
        print(f"{feature_name}: Mean={mean_weight:.4f}")

print("="*50 + "\n")


# Perform PCA on the transformed data
pca_full = PCA(n_components=2)
pca_transformed = pca_full.fit_transform(transformed_data)


# Calculate PC2 value for each neuron
pc2_neuron_map = np.zeros((final_nx, final_ny))

for i in range(final_nx):
    for j in range(final_ny):
        # Get weights for this neuron across all features
        neuron_weights = weights_normalized[i, j, :]
        
        # Calculate PC2 value as dot product of neuron weights and PC2 loadings
        pc2_neuron_map[i, j] = np.dot(neuron_weights, pca.components_[1, :])

# Use mean PC2 value as the isoline threshold
pc2_isoline_value = np.mean(pc2_neuron_map)

# Add PC2 isolines to all feature plots (excluding pH_L if needed)
pc2_features = input_columns.copy()

for feat_name in pc2_features:
    if feat_name in input_columns:
        feat_idx = input_columns.index(feat_name)
        plot_position = feat_idx
        row_idx = subplot_ix[plot_position][0]
        col_idx = subplot_ix[plot_position][1]
        
        add_pc2_isolines(
            features_fig,
            (xx, yy),
            hexagons,
            feat_name,
            row_idx,
            col_idx
        )

print(f"PC2 isolines added to all feature plots")
print(f"PC2 Isoline Value: {pc2_isoline_value:.6f}")
print("="*50 + "\n")

print("="*50 + "\n")

# Display updated figures
show_figure(features_fig)

# -------------------------------------------------------
# PCA Biplot
# -------------------------------------------------------

# Perform PCA on the transformed data (already done above, but ensure it's available)
pca_biplot = PCA(n_components=2)
pca_scores = pca_biplot.fit_transform(transformed_data)

# Get the loadings (principal components)
loadings_biplot = pca_biplot.components_.T * np.sqrt(pca_biplot.explained_variance_)

# Create the enhanced biplot
fig_biplot_enhanced = go.Figure()

# Add scatter plot for samples with cluster colors
cluster_colors = df['cluster_label'].values
unique_clusters = sorted(df['cluster_label'].dropna().unique())

for cluster_id in unique_clusters:
    cluster_mask = df['cluster_label'] == cluster_id
    cluster_samples = df[cluster_mask]
    
    # Get base color and create darker border
    base_color = DEFAULT_PLOTLY_COLORS[int(cluster_id) % len(DEFAULT_PLOTLY_COLORS)]
    # Convert to darker shade for border
    import plotly.colors as pc
    
    # Parse RGB color string if it's in rgb() format, otherwise convert from hex
    if base_color.startswith('rgb'):
        rgb_match = re.findall(r'\d+', base_color)
        rgb = tuple(int(x) for x in rgb_match[:3])
    else:
        rgb = pc.hex_to_rgb(base_color)
    
    darker_rgb = tuple(int(c * 0.6) for c in rgb)  # 40% darker
    darker_color = f'rgb({darker_rgb[0]},{darker_rgb[1]},{darker_rgb[2]})'
    
    fig_biplot_enhanced.add_scatter(
        x=pca_scores[cluster_mask, 0],
        y=pca_scores[cluster_mask, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=base_color,
            line=dict(width=2, color=darker_color),
            opacity=0.8
        ),
        text=cluster_samples['Sample'],
        name=f'Cluster {int(cluster_id)}',
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
    )

# Add loading vectors with enhanced styling
scale_factor = 4.0  # Adjust for better visibility

# Mapping of original feature names to display names
feature_display_names = {
    'F4': 'f1',
    'F5': 'f2',
    'F7': 'f3',
    'F8': 'f4',
    'F9': 'f5'
}

for i, feature in enumerate(input_columns):
    # Calculate arrow position
    arrow_x = loadings_biplot[i, 0] * scale_factor
    arrow_y = loadings_biplot[i, 1] * scale_factor
    
    # Add arrow as a line with annotation
    fig_biplot_enhanced.add_trace(go.Scatter(
        x=[0, arrow_x],
        y=[0, arrow_y],
        mode='lines',
        line=dict(
            color='black',
            width=1.5
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add arrowhead
    fig_biplot_enhanced.add_annotation(
        x=arrow_x,
        y=arrow_y,
        ax=0.0,
        ay=0.0,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=1.5,
        arrowcolor='black',
    )
    
    # Add feature labels with better positioning
    label_offset = 1.05
    # Use display name if available, otherwise use original name
    display_name = feature_display_names.get(feature, feature)
    
    fig_biplot_enhanced.add_annotation(
        x=arrow_x * label_offset,
        y=arrow_y * label_offset,
        text=f'<b>{display_name}</b>',
        showarrow=False,
        font=dict(
            size=16,
            color='black',
            family='Arial Black'
        ),
    )

# Add grid lines at origin
fig_biplot_enhanced.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_biplot_enhanced.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

# Add square frame around the plot
fig_biplot_enhanced.add_shape(
    type="rect",
    x0=-1, y0=-1, x1=1, y1=1,
    line=dict(color="black", width=2),
    fillcolor="rgba(0,0,0,0)"
)

# Update layout with enhanced styling
fig_biplot_enhanced.update_layout(
    title=dict(
        text=f'<b>PCA Biplot - Samples and Feature Loadings</b><br>' +
             f'<sub>Variance Explained: PC1={pca_biplot.explained_variance_ratio_[0]*100:.1f}%, ' +
             f'PC2={pca_biplot.explained_variance_ratio_[1]*100:.1f}%</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    xaxis_title=f'<b>PC1 ({pca_biplot.explained_variance_ratio_[0]*100:.1f}%)</b>',
    yaxis_title=f'<b>PC2 ({pca_biplot.explained_variance_ratio_[1]*100:.1f}%)</b>',
    width=1000,
    height=800,
    showlegend=True,
    legend=dict(
        title=dict(text='<b>Clusters</b>'),
        orientation='v',
        x=1.02,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    ),
    plot_bgcolor='rgba(245, 245, 245, 0.5)',
    xaxis=dict(
        gridcolor='white',
        gridwidth=2,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        range=[-1, 1],
        tickfont=dict(size=16),
        constrain='domain',
        tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
        tickformat='.1f'
    ),
    yaxis=dict(
        gridcolor='white',
        gridwidth=2,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        scaleanchor='x',
        scaleratio=1,
        range=[-1, 1],
        tickfont=dict(size=18),
        constrain='domain',
        tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
        tickformat='.1f'
    ),
    font=dict(size=16)
)

# Save the biplot figure
fig_biplot_enhanced.write_html('PCA_Biplot_Enhanced.html')
print("Enhanced PCA biplot saved to 'PCA_Biplot_Enhanced.html'")

show_figure(fig_biplot_enhanced)


# -----------------------------------------
# Dendrogram plot
# -----------------------------------------
fig = plt.figure(figsize=(10, 8))

# Axis settings
left, bottom, width, height = 0.1, 0.35, 0.8, 0.35
ax1 = fig.add_axes([left, bottom, width, height])

# Customize the colors of the clusters
# colors=['#BA2F29', '#E9C832', '#8EBA42', '#67ACE6']
# NB! First plotly color is skipped to align with the coloring above
colors = [  # Convert from "rgb(123, 123, 123)" to "#445566"
    '#%02X%02X%02X' % tuple(map(int, c.strip("rgb() ").split(",")))
    for c in DEFAULT_PLOTLY_COLORS[1:]
]
hierarchy.set_link_color_palette(colors)

# Plot the dendrogram using the built-in scipy function
# Ideally, we could use maxclust directly, like we do with fcluster:
#   hierarchy.fcluster(links, i, 'maxclust')
# However, this is not supported in dendrogram, so we need to find
# the appropriate distance threshold to get the desired number of clusters
max_d = 0.7
for _ in range(20):
    # Don't plot yet, this is just to get the number of clusters
    dendrogram = hierarchy.dendrogram(
        links,
        leaf_rotation=90,
        leaf_font_size=0,
        color_threshold=max_d,
        above_threshold_color='grey',
        no_plot=True
    )
    d_clusts = len(set(dendrogram['leaves_color_list']))
    if d_clusts == final_n_clusters:
        break
    if d_clusts < final_n_clusters:
        max_d *= 0.8
    if d_clusts > final_n_clusters:
        max_d *= 1.2
else:
    # This can happen if e.g. two clusters are very close together
    print(f"Warning: Could not find a distance threshold to get {final_n_clusters} clusters")


# Main dendrogram plot
print(f"Using distance threshold {max_d:.3f} to get {d_clusts} clusters")
dendrogram = hierarchy.dendrogram(
        links,
        leaf_rotation=90,
        leaf_font_size=0,
        color_threshold=max_d,
        above_threshold_color='grey',
)
hierarchy.set_link_color_palette(None)
ax1.axhline(y=max_d, linestyle='-.', color='k', lw=1.25) 
ax1.set_ylabel('Linkage distance', fontsize=13)
ax1.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35], fontsize=10)
ax1.set_xticks([])
ax1.spines['top'].set_linewidth(1.25)
ax1.spines['bottom'].set_linewidth(1.25)
ax1.spines['left'].set_linewidth(1.25)
ax1.spines['right'].set_linewidth(1.25)
# ax1.text(150, 20.5, '$Phenon$' + ' ' + '$Line$', fontsize=12)


# The bottom figure showing the cluster names
left, bottom, width, height = 0.1, 0.15, 0.8, 0.2
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_xlim(0, (len(links) + 1) * 10)
ax2.set_ylim(0, 2)


# Dot line to split each cluster 
unique, counts = np.unique(dendrogram['leaves_color_list'], return_counts=True)
cluster_size_dict = dict(zip(unique, counts))
cluster_size = [
    cluster_size_dict[clr]
    for clr in dict.fromkeys(dendrogram['leaves_color_list'])
]
boundaries = np.cumsum(cluster_size) * 10
for i in range(final_n_clusters - 1):
    ax2.plot([boundaries[i], boundaries[i]], [1.5, 2.0], linestyle='--', color='k', lw=1.5) 

ax2.set_xticks([])
ax2.set_yticks([])
    
ax2.spines['top'].set_color('none')
ax2.spines['bottom'].set_color('none')
ax2.spines['left'].set_color('none')
ax2.spines['right'].set_color('none')

# Add cluster labels for arbitrary number of clusters
extended_boundaries = np.concatenate([[0], boundaries])
for i in range(final_n_clusters):
    left_bound = extended_boundaries[i]
    right_bound = extended_boundaries[i + 1]
    center_x = left_bound + (right_bound - left_bound) / 2
    ax2.text(center_x, 1.5, f'C{i+1} \nN={cluster_size[i]}', ha='center', va='bottom', fontsize=12)

plt.show()
# ----------------------------------------
