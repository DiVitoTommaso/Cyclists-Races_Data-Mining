from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
import lz4.frame
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import gower
import os
from scipy.spatial.distance import squareform


# Define the transform function
def transform(df, categorical_cols=[], standardize_cols=[], minmax_cols=[], samples=0):
    if samples != 0:
        df = df.sample(n=samples, random_state=1804)

    preprocessor = ColumnTransformer(
        transformers=[
            ("standardize", StandardScaler(), standardize_cols),
            ("minmax", MinMaxScaler(), minmax_cols),
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
        ]
    )
    transformed_data = preprocessor.fit_transform(df)
    return transformed_data, preprocessor


# Define the inverse transform function
def inverse_transform(transformed_data, preprocessor, categorical_cols=[], standardize_cols=[],
                      minmax_cols=[]):
    # Initialize a DataFrame to reconstruct the inverse-transformed data
    inverse_df = pd.DataFrame(index=range(transformed_data.shape[0]))

    # Step 1: Extract transformed components
    start_idx = 0

    # Inverse StandardScaler
    if standardize_cols:
        end_idx = start_idx + len(standardize_cols)
        standard_data = transformed_data[:, start_idx:end_idx]
        inverse_df[standardize_cols] = preprocessor.named_transformers_["standardize"].inverse_transform(standard_data)
        start_idx = end_idx

    # Inverse MinMaxScaler
    if minmax_cols:
        end_idx = start_idx + len(minmax_cols)
        minmax_data = transformed_data[:, start_idx:end_idx]
        inverse_df[minmax_cols] = preprocessor.named_transformers_["minmax"].inverse_transform(minmax_data)
        start_idx = end_idx

    # Inverse OneHotEncoder
    if categorical_cols:
        cat_data = transformed_data[:, start_idx:]
        inverse_cats = preprocessor.named_transformers_["cat"].inverse_transform(cat_data)
        inverse_df[categorical_cols] = pd.DataFrame(inverse_cats, columns=categorical_cols)

    return inverse_df


def compute_hc(distance_matrix, methods=["ward"], is_square=False):
    results = {}
    # Convert the distance matrix to condensed form if it's in square form
    if is_square:
        dst_matrix = squareform(distance_matrix, force="tovector")
    else:
        dst_matrix = distance_matrix

    for method in methods:
        try:
            data_link = linkage(dst_matrix, method=method)
            results[method] = data_link
        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = None
    return results


from scipy.cluster.hierarchy import dendrogram

def plot_hc(ax, data_link, method, metric, color_threshold=10, truncate_mode="lastp", p=100):
    if data_link is None:
        print(f"No data to plot for {method}-{metric}")
        return

    # Plot dendrogram on the specified axis
    dendrogram(
        data_link,
        color_threshold=color_threshold,
        truncate_mode=truncate_mode,
        p=p,
        no_labels=False,
        ax=ax
    )
    ax.set_title(f"Method: {method.capitalize()} | Metric: {metric}", fontsize=12)
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Linkage Distance")


def plot_all_hc(results, metric, color_threshold=10, truncate_mode="lastp", p=100):
    num_methods = len(results)

    # Set up the subplot grid with one plot per row
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 5 * num_methods), squeeze=False)

    # Iterate through methods and linkage data
    for ax, (method, data_link) in zip(axes[:, 0], results.items()):
        plot_hc(ax, data_link, method, metric, color_threshold, truncate_mode, p)

    plt.tight_layout()
    plt.show()

def compute_distance_matrix(df, metric="euclidean", cat_features=[], cache_file=None):
    if metric == "gower":
        distance_matrix = gower.gower_matrix(df, cat_features=cat_features)
    else:
        distance_matrix = pdist(df, metric=metric)

    return distance_matrix


from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import random_center_initializer


def x_means(df):
    np.random.seed(1804)

    # Prepare data for pyclustering (as list of lists)
    data_points = df.values.tolist()

    # Define initial conditions
    initial_centers = random_center_initializer(data_points, 2).initialize()  # Start with 2 clusters

    # Run X-Means algorithm
    xmeans_instance = xmeans(data_points, initial_centers, ccore=True)  # Use CCORE for speed
    xmeans_instance.process()
    print("Done")
    # Extract clustering results
    clusters = xmeans_instance.get_clusters()  # List of cluster indices
    centers = xmeans_instance.get_centers()  # Final cluster centers

    # Assign cluster labels to DataFrame
    labels = [-1] * len(data_points)  # Default label (-1 for unassigned points)
    for cluster_id, cluster_points in enumerate(clusters):
        for point_index in cluster_points:
            labels[point_index] = cluster_id

    return labels, centers


def sil_vs_coh(df, labels_list, x_values):
    sils = []
    coh = []
    for labels in labels_list:
        sils.append(silhouette_score(df, labels))

        cohesion = 0
        for cluster_id in np.unique(labels):
            # Get the points in the current cluster
            cluster_points = df[labels == cluster_id]

            # Calculate the centroid of the cluster (mean of points)
            centroid = np.mean(cluster_points, axis=0)

            # Calculate the squared distances from each point to the centroid
            distances = np.sum((cluster_points - centroid) ** 2, axis=1)

            # Add to the total cohesion
            cohesion += np.sum(distances)
        coh.append(cohesion)

    plt.figure(figsize=(10, 6))

    # Plot silhouette scores
    plt.plot(x_values, sils, marker='o', linestyle='-', color='b', label='Silhouette Score')
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Metric Value")
    plt.show()
    # Plot cohesion values (su un asse separato per scalare i valori)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, coh, marker='x', linestyle='--', color='r', label='Cohesion')

    # 4. Personalizzazione del grafico
    plt.title("Cohesion across Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Metric Value")
    plt.legend()  # Mostra legenda
    plt.grid(True)  # Aggiunge la griglia

    # 5. Visualizzazione del grafico
    plt.show()

    return sils, coh
