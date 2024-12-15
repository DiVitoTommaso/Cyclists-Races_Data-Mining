from scipy.sparse import csgraph
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, OPTICS, DBSCAN
import math
import lz4.frame
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import gower
import os
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances


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


def sil_vs_coh(df, labels_list, x_values, x_name="Number of Clusters (K)"):
    sils = []
    coh = []

    for labels in labels_list:
        # Filter out noise points (labels == -1)
        valid_indices = labels != -1
        filtered_df = df[valid_indices]
        filtered_labels = labels[valid_indices]

        # Check if valid clusters remain
        if len(np.unique(filtered_labels)) > 1:  # Silhouette requires at least 2 clusters
            sils.append(silhouette_score(filtered_df, filtered_labels))
        else:
            sils.append(0)  # Assign 0 if not enough clusters

        # Cohesion calculation
        cohesion = 0
        for cluster_id in np.unique(filtered_labels):
            # Get points in the current cluster
            cluster_points = filtered_df[filtered_labels == cluster_id]

            # Calculate the centroid
            centroid = np.mean(cluster_points, axis=0)

            # Squared distances from points to the centroid
            distances = np.sum((cluster_points - centroid) ** 2, axis=1)
            cohesion += np.sum(distances)

        coh.append(cohesion)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, sils, marker='o', color='b', label='Silhouette Score')
    plt.title("Silhouette Score")
    plt.xlabel(x_name)
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot cohesion values
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, coh, marker='o', color='r', label='Cohesion')
    plt.title("Cohesion across Clusters")
    plt.xlabel(x_name)
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    return sils, coh

from sklearn.cluster import KMeans
from math import pi

def plot_centers(df, centers, n_rows=4, n_cols=3):
    # First part: Line plot for cluster centers
    plt.figure(figsize=(8, 4))
    for i in range(len(centers)):
        plt.plot(centers.iloc[i], marker='o', label='Cluster %s' % i)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(df.columns)), df.columns, fontsize=8)
    plt.legend(fontsize=10)
    plt.title("Cluster Centers (Line Plot)", fontsize=12)
    plt.show()

    N = len(df.columns)  # Number of features
    num_clusters = len(centers)

    # Set up the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': 'polar'}, figsize=(6 * n_cols, 6 * n_rows))

    # Flatten axes to easily index them if the grid is not square
    axes = axes.flatten()

    # If there are more clusters than available subplots, adjust the number of subplots dynamically
    for i in range(num_clusters):
        ax = axes[i]

        # Calculate the angles for the radar chart
        angles = [n / float(N) * 2 * pi for n in range(N)]  # Angle for each feature
        values = centers.iloc[i].tolist()  # Cluster center values
        values += values[:1]  # Ensure the plot is closed by repeating the first value at the end
        angles += angles[:1]  # Repeat the first angle to close the radar chart

        ax.set_theta_offset(pi / 2)  # Rotate the chart
        ax.set_theta_direction(-1)  # Clockwise direction
        ax.set_rlabel_position(0)  # Remove radial labels
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Cluster {i}')
        ax.fill(angles, values, 'b', alpha=0.1)  # Fill the area
        ax.set_xticks(angles[:-1])  # Set the feature names as ticks
        ax.set_xticklabels(df.columns, fontsize=8)
        ax.set_title(f'Cluster {i} - Radar Plot', fontsize=10)
        ax.legend(fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_scores(df):
    cohesion = []
    silhouette_scores = []
    results = []
    # Run KMeans for different values of K
    for k in range(2, 100):
        kmeans = KMeans(n_clusters=k, random_state=1804, n_init=10)
        kmeans.fit(df)

        # Calculate cohesion (inertia)
        cohesion.append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(silhouette)

        results.append(kmeans)

    # Plot Cohesion (Inertia) vs K
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(list(range(2,100)), cohesion, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cohesion (Inertia)')
    plt.title('Elbow Method for Optimal K')

    # Plot Silhouette Score vs K
    plt.subplot(1, 2, 2)
    plt.plot(list(range(2,100)), silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')

    plt.tight_layout()
    plt.show()

    return results

def optimal_pair_plot(df, cluster_name, labels, centroids=None):
    # Add the cluster labels to the DataFrame
    df[cluster_name] = labels

    # Create the pairplot with seaborn
    pairplot = sns.pairplot(df, hue=cluster_name, palette='husl', diag_kind='kde')

    if centroids is not None:
        # Add centroids to each plot in the pairplot
        for i, ax in enumerate(pairplot.axes.flatten()):
            if centroids.shape[1] == 2:  # Handle 2D data
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label="Centroids")
            else:  # Handle 1D data
                ax.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), c='red', marker='X', s=100, label="Centroids")
            ax.legend()
    plt.show()

    # Remove cluster label column
    df.drop(columns=[cluster_name], inplace=True)


# Step 2: Apply PCA
def apply_pca(data, dimensions=2):
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(data)

# Step 3: Apply t-SNE
def apply_tsne(data, dimensions=2):
    tsne = TSNE(n_components=dimensions, random_state=42)
    return tsne.fit_transform(data)

# Step 4: Plot PCA and t-SNE results
def plot_results(embedding, labels, title, dimensions=2):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("hsv", len(unique_labels))

    if dimensions == 2:
        for i, label in enumerate(unique_labels):
            plt.scatter(
                embedding[np.array(labels) == label, 0],
                embedding[np.array(labels) == label, 1],
                label=f"Cluster {label}",
                color=colors[i],
                alpha=0.7
            )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, label in enumerate(unique_labels):
            ax.scatter(
                embedding[np.array(labels) == label, 0],
                embedding[np.array(labels) == label, 1],
                embedding[np.array(labels) == label, 2],
                label=f"Cluster {label}",
                color=colors[i],
                alpha=0.7
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

    plt.title(title)
    plt.legend()
    plt.show()


def spectral_eigen_gap(df):
    sigma = np.median(np.linalg.norm(df[:, np.newaxis] - df, axis=2))  # Stima di sigma
    affinity_matrix = rbf_kernel(df, gamma=1.0 / (2 * sigma ** 2))

    # 3. Calcola la matrice Laplaciana normalizzata
    laplacian_matrix = csgraph.laplacian(affinity_matrix, normed=True)

    # 4. Calcola gli autovalori e autovettori della matrice Laplaciana
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # 5. Calcola il "gap" tra autovalori consecutivi
    eigengaps = np.diff(eigenvalues)

    # 6. Trova automaticamente il numero ottimale di cluster
    optimal_clusters = np.argmax(eigengaps) + 1  # Indice del salto + 1

    # 7. Visualizza gli autovalori e l'eigengap
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=5, label='Autovalori')
    plt.axvline(optimal_clusters, color='r', linestyle='--', label=f"Numero di cluster = {optimal_clusters}")
    plt.xlabel("Indice dell'autovalore")
    plt.ylabel("Autovalore")
    plt.title("Autovalori della matrice Laplaciana (Eigengap Heuristic)")
    plt.legend()
    plt.grid()
    plt.show()

def dbscan_tune(df, dst_matrix, eps_range):
    dbscan_labels = []
    data = pd.DataFrame({'eps': [], 'Clusters': [], 'Noise': []})
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=pow(2, df.shape[1]), metric="precomputed", n_jobs=-1)
        dbscan.fit_predict(dst_matrix)

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)

        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise (-1) as a cluster
        n_noise = counts[unique_labels == -1][0] if -1 in unique_labels else 0

        new_row = {'eps': eps, 'Clusters': n_clusters, 'Noise': n_noise}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

        dbscan_labels.append(dbscan.labels_)

    print(data)
    sil_vs_coh(df, dbscan_labels, eps_range, x_name='eps')

def optics_tune(df, eps_range):
    optics_labels = []
    results = pd.DataFrame({'eps': [], 'Clusters': [], 'Noise': []})  # Results table

    # Loop over different max_eps values to tune OPTICS
    for eps in eps_range:
        # Run OPTICS with specified max_eps
        optics = OPTICS(max_eps=eps, min_samples=df.shape[1]*2, metric='euclidean')
        optics.fit(df.values)

        # Get labels and noise points
        labels = optics.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
        n_noise = np.sum(labels == -1)

        # Append metrics to the results DataFrame
        new_row = {'eps': eps, 'Clusters': n_clusters, 'Noise': n_noise}
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        optics_labels.append(labels)

    # Print results table
    print("Results for different eps values:")
    print(results)

    # Evaluate silhouette and cohesion metrics
    sil_vs_coh(df, optics_labels, eps_range)

def similarity_matrix(df):
    distance_matrix = pairwise_distances(df, metric='euclidean')  # Matrice delle distanze
    sim_matrix = 1 / (1 + distance_matrix)  # Trasforma le distanze in similarità

    # 3. Visualizza la Similarity Matrix come Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap='coolwarm', square=True)
    plt.title("Similarity Matrix Heatmap")
    plt.xlabel("Data Points")
    plt.ylabel("Data Points")
    plt.show()