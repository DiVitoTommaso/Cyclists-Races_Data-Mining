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


# Define the transform function for preprocessing data
def transform(df, categorical_cols=[], standardize_cols=[], minmax_cols=[], samples=0):
    """
    Transforms the data by applying standardization, min-max scaling, and one-hot encoding.

    :param df: The input dataframe to be transformed
    :param categorical_cols: List of categorical columns to one-hot encode
    :param standardize_cols: List of columns to standardize (mean=0, std=1)
    :param minmax_cols: List of columns to apply min-max scaling (scaled between 0 and 1)
    :param samples: If non-zero, downsample the dataframe to the specified number of samples
    :return: Transformed data and the preprocessor used to transform the data
    """
    if samples != 0:
        df = df.sample(n=samples, random_state=1804)  # Randomly sample rows if specified

    # Define a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("standardize", StandardScaler(), standardize_cols),  # Apply standardization
            ("minmax", MinMaxScaler(), minmax_cols),  # Apply min-max scaling
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
            # Apply one-hot encoding for categorical columns
        ]
    )

    # Apply the transformations to the dataframe
    transformed_data = preprocessor.fit_transform(df)
    return transformed_data, preprocessor


# Define the inverse transform function to recover original data
def inverse_transform(transformed_data, preprocessor, categorical_cols=[], standardize_cols=[], minmax_cols=[]):
    """
    Inversely transforms the data to recover the original values (before transformation).

    :param transformed_data: The transformed data to be inverse transformed
    :param preprocessor: The preprocessor used to transform the data initially
    :param categorical_cols: List of categorical columns
    :param standardize_cols: List of standardized columns
    :param minmax_cols: List of min-max scaled columns
    :return: The inverse transformed dataframe (reconstructed data)
    """
    # Initialize an empty DataFrame for the inverse transformed data
    inverse_df = pd.DataFrame(
        index=range(transformed_data.shape[0]))  # Initialize with the number of rows of transformed data

    start_idx = 0

    # Step 1: Inverse transform StandardScaler columns
    if standardize_cols:
        end_idx = start_idx + len(standardize_cols)
        standard_data = transformed_data[:, start_idx:end_idx]  # Extract the transformed standard data
        inverse_df[standardize_cols] = preprocessor.named_transformers_["standardize"].inverse_transform(standard_data)
        start_idx = end_idx

    # Step 2: Inverse transform MinMaxScaler columns
    if minmax_cols:
        end_idx = start_idx + len(minmax_cols)
        minmax_data = transformed_data[:, start_idx:end_idx]  # Extract the transformed min-max data
        inverse_df[minmax_cols] = preprocessor.named_transformers_["minmax"].inverse_transform(minmax_data)
        start_idx = end_idx

    # Step 3: Inverse transform OneHotEncoder columns
    if categorical_cols:
        cat_data = transformed_data[:, start_idx:]  # Extract the transformed categorical data
        inverse_cats = preprocessor.named_transformers_["cat"].inverse_transform(cat_data)  # Inverse one-hot encoding
        inverse_df[categorical_cols] = pd.DataFrame(inverse_cats,
                                                    columns=categorical_cols)  # Add categorical columns back

    return inverse_df


# Compute hierarchical clustering using different linkage methods
def compute_hc(distance_matrix, methods=None, is_square=False):
    """
    Performs hierarchical clustering using specified linkage methods.

    :param distance_matrix: A pre-computed distance matrix (e.g., Euclidean, Gower)
    :param methods: A list of linkage methods (e.g., 'ward', 'single', 'complete')
    :param is_square: Boolean indicating if the input distance matrix is in square form
    :return: A dictionary of linkage results for each method
    """
    if methods is None:
        methods = ["ward"]  # Default to 'ward' method for hierarchical clustering
    results = {}

    # Convert the distance matrix to condensed form if it is in square form
    if is_square:
        dst_matrix = squareform(distance_matrix, force="tovector")  # Convert squareform to condensed form
    else:
        dst_matrix = distance_matrix  # Use the provided matrix directly

    # Perform hierarchical clustering for each specified method
    for method in methods:
        try:
            data_link = linkage(dst_matrix, method=method)  # Apply the linkage method to the distance matrix
            results[method] = data_link  # Store the results for each method
        except Exception as e:
            print(f"Error with {method}: {e}")  # Handle any errors during clustering
            results[method] = None
    return results


# Plot a dendrogram for hierarchical clustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, fcluster, inconsistent


def plot_hc(ax, data_link, method, metric, inconsistency_threshold=1.5):
    """
    Plots a hierarchical clustering dendrogram.

    :param ax: The axis on which to plot the dendrogram
    :param data_link: Linkage data obtained from hierarchical clustering
    :param method: The linkage method used (e.g., 'ward', 'single')
    :param metric: The distance metric used (e.g., 'euclidean', 'gower')
    :param inconsistency_threshold: Threshold for inconsistency measure to detect outliers
    """
    if data_link is None:
        print(f"No data to plot for {method}-{metric}")
        return

    # Plot the dendrogram on the given axis
    dendrogram(data_link, ax=ax)
    ax.set_title(f"Method: {method.capitalize()} | Metric: {metric}", fontsize=12)
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Linkage Distance")


# Plot dendrograms for multiple clustering methods
def plot_all_hc(results, metric):
    """
    Plots hierarchical clustering dendrograms for multiple linkage methods.

    :param results: A dictionary of linkage results for different methods
    :param metric: The distance metric used for clustering
    """
    num_methods = len(results)

    # Create subplots to display dendrograms for each method
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 5 * num_methods), squeeze=False)

    # Iterate through the results and plot dendrogram for each method
    for ax, (method, data_link) in zip(axes[:, 0], results.items()):
        plot_hc(ax, data_link, method, metric)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


# Compute the distance matrix for the dataset using a specified metric
def compute_distance_matrix(df, metric="euclidean", cat_features=[], cache_file=None):
    """
    Computes a distance matrix for the dataframe using the specified metric (e.g., Euclidean, Gower).

    :param df: The input dataframe for which the distance matrix will be calculated
    :param metric: The distance metric to use ('euclidean', 'gower', etc.)
    :param cat_features: List of categorical features (if using Gower distance)
    :param cache_file: If provided, the distance matrix will be cached to this file
    :return: The computed distance matrix (as a 1D array for pairwise distances)
    """
    if metric == "gower":
        # If using Gower distance, calculate the Gower similarity matrix
        distance_matrix = gower.gower_matrix(df, cat_features=cat_features)
    else:
        # For other metrics (e.g., Euclidean), use pdist to compute pairwise distances
        distance_matrix = pdist(df, metric=metric)

    return distance_matrix


from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import random_center_initializer


def x_means(df):
    """
    Performs clustering using the X-Means algorithm and returns cluster labels and centers.

    :param df: The input dataframe for clustering
    :return: A tuple of (labels, centers) where:
        - labels: List of cluster labels for each data point
        - centers: List of final cluster centers
    """
    np.random.seed(1804)  # Set random seed for reproducibility

    # Prepare the data for pyclustering (convert dataframe to list of lists)
    data_points = df.values.tolist()

    # Initialize the X-Means algorithm with 2 clusters to start
    initial_centers = random_center_initializer(data_points, 2).initialize()

    # Run X-Means clustering with the initialized centers and CCORE for performance
    xmeans_instance = xmeans(data_points, initial_centers, ccore=True)
    xmeans_instance.process()
    print("Done")

    # Extract clustering results
    clusters = xmeans_instance.get_clusters()  # List of clusters with indices of data points
    centers = xmeans_instance.get_centers()  # Final cluster centers

    # Assign cluster labels to each data point
    labels = [-1] * len(data_points)  # Default label is -1 for unassigned points
    for cluster_id, cluster_points in enumerate(clusters):
        for point_index in cluster_points:
            labels[point_index] = cluster_id  # Assign the cluster label

    return labels, centers  # Return cluster labels and centers


def sil_vs_coh(df, labels_list, x_values, x_name="Number of Clusters (K)"):
    """
    Evaluates clustering performance by calculating silhouette scores and cohesion for different clusterings.

    :param df: The input dataframe
    :param labels_list: List of cluster labels for different clustering results
    :param x_values: The x-axis values (e.g., number of clusters) for plotting
    :param x_name: Label for the x-axis in the plot (default: "Number of Clusters (K)")
    :return: Tuple of silhouette scores and cohesion values for each clustering result
    """
    sils = []  # List to store silhouette scores
    coh = []  # List to store cohesion values

    for labels in labels_list:
        # Filter out points labeled as noise (label = -1)
        valid_indices = labels != -1
        filtered_df = df[valid_indices]  # Filtered dataframe with valid points
        filtered_labels = labels[valid_indices]  # Filtered labels

        # Calculate silhouette score if there are more than 1 cluster
        if len(np.unique(filtered_labels)) > 1:
            sils.append(silhouette_score(filtered_df, filtered_labels))  # Calculate silhouette score
        else:
            sils.append(0)  # Assign 0 if there are not enough clusters

        # Calculate cohesion: sum of squared distances from points to their centroid
        cohesion = 0
        for cluster_id in np.unique(filtered_labels):
            cluster_points = filtered_df[filtered_labels == cluster_id]  # Points in the current cluster
            centroid = np.mean(cluster_points, axis=0)  # Calculate centroid of the cluster
            distances = np.sum((cluster_points - centroid) ** 2, axis=1)  # Squared distances from centroid
            cohesion += np.sum(distances)  # Add to total cohesion

        coh.append(cohesion)  # Store cohesion for current clustering result

    # Plot silhouette scores vs. number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, sils, marker='o', color='b', label='Silhouette Score')
    plt.title("Silhouette Score")
    plt.xlabel(x_name)
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot cohesion vs. number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, coh, marker='o', color='r', label='Cohesion')
    plt.title("Cohesion across Clusters")
    plt.xlabel(x_name)
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    return sils, coh  # Return silhouette scores and cohesion values


from sklearn.cluster import KMeans
from math import pi
import umap

def plot_centers(df, centers, n_rows=4, n_cols=3):
    """
    Plots the cluster centers in both a line plot and radar (polar) plot.

    :param df: The input dataframe containing the data points
    :param centers: A DataFrame or array containing the cluster centers
    :param n_rows: Number of rows in the grid for radar plots (default: 4)
    :param n_cols: Number of columns in the grid for radar plots (default: 3)
    """
    # First part: Line plot for the cluster centers
    plt.figure(figsize=(8, 4))
    for i in range(len(centers)):
        plt.plot(centers.iloc[i], marker='o', label=f'Cluster {i}')  # Plot each cluster center
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(range(0, len(df.columns)), df.columns, fontsize=8)  # Set x-ticks to feature names
    plt.legend(fontsize=10)
    plt.title("Cluster Centers (Line Plot)", fontsize=12)
    plt.show()

    # Number of features (columns in the dataframe)
    N = len(df.columns)
    num_clusters = len(centers)  # Number of clusters

    # Set up the figure and subplots for radar charts
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': 'polar'}, figsize=(6 * n_cols, 6 * n_rows))

    # Flatten the axes array for easier indexing if the grid is not square
    axes = axes.flatten()

    # Loop through each cluster and plot its radar chart
    for i in range(num_clusters):
        ax = axes[i]  # Get the current axis for the radar plot

        # Calculate angles for the radar chart (evenly spaced around the circle)
        angles = [n / float(N) * 2 * pi for n in range(N)]  # Angle for each feature
        values = centers.iloc[i].tolist()  # Get the center values for the cluster
        values += values[:1]  # Repeat the first value to close the radar chart
        angles += angles[:1]  # Repeat the first angle to close the radar chart

        ax.set_theta_offset(pi / 2)  # Rotate the chart so that the first feature is at the top
        ax.set_theta_direction(-1)  # Set the direction to clockwise
        ax.set_rlabel_position(0)  # Remove radial labels
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Cluster {i}')  # Plot the cluster center
        ax.fill(angles, values, 'b', alpha=0.1)  # Fill the area under the plot
        ax.set_xticks(angles[:-1])  # Set feature names as ticks
        ax.set_xticklabels(df.columns, fontsize=8)  # Set feature names as labels
        ax.set_title(f'Cluster {i} - Radar Plot', fontsize=10)  # Title for each subplot
        ax.legend(fontsize=8)

    # Adjust layout for tight plotting and display the figure
    plt.tight_layout()
    plt.show()


def plot_scores(df):
    """
    Plots the Cohesion (Inertia) and Silhouette Score for different values of K (number of clusters).

    :param df: The input dataframe containing the data to be clustered
    :return: A list of KMeans models for each K value (from 2 to 99)
    """
    cohesion = []  # List to store cohesion values (inertia)
    silhouette_scores = []  # List to store silhouette scores
    results = []  # List to store KMeans models for each K

    # Run KMeans for different values of K (from 2 to 99)
    for k in range(2, 100):
        kmeans = KMeans(n_clusters=k, random_state=1804, n_init=10)  # Initialize KMeans
        kmeans.fit(df)  # Fit the model to the data

        # Calculate and store cohesion (inertia)
        cohesion.append(kmeans.inertia_)

        # Calculate and store silhouette score
        silhouette = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(silhouette)

        # Store the fitted KMeans model
        results.append(kmeans)

    # Plot Cohesion (Inertia) vs K (Elbow Method)
    plt.figure(figsize=(12, 5))

    # First subplot: Elbow Method plot for optimal K
    plt.subplot(1, 2, 1)
    plt.plot(list(range(2, 100)), cohesion, 'bo-')  # Plot inertia vs number of clusters
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cohesion (Inertia)')
    plt.title('Elbow Method for Optimal K')

    # Second subplot: Silhouette Score vs K plot
    plt.subplot(1, 2, 2)
    plt.plot(list(range(2, 100)), silhouette_scores, 'ro-')  # Plot silhouette score vs number of clusters
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

    return results  # Return the list of KMeans models for each K value


def optimal_pair_plot(df, cluster_name, labels, centroids=None):
    """
    Creates a pairplot showing the relationships between features and color-codes the data points by cluster label.
    Optionally adds the centroids of clusters to each pairplot.

    :param df: The input dataframe with features
    :param cluster_name: Name of the cluster label column to color the data points by
    :param labels: The cluster labels for each data point
    :param centroids: Optionally provide centroids to plot on the pairplot (default: None)
    """
    # Add the cluster labels to the DataFrame
    df[cluster_name] = labels

    # Create pairplot with seaborn, coloring by cluster labels
    pairplot = sns.pairplot(df, hue=cluster_name, palette='husl', diag_kind='kde')

    if centroids is not None:
        # Add centroids to each plot in the pairplot
        for i, ax in enumerate(pairplot.axes.flatten()):
            if centroids.shape[1] == 2:  # Handle 2D data for centroids
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label="Centroids")
            else:  # Handle 1D data
                ax.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), c='red', marker='X', s=100,
                           label="Centroids")
            ax.legend()

    # Display the pairplot
    plt.show()

    # Remove the cluster label column after plotting
    df.drop(columns=[cluster_name], inplace=True)

# Step 5: Apply UMAP (Uniform Manifold Approximation and Projection)
def apply_umap(data, dimensions=2):
    """
    Applies UMAP to reduce the dimensionality of the data to the specified number of dimensions.

    :param data: The input data (features) to be transformed
    :param dimensions: The number of dimensions to reduce the data to (default: 2)
    :return: The transformed data with reduced dimensions
    """
    umap_model = umap.UMAP(n_components=dimensions, random_state=1804)  # Initialize UMAP with the desired number of dimensions
    return umap_model.fit_transform(data)  # Fit and transform the data


# Step 2: Apply PCA (Principal Component Analysis)
def apply_pca(data, dimensions=2):
    """
    Applies PCA to reduce the dimensionality of the data to the specified number of dimensions.

    :param data: The input data (features) to be transformed
    :param dimensions: The number of dimensions to reduce the data to (default: 2)
    :return: The transformed data with reduced dimensions
    """
    pca = PCA(n_components=dimensions)  # Initialize PCA with the desired number of dimensions
    return pca.fit_transform(data)  # Fit and transform the data


# Step 3: Apply t-SNE (t-Distributed Stochastic Neighbor Embedding)
def apply_tsne(data, dimensions=2):
    """
    Applies t-SNE to reduce the dimensionality of the data to the specified number of dimensions.

    :param data: The input data (features) to be transformed
    :param dimensions: The number of dimensions to reduce the data to (default: 2)
    :return: The transformed data with reduced dimensions
    """
    tsne = TSNE(n_components=dimensions, random_state=1804, n_jobs=-1)  # Initialize t-SNE with the desired number of dimensions
    return tsne.fit_transform(data)  # Fit and transform the data


# Step 4: Plot PCA and t-SNE results
def plot_results(embedding, labels, title, dimensions=2):
    """
    Plots the results of dimensionality reduction (PCA or t-SNE) in a 2D or 3D scatter plot,
    color-coded by the cluster labels.

    :param embedding: The transformed data (PCA or t-SNE output)
    :param labels: The cluster labels for each data point
    :param title: The title for the plot
    :param dimensions: The number of dimensions for the plot (2 or 3)
    """
    plt.figure(figsize=(8, 6))  # Set up the figure for 2D plot
    unique_labels = np.unique(labels)  # Get unique cluster labels
    colors = sns.color_palette("hsv", len(unique_labels))  # Generate a color palette

    if dimensions == 2:
        # Plot the 2D scatter plot for PCA/t-SNE results
        for i, label in enumerate(unique_labels):
            plt.scatter(
                embedding[np.array(labels) == label, 0],  # x-coordinates
                embedding[np.array(labels) == label, 1],  # y-coordinates
                label=f"Cluster {label}",  # Label each cluster
                color=colors[i],  # Use a different color for each cluster
                alpha=0.7  # Set transparency for points
            )
        plt.xlabel("Component 1")  # Label x-axis
        plt.ylabel("Component 2")  # Label y-axis
    elif dimensions == 3:
        # Plot the 3D scatter plot for PCA/t-SNE results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, label in enumerate(unique_labels):
            ax.scatter(
                embedding[np.array(labels) == label, 0],  # x-coordinates
                embedding[np.array(labels) == label, 1],  # y-coordinates
                embedding[np.array(labels) == label, 2],  # z-coordinates
                label=f"Cluster {label}",  # Label each cluster
                color=colors[i],  # Use a different color for each cluster
                alpha=0.7  # Set transparency for points
            )
        ax.set_xlabel("Component 1")  # Label x-axis
        ax.set_ylabel("Component 2")  # Label y-axis
        ax.set_zlabel("Component 3")  # Label z-axis

    # Set title and display the plot
    plt.title(title)
    plt.legend()
    plt.show()


def spectral_eigen_gap(df):
    """
    Uses the eigenvalue gap (eigengap) heuristic to determine the optimal number of clusters
    based on the spectral clustering approach.

    :param df: The input data (features) to be clustered
    :return: None (displays a plot of eigenvalues and eigengaps)
    """
    # Estimate sigma for the RBF kernel (median distance between data points)
    sigma = np.median(np.linalg.norm(df[:, np.newaxis] - df, axis=2))
    affinity_matrix = rbf_kernel(df, gamma=1.0 / (2 * sigma ** 2))  # Calculate the affinity (similarity) matrix

    # Compute the normalized Laplacian matrix from the affinity matrix
    laplacian_matrix = csgraph.laplacian(affinity_matrix, normed=True)

    # Calculate eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Remove small eigenvalues close to zero that are typically present due to the structure of Laplacians
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]

    # Compute the gaps between consecutive eigenvalues
    eigengaps = np.diff(eigenvalues)

    # Find the index of the largest eigenvalue gap, which gives an estimate of the optimal number of clusters
    optimal_clusters = np.argmax(eigengaps) + 1  # Adding 1 because index is zero-based

    # Plot the eigenvalues and highlight the optimal number of clusters
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=5, label='Eigenvalues')
    plt.axvline(optimal_clusters, color='r', linestyle='--', label=f"Optimal Clusters = {optimal_clusters}")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of the Laplacian Matrix (Eigengap Heuristic)")
    plt.legend()
    plt.grid()
    plt.show()


def dbscan_tune(df, dst_matrix, eps_range, samples):
    """
    Tunes the DBSCAN algorithm over a range of epsilon values to find the best clustering.

    :param df: The input data (features) to be clustered
    :param dst_matrix: The precomputed distance matrix used by DBSCAN
    :param eps_range: Range of epsilon values to tune DBSCAN
    :return: None (displays clustering results and plots silhouette vs cohesion)
    """
    dbscan_labels = []  # List to store the labels generated by DBSCAN for each epsilon value
    data = pd.DataFrame({'eps': [], 'Clusters': [], 'Noise': []})  # Dataframe to store results

    # Loop over different epsilon values
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=samples, metric="precomputed", n_jobs=-1)
        dbscan.fit_predict(dst_matrix)  # Apply DBSCAN to the distance matrix

        # Get unique cluster labels and their counts
        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)

        # Number of clusters is the unique labels excluding noise (-1)
        n_clusters = len(unique_labels[unique_labels != -1])
        # Number of noise points is those labeled as -1
        n_noise = counts[unique_labels == -1][0] if -1 in unique_labels else 0

        # Store the results for this epsilon value
        new_row = {'eps': eps, 'Clusters': n_clusters, 'Noise': n_noise}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

        dbscan_labels.append(dbscan.labels_)

    print(data)  # Print the results dataframe
    sil_vs_coh(df, dbscan_labels, eps_range, x_name='eps')  # Evaluate silhouette and cohesion for DBSCAN


def optics_tune(df, eps_range, samples):
    """
    Tunes the OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
    over a range of epsilon values to find the best clustering.

    :param df: The input data (features) to be clustered
    :param eps_range: Range of epsilon values to tune OPTICS
    :return: None (displays clustering results and plots silhouette vs cohesion)
    """
    optics_labels = []  # List to store labels generated by OPTICS for each epsilon value
    results = pd.DataFrame({'eps': [], 'Clusters': [], 'Noise': []})  # Dataframe to store results

    # Loop over different epsilon values
    for eps in eps_range:
        optics = OPTICS(max_eps=eps, min_samples=samples, metric='euclidean')
        optics.fit(df.values)  # Apply OPTICS to the data

        # Get the labels and count noise points (-1 is the label for noise)
        labels = optics.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
        n_noise = np.sum(labels == -1)

        # Store the results for this epsilon value
        new_row = {'eps': eps, 'Clusters': n_clusters, 'Noise': n_noise}
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

        optics_labels.append(labels)

    print("Results for different eps values:")
    print(results)  # Print the results dataframe

    # Evaluate silhouette and cohesion for OPTICS
    sil_vs_coh(df, optics_labels, eps_range, x_name='max-eps')


def similarity_matrix(df, labels):
    """
    Creates a similarity matrix (proximity matrix) and visualizes it as a heatmap, ordered by cluster labels.

    :param df: The input data (features) to be clustered
    :param labels: The cluster labels for each data point
    :return: None (displays a heatmap of the sorted proximity matrix)
    """

    labels = np.array(labels)
    # Compute the pairwise Euclidean distances between all pairs of points
    proximity_matrix = pairwise_distances(df, metric='euclidean')

    # Sort the proximity matrix by the cluster labels
    sorted_indices = np.argsort(labels)  # Get indices to sort the labels
    sorted_proximity_matrix = proximity_matrix[sorted_indices, :][:, sorted_indices]  # Reorder rows/columns
    sorted_labels = labels[sorted_indices]  # Reordered labels

    # Convert the sorted proximity matrix to a DataFrame for better visualization
    sorted_df = pd.DataFrame(sorted_proximity_matrix,
                             index=sorted_labels,
                             columns=sorted_labels)

    # Plot the sorted proximity matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sorted_df, cmap="rainbow", xticklabels=False, yticklabels=False)
    plt.title("Proximity Matrix Ordered by Cluster Labels")
    plt.show()
