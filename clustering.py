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


def transform(df, categorical_cols=[], standardize_cols=[], minmax_cols=[], samples=0):
    if samples != 0:
        df = df.sample(n=samples, random_state=1804)

    # Preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ("standardize", StandardScaler(), standardize_cols),
            ("minmax", MinMaxScaler(), minmax_cols),
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
        ]
    )
    transformed_data = preprocessor.fit_transform(df)
    return transformed_data, preprocessor


def inverse_transformation(
    transformed_data,
    preprocessor,
    categorical_cols=[],
    standardize_cols=[],
    minmax_cols=[],
):
    # Extract transformed numerical and categorical parts
    standardize_transformer = preprocessor.named_transformers_["standardize"]
    minmax_transformer = preprocessor.named_transformers_["minmax"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    # Separate numerical and categorical data
    standardize_data = transformed_data[:, : len(standardize_cols)]
    minmax_data = transformed_data[
        :, len(standardize_cols) : len(standardize_cols) + len(minmax_cols)
    ]
    cat_data = transformed_data[:, len(standardize_cols) + len(minmax_cols) :]

    # Inverse the transformations
    inverse_standardize_data = standardize_transformer.inverse_transform(
        standardize_data
    )
    inverse_minmax_data = minmax_transformer.inverse_transform(minmax_data)
    inverse_cat_data = cat_transformer.inverse_transform(cat_data)

    # Recombine columns
    return pd.DataFrame(
        np.hstack((inverse_standardize_data, inverse_minmax_data, inverse_cat_data)),
        columns=standardize_cols + minmax_cols + categorical_cols,
    )


def plot_sse_vs_clusters(kmeans_list, sse):
    # Create a DataFrame for SSE values
    num_clusters = [kmeans.n_clusters for kmeans in kmeans_list]
    sse_df = pd.DataFrame({"k": num_clusters, "SSE": sse})

    # Plot the SSE values
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="k", y="SSE", data=sse_df, marker="o")
    plt.title("SSE vs. Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.show()


def kmeans_explore(
    transformed_df,
    comps=0,
    max_k=40,
    random_state=1804,
    stride=1,
    exponential_stride=False,
):
    transformed_data = transformed_df.values
    # Reduce dims if requested.
    if comps != 0:
        pca = PCA(n_components=comps)
        transformed_data = pca.fit_transform(transformed_data)

    # Compute SSE for different cluster sizes
    sse = []
    kmeans_list = []
    if exponential_stride:
        k_values = [int(stride**i) for i in range(int(math.log2(max_k)) + 1)]
    else:
        k_values = range(1, max_k + 1, stride)

    for k in k_values:
        print(f"Computing KMeans for k={k}")
        kmeans = KMeans(
            n_clusters=k, init="k-means++", n_init=5, random_state=random_state
        )
        kmeans.fit(transformed_data)
        sse.append(kmeans.inertia_)
        kmeans_list.append(kmeans)

    return kmeans_list, sse


def save_kmeans_results(kmeans_list, sse, output_file):
    with lz4.frame.open(output_file, "wb+") as f:
        pickle.dump((kmeans_list, sse), f)


def optimal_kmeans(df, cat_cols, stand_cols, minmax_cols, n_clusters, fname):
    df = df.copy()
    transformed_data, preproc = transform(df, cat_cols, stand_cols, minmax_cols)

    # Explore more after we find the best K
    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", n_init=50, random_state=1804
    )
    kmeans.fit(transformed_data)

    df["Cluster"] = kmeans.labels_
    df["Distance_to_Centroid"] = kmeans.transform(transformed_data).min(axis=1)

    # Inverse transform centroids for interpretability
    centroids_df = inverse_transformation(
        kmeans.cluster_centers_, preproc, cat_cols, stand_cols, minmax_cols
    )
    centroids_df["Cluster"] = centroids_df.index

    df.to_csv(f"{fname}-kmeans-result.csv", index=False)
    centroids_df.to_csv(f"{fname}-kmeans_centroids.csv", index=False)

    return df, centroids_df, kmeans


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


def plot_hc(
    ax, data_link, method, metric, color_threshold=10, truncate_mode="lastp", p=12
):
    if data_link is None:
        print(f"No data to plot for {method}-{metric}")
        return

    dendrogram(
        data_link,
        color_threshold=color_threshold,
        truncate_mode=truncate_mode,
        p=p,
        no_labels=False,
        ax=ax,
    )
    ax.set_title(f"{method.capitalize()} - {metric.capitalize()}")


def plot_all_hc(results, metric, color_threshold=10, truncate_mode="lastp", p=12):
    num_methods = len(results)
    fig, axes = plt.subplots(1, num_methods, figsize=(7 * num_methods, 5))

    if num_methods == 1:
        axes = [axes]

    for ax, (method, data_link) in zip(axes, results.items()):
        plot_hc(ax, data_link, method, metric, color_threshold, truncate_mode, p)

    plt.tight_layout()
    plt.show()


def compute_distance_matrix(df, metric="euclidean", cat_features=[], cache_file=None):
    if cache_file is not None and os.path.exists(cache_file):
        with lz4.frame.open(cache_file, "rb") as f:
            distance_matrix = pickle.load(f)
        return distance_matrix

    if metric == "gower":
        distance_matrix = gower.gower_matrix(df, cat_features=cat_features)
    else:
        distance_matrix = pdist(df, metric=metric)

    if cache_file is not None:
        with lz4.frame.open(cache_file, "wb") as f:
            pickle.dump(distance_matrix, f)

    return distance_matrix


"""
X: La matrice dei dati, dove ogni riga rappresenta un punto e ogni colonna una caratteristica.
k: Numero di vicini più prossimi da considerare per ogni punto (default: 5).
strata: Numero di strati in cui suddividere le distanze medie dei vicini (default: 10).
m: Numero di campioni da prelevare da ciascuno strato per costruire il grafico delle distanze (default: 5).
"""
from sklearn.neighbors import NearestNeighbors

def stratified_sampling(X, k=5, strata=10, m=5):
    # Calcolo delle distanze k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Media delle distanze dei k-nearest neighbors per ogni punto
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Stratificazione delle distanze
    strata_limits = np.linspace(min(avg_distances), max(avg_distances), strata + 1)
    sampled_indices = []
    for i in range(strata):
        stratum_indices = np.where((avg_distances >= strata_limits[i]) & (avg_distances < strata_limits[i + 1]))[0]
        if len(stratum_indices) > 0:
            sampled_indices.extend(np.random.choice(stratum_indices, min(m, len(stratum_indices)), replace=False))
    return X.iloc[sampled_indices]

