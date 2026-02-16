"""
Market regime changes detection
using Wasserstein K-Means algorithm
"""

import numpy as np
from tqdm import trange


def split_array_into_windows(market_returns: np.ndarray, window_shape: int):
    """
    Split market returns array into overlapping arrays
    with each having "window_shape" days in it
    """
    chunks = np.lib.stride_tricks.sliding_window_view(
        market_returns, window_shape=window_shape
    )

    chunks = np.sort(chunks, axis=1)

    return chunks


def sliced_wasserstein_distance(
    X: np.ndarray, Y: np.ndarray, max_iter: int = 50, seed: int = 42
):
    """
    X, Y : arrays of shape (N, d)
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    distances = []

    for _ in range(max_iter):
        theta = rng.normal(size=d)
        theta /= np.linalg.norm(theta)

        proj_X = np.sort(X @ theta)
        proj_Y = np.sort(Y @ theta)

        distances.append(np.mean(np.abs(proj_X - proj_Y)))

    return np.mean(distances)


def sliced_wasserstein_barycenter(
    cluster: np.ndarray, max_iter: int = 50, seed: int = 42
):
    rng = np.random.default_rng(seed)
    d = cluster[0].shape[1]
    barycenter = np.zeros_like(cluster[0])

    for _ in range(max_iter):
        theta = rng.normal(size=d)
        theta /= np.linalg.norm(theta)

        projections = [np.sort(c @ theta) for c in cluster]
        median_proj = np.median(np.stack(projections), axis=0)

        barycenter += np.outer(median_proj, theta)

    return barycenter / max_iter


def wasserstein_k_means_multivariate(
    measures: np.ndarray,
    n_clusters: int,
    max_iter: int = 50,
    tol: float = 1e-3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    centroids = rng.choice(measures, size=n_clusters, replace=False)

    for _ in trange(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        assignments = []

        for X in measures:
            dists = [
                sliced_wasserstein_distance(X, c, max_iter, seed) for c in centroids
            ]
            cid = np.argmin(dists)
            clusters[cid].append(X)
            assignments.append(cid)

        new_centroids = []
        for c in clusters:
            if len(c) == 0:
                new_centroids.append(rng.choice(measures))
            else:
                new_centroids.append(sliced_wasserstein_barycenter(c, max_iter, seed))

        shift = sum(
            sliced_wasserstein_distance(a, b, max_iter, seed)
            for a, b in zip(centroids, new_centroids)
        )

        centroids = new_centroids
        if shift < tol:
            break

    return assignments, clusters, centroids


def get_corrected_assignments(assignments: np.ndarray, window_shape: int) -> list[int]:
    """
    Since one observation falls into multiple chunks
    we have to detect which market regime it appears in
    the most

    ---
    Parameters
    ---
    assignments: np.ndarray:
        result of clusterization
    window_shape: int
        days in one chunk
    """

    corrected_assignments = []

    for i in range(len(assignments)):
        subset = assignments[
            max(0, i - window_shape) : min(len(assignments), i + window_shape)
        ]
        correct_cluster = np.argmax(np.bincount(subset))
        corrected_assignments.append(correct_cluster)

    return corrected_assignments
