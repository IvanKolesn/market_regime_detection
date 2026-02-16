"""
Market regime changes detection
using Wasserstein K-Means algorithm
"""

import numpy as np

from scipy.stats import wasserstein_distance


def split_array_into_windows(market_returns: np.array, window_shape: int):
    """
    Split market returns array into overlapping arrays
    with each having "window_shape" days in it
    """
    chunks = np.lib.stride_tricks.sliding_window_view(
        market_returns, window_shape=window_shape
    )

    chunks = np.sort(chunks, axis=1)

    return chunks


def wasserstein_barycenter(cluster):
    return np.median(np.stack(cluster), axis=0)


def wasserstein_k_means(
    chunks: np.array,
    n_clusters: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple:
    """
    ----
    Parameters
    ----
    Chunks: np.array
        previously split data [n_chunks, n_days]
        (empirical distibutions or measures)
    n_clusters:  int
        number of market regimes to detect
    max_iter: int
        number of attempts
    tol: float
        if average wassertine distance in cluster is less
        algorithm stops
    seed: int
        random seed fixed for reproducability
    """

    assert len(chunks.shape) == 2, "Incorrect dimension of data"

    rng = np.random.default_rng(seed)
    centroids = rng.choice(chunks, size=n_clusters, replace=False)

    for _ in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        assignments = []

        for mu in chunks:
            dists = [wasserstein_distance(mu, c) for c in centroids]
            cid = np.argmin(dists)
            clusters[cid].append(mu)
            assignments.append(cid)

        new_centroids = []

        for c in clusters:
            if len(c) == 0:
                new_centroids.append(rng.choice(chunks))
            else:
                new_centroids.append(wasserstein_barycenter(c))

        shift = sum(
            wasserstein_distance(a, b) for a, b in zip(centroids, new_centroids)
        )

        centroids = new_centroids

        if shift < tol:
            break

    return assignments, clusters, centroids


def get_corrected_assignments(assignments: np.array, window_shape: int) -> list[int]:
    """
    Since one observation falls into multiple chunks
    we have to detect which market regime it appears in
    the most

    ---
    Parameters
    ---
    assignments: np.array:
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
