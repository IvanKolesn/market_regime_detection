"""
Implements statistical tests for breakpoint detection in time series.
- detect_mean_breaks: multiple breaks in the mean (Bai–Perron approach)
- detect_variance_breaks: multiple breaks in the variance (constant mean)
"""

import numpy as np

# from tqdm import trange


def _rss_segment(y: np.ndarray, start: int, end: int) -> float:
    """
    Sum of squared deviations from the mean of y[start:end] (exclusive end).
    """
    seg = y[start:end]
    if len(seg) == 0:
        return 0.0
    return np.sum((seg - np.mean(seg)) ** 2)


def _neg_loglik_var_segment(
    y: np.ndarray, start: int, end: int, global_mean: float
) -> float:
    """
    Negative log‑likelihood kernel (0.5 * n * log(variance)) for a segment,
    assuming a common global mean and segment‑specific variance.
    """
    seg = y[start:end]
    if len(seg) == 0:
        return 0.0
    var = np.mean((seg - global_mean) ** 2)
    if var == 0:
        return np.inf
    return 0.5 * len(seg) * np.log(var)


def _optimal_breaks_dp(
    y: np.ndarray, max_breaks: int, min_segment: int, cost_func, **cost_kwargs
):
    """
    Generic dynamic programming to find optimal break positions.
    cost_func(y, i, j, **cost_kwargs) returns the cost for segment [i, j).
    Returns (breaks_for_k, cost_for_k) where breaks_for_k[k] is a list of
    break indices for k breaks, and cost_for_k[k] is the total cost.
    """
    T = len(y)

    # Precompute segment costs for all intervals of len [min_segment, 5 * min_segment]
    seg_cost = np.full((T, T + 1), 0)

    for i in range(T):
        for j in range(i + min_segment, T + 1):
            seg_cost[i, j] = _rss_segment(y, i, j)

    # dp[k][t] = minimum cost for first t observations with k breaks
    dp = [[np.inf] * (T + 1) for _ in range(max_breaks + 1)]
    breaks = [[None] * (T + 1) for _ in range(max_breaks + 1)]

    # 0 breaks
    for t in range(1, T + 1):
        dp[0][t] = seg_cost[0, t]
        breaks[0][t] = []

    # k >= 1
    for k in range(1, max_breaks + 1):
        for t in range(1, T + 1):
            best = np.inf
            best_br = None
            # last break at s (s < t)
            for s in range(k, t):
                if dp[k - 1][s] < np.inf and seg_cost[s, t] < np.inf:
                    cand = dp[k - 1][s] + seg_cost[s, t]
                    if cand < best:
                        best = cand
                        best_br = breaks[k - 1][s] + [s]
            if best < np.inf:
                dp[k][t] = best
                breaks[k][t] = best_br

    breaks_for_k = [breaks[k][T] for k in range(max_breaks + 1)]
    cost_for_k = [dp[k][T] for k in range(max_breaks + 1)]

    return breaks_for_k, cost_for_k


def detect_mean_breaks(
    y: np.ndarray, max_breaks: int = None, min_segment: int = 10
) -> tuple:
    """
    Detect multiple breaks in the mean (constant variance) using the
    Bai–Perron dynamic programming algorithm with BIC.

    Parameters
    ----------
    y : array_like
        Time series data.
    max_breaks : int, optional
        Maximum number of breaks to consider. If None, set to
        min(5, T // (2 * min_segment)).
    min_segment : int, default=10
        Minimum number of observations in each segment.

    Returns
    -------
    breaks : list of int
        Estimated break positions (indices just before the break, i.e.,
        the last index of the previous segment). Sorted.
    n_breaks : int
        Number of breaks selected by BIC.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    if max_breaks is None:
        max_breaks = min(5, T // (2 * min_segment))

    breaks_for_k, rss_for_k = _optimal_breaks_dp(
        y, max_breaks, min_segment, _rss_segment
    )

    # BIC: T * log(RSS/T) + (2 * k + 1) * log(T)
    # where k = number of breaks (each break adds one parameter, plus one global variance).
    bic_values = []
    for k in range(max_breaks + 1):
        rss = rss_for_k[k]
        if rss <= 0:
            bic = np.inf
        else:
            bic = T * np.log(rss / T) + (2 * k + 1) * np.log(T)
        bic_values.append(bic)
    print(bic_values)
    best_k = int(np.argmin(bic_values))
    return breaks_for_k[best_k], best_k


def detect_variance_breaks(
    y: np.ndarray, max_breaks: int = None, min_segment: int = 10
) -> tuple:
    """
    Detect multiple breaks in the variance, assuming a constant mean
    (estimated globally). Uses dynamic programming with a cost based on
    the negative log‑likelihood, and selects the number of breaks with BIC.

    Parameters
    ----------
    y : array_like
        Time series data.
    max_breaks : int, optional
        Maximum number of breaks to consider. If None, set to
        min(5, T // (2 * min_segment)).
    min_segment : int, default=10
        Minimum number of observations in each segment.

    Returns
    -------
    breaks : list of int
        Estimated break positions (indices just before the break).
    n_breaks : int
        Number of breaks selected by BIC.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    global_mean = np.mean(y)

    if max_breaks is None:
        max_breaks = min(5, T // (2 * min_segment))

    breaks_for_k, cost_for_k = _optimal_breaks_dp(
        y, max_breaks, min_segment, _neg_loglik_var_segment, global_mean=global_mean
    )

    # BIC: 2 * total_cost + (k + 2) * log(T)
    # total_cost = sum of 0.5 * n_j * log(var_j)  (negative log‑likelihood)
    # 2 * total_cost = sum n_j * log(var_j)
    # Number of parameters: (k+1) variances + 1 global mean = k+2
    bic_values = []
    for k in range(max_breaks + 1):
        total_cost = cost_for_k[k]
        if np.isinf(total_cost):
            bic = np.inf
        else:
            bic = 2 * total_cost + (k + 2) * np.log(T)
        bic_values.append(bic)
    print(bic_values)
    best_k = int(np.argmin(bic_values))
    return breaks_for_k[best_k], best_k
