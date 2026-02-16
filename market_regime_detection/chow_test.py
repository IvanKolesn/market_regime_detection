"""
Test for a known breakpoint
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f


def create_timeseries_x(n_obs: int) -> np.array:
    """
    Time trend + intersept
    """
    X = np.arange(1, n_obs + 1)[:, np.newaxis]
    X = sm.add_constant(X)
    return X


def chow_test(y: np.array, X: np.array, break_idx: int):
    """
    Perform a Chow test for a structural break at a known breakpoint.

    Parameters
    ----------
    y : array_like, shape (n,)
        Dependent variable (time series).
    X : array_like, shape (n, k)
        Independent variables, including a column of ones if an intercept is desired.
    break_idx : int
        Index separating the two periods. The first period includes observations
        0 .. break_idx-1, the second period break_idx .. n-1.

    Returns
    -------
    F : float
        Chow test F-statistic.
    p_value : float
        P-value of the test.
    df1, df2 : int
        Degrees of freedom used in the F-test.

    Raises
    ------
    ValueError
        If either subsample has fewer observations than the number of parameters.
    """
    n = len(y)
    k = X.shape[1]

    # Ensure the break index is valid and subsamples are large enough
    if break_idx < k or (n - break_idx) < k:
        raise ValueError(
            f"Each subsample must have at least {k} observations. "
            f"First period has {break_idx}, second has {n - break_idx}."
        )

    # Full sample regression (restricted model)
    model_full = sm.OLS(y, X).fit()
    rss_full = model_full.ssr

    # First subsample regression
    y1 = y[:break_idx]
    X1 = X[:break_idx, :]
    model1 = sm.OLS(y1, X1).fit()
    rss1 = model1.ssr

    # Second subsample regression
    y2 = y[break_idx:]
    X2 = X[break_idx:, :]
    model2 = sm.OLS(y2, X2).fit()
    rss2 = model2.ssr

    # Chow test statistic
    F = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2 * k))
    df1 = k
    df2 = n - 2 * k
    p_value = 1 - f.cdf(F, df1, df2)

    return F, p_value, df1, df2
