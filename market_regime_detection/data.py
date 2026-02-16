"""
Data loading and handling
"""

import yfinance as yf
import pandas as pd
import numpy as np


def compute_log_returns(data: pd.Series) -> pd.Series:
    """
    Computes log-returns from price columns
    in dataframe
    """
    return np.diff(np.log(data))


# todo: adjust price for stock splits
def load_yf_data(
    tickers: str | list[str],
    date_from: str = "2000-01-01",
    date_to: str = "2100-01-01",
) -> pd.DataFrame:
    """
    Loads data from yahoo finance
    """
    if isinstance(tickers, str):
        yf_tickers = yf.Ticker(tickers)
    else:
        yf_tickers = yf.Tickers(tickers)

    history = yf_tickers.history(start=date_from, end=date_to, interval="1d")

    if isinstance(tickers, str):
        return history[["Close"]].rename(columns={"Close": tickers})

    return history["Close"]
