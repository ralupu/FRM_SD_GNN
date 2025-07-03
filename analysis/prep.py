"""
prep.py
-------
Utility functions that take a wide (Date × Ticker) price or return matrix
and return: prices, simple returns, log returns, and a melted long-form DF.
Supports any frequency (daily, weekly, monthly) as provided.
"""

import pandas as pd
import numpy as np

# ╔════════════════════════════════════════════════╗
# 1. Return calculations
# ╚════════════════════════════════════════════════╝
def _final_return_cleanup(ret_df: pd.DataFrame, ffill_limit: int = 3) -> pd.DataFrame:
    """
    Ensure the return matrix is free of NaNs:
      • drop the first row created by diff()
      • forward-fill tiny gaps (≤ ffill_limit)
      • *then* drop any residual rows/cols that still contain NaNs
    """
    ret_df = (
        ret_df.iloc[1:]                         # toss the all-NaN first row
               .ffill(limit=ffill_limit)        # small holes
               .dropna(axis=0, how="any")       # purge rows with remaining NaNs
               .dropna(axis=1, how="any")       # purge columns with remaining NaNs
    )
    return ret_df

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = P_t / P_{t-1} - 1   (preserves NaNs where data is missing). Works for any frequency."""
    ret = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    return _final_return_cleanup(ret)

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t) - ln(P_{t-1}); works for any frequency."""
    ret = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)
    return _final_return_cleanup(ret)

# ╔════════════════════════════════════════════════╗
# 2. Tidying helpers
# ╚════════════════════════════════════════════════╝
def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide (Date index, tickers in columns) → long
    with columns: Date | Ticker | <value_name>.
    """
    long_df = (
        df.reset_index()
          .rename(columns={df.index.name or df.index.names[0]: "Date"})
          .melt(id_vars="Date", var_name="Ticker", value_name=value_name)
          .dropna(subset=[value_name])
    )
    return long_df

# ╔════════════════════════════════════════════════╗
# 3. Convenience bundle
# ╚════════════════════════════════════════════════╝
def prepare_all(prices: pd.DataFrame, sample: int | None = None) -> dict:
    """
    Optionally sub-samples a random set of columns first (for fast smoke-runs).
    Returns a dict with:
        - prices        (input, no further cleaning)
        - ret_simple    (pct returns)
        - ret_log       (log returns)
        - long_prices   (tidy, long-form)
        - long_log_ret  (tidy, long-form)
    """
    if sample and sample < prices.shape[1]:
        keep = np.random.default_rng(42).choice(prices.columns, size=sample, replace=False)
        prices = prices[keep]

    ret_simple = compute_simple_returns(prices)
    ret_log    = compute_log_returns(prices)

    out = {
        "prices":       prices,
        "ret_simple":   ret_simple,
        "ret_log":      ret_log,
        "long_prices":  wide_to_long(prices, "Price"),
        "long_log_ret": wide_to_long(ret_log, "LogRet"),
    }
    return out
