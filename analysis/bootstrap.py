"""
bootstrap.py
------------
Time-series resampling utilities.

• moving_block_bootstrap
    Yields `n_boot` resampled return DataFrames using an
    overlapping moving-block scheme (Künsch, 1989).
    Preserves cross-sectional dependence (all tickers move together).

Public API
----------
moving_block_bootstrap(df, block_len=10, n_boot=500,
                       progress=False, random_state=None)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from tqdm.auto import tqdm

__all__ = ["moving_block_bootstrap"]


# ╔═══════════════════════════════════════════════════════╗
# 1. Core generator
# ╚═══════════════════════════════════════════════════════╝
def moving_block_bootstrap(
    returns: pd.DataFrame,
    block_len: int = 10,
    n_boot: int = 500,
    progress: bool = False,
    random_state: int | None = None,
):
    """
    Parameters
    ----------
    returns : DataFrame (T × N)  NaN-free
    block_len : length of each moving block (in rows)
    n_boot    : number of bootstrap replicates to generate
    progress  : if True, wrap loop in tqdm bar
    random_state : seed for reproducibility

    Yields
    ------
    resampled DataFrame (same shape as `returns`) on each iteration.
    """
    rng = np.random.default_rng(random_state)
    T, _ = returns.shape
    n_blocks = int(np.ceil(T / block_len))

    # Pre-compute overlapping blocks as numpy arrays
    blocks = [
        returns.iloc[i : i + block_len].values
        for i in range(T - block_len + 1)
    ]

    iterator = range(n_boot)
    if progress:
        iterator = tqdm(iterator, desc="bootstrap", unit="rep")

    for _ in iterator:
        draw_idx = rng.integers(0, len(blocks), size=n_blocks)
        sample   = np.concatenate([blocks[i] for i in draw_idx], axis=0)[:T]
        df_boot  = pd.DataFrame(sample, index=returns.index,
                                columns=returns.columns)
        yield df_boot
