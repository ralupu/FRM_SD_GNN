import numpy as np
import pandas as pd

def bootstrap_history(
    returns: pd.DataFrame,
    H: int,
    block_len: int,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Generate one bootstrap-resampled history of length H using moving-block bootstrap.

    Parameters
    ----------
    returns    : DataFrame of shape (T × N), indexed by Date
    H          : number of days to draw in the synthetic history
    block_len  : length of each moving block
    random_state: seed for reproducibility

    Returns
    -------
    DataFrame of shape (H × N) with bootstrap sample, index = last H dates
    """
    rng = np.random.default_rng(random_state)
    T, N = returns.shape
    # Ensure H <= T
    if H > T:
        raise ValueError(f"H={H} longer than available returns length T={T}")

    # Precompute overlapping blocks
    blocks = [returns.values[i:i+block_len] for i in range(T - block_len + 1)]
    n_blocks = int(np.ceil(H / block_len))

    # Draw block indices
    idx = rng.integers(0, len(blocks), size=n_blocks)
    # Concatenate and trim to length H
    sampled = np.vstack([blocks[i] for i in idx])[:H]

    # Build DataFrame with same columns and use the last H dates as index
    sampled_df = pd.DataFrame(
        sampled,
        index=returns.index[-H:],
        columns=returns.columns
    )
    return sampled_df
