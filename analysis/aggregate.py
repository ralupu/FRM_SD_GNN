import pandas as pd

def aggregate_to_sector(
    prices: pd.DataFrame,
    meta: pd.DataFrame,
    weight_col: str | None = None
) -> pd.DataFrame:
    """
    Aggregate asset-level prices to sector-level indices.

    Parameters
    ----------
    prices    : DataFrame, shape (T × N)
                Index = dates, columns = ticker symbols
    meta      : DataFrame, index = ticker symbols, must contain 'Sector' column
    weight_col: optional column in `meta` for weighting (e.g., market cap).
                If None, uses simple equally-weighted average.

    Returns
    -------
    DataFrame, shape (T × S)
      Index = dates, columns = sector names
    """
    # Filter to tickers present in both
    tickers = [t for t in prices.columns if t in meta.index]
    df = prices[tickers]

    # Map ticker → sector
    sectors = meta.loc[tickers, 'Sector']

    if weight_col and weight_col in meta.columns:
        weights = meta.loc[tickers, weight_col]
        # normalize weights within each sector
        weights = weights.div(weights.groupby(sectors).transform('sum'))
        # weighted average: (prices * weights).groupby(axis=1, by=sectors).sum()
        weighted = df * weights
        sector_df = weighted.groupby(by=sectors, axis=1).sum()
    else:
        # equal-weighted average
        df.columns = sectors
        sector_df = df.groupby(axis=1, level=0).mean()

    return sector_df
