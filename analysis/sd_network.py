"""
sd_network.py
-------------
Construct a directed dominance graph from a single FRM λ‐vector or a bootstrap samples matrix.
If you pass only scalar λ (one per asset), this function falls back to a simple comparison:
edge i->j if λ_i > λ_j.
If you pass bootstrap samples (arrays for each asset), it runs the SD test.

Usage:
    from analysis.sd_network import dominance_graph_single
"""

import networkx as nx
import numpy as np
from analysis.sd_utils import sd_stat_pvalue

def dominance_graph_single(
    lambda_vec,
    s: int = 1,
    alpha: float = 0.05,
    ngrid: int = 100,
    nboot: int = 200,
    debug: bool = False
) -> nx.DiGraph:
    """
    Build a directed graph G where nodes are tickers (index of lambda_vec)
    and there is an edge i -> j if sample_i SD-dominates sample_j
    (or if lambda_i > lambda_j when working with scalars).

    Parameters
    ----------
    lambda_vec : pd.Series, list, or np.ndarray
        If 1D with length >1: interpreted as a bootstrap sample of lambdas for each ticker.
        If 1D with length == N assets and we want cross-sectional comparison:
            pass in a pd.Series of length N where each entry is itself a scalar or array.
        If comparing scalars, pass a length-N iterable of scalars.
    s : int
        Order of stochastic dominance (1=FSD, 2=SSD, …).
    alpha : float
        Significance threshold for the bootstrap p-value.
    ngrid : int
        # of ECDF grid points for the SD test.
    nboot : int
        # of bootstrap replications in the SD test.
    debug : bool
        If True, forward debug to sd_stat_pvalue.

    Returns
    -------
    G : networkx.DiGraph
        Directed graph where edge i->j indicates dominance.
    """
    # Ensure Series-like interface
    try:
        tickers = list(lambda_vec.index)
        samples = list(lambda_vec.values)
    except AttributeError:
        # lambda_vec may be dict-like or list
        tickers = list(range(len(lambda_vec)))
        samples = list(lambda_vec)

    N = len(tickers)
    G = nx.DiGraph()
    G.add_nodes_from(tickers)

    for i in range(N):
        sample_i = samples[i]
        # Ensure sample_i is an array for size checks
        arr_i = np.atleast_1d(sample_i)

        for i in range(N):
            sample_i = samples[i]
            arr_i = np.atleast_1d(sample_i)
            for j in range(N):
                if i == j:
                    continue
                sample_j = samples[j]
                arr_j = np.atleast_1d(sample_j)
                # DIAGNOSTIC
                if debug:
                    print(f"Comparing {tickers[i]} (size={arr_i.size}) vs {tickers[j]} (size={arr_j.size})")
                if arr_i.size == 1 and arr_j.size == 1:
                    if arr_i.item() > arr_j.item():
                        G.add_edge(
                            tickers[i],
                            tickers[j],
                            weight=float(arr_i.item() - arr_j.item()),
                            pvalue=0.0
                        )
                    continue
                # SD mode
                stat, pval = sd_stat_pvalue(
                    x=arr_i,
                    y=arr_j,
                    s=s,
                    nboot=nboot,
                    ngrid=ngrid,
                    debug=debug
                )
                if (stat > 0) and (pval < alpha):
                    G.add_edge(
                        tickers[i],
                        tickers[j],
                        weight=float(stat),
                        pvalue=float(pval)
                    )

    return G
