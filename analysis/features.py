"""
features.py
===========
Extracts features from FRM outputs and SD networks, and merges control variables.
"""

import pandas as pd
import networkx as nx

def compute_graph_centralities(G: nx.DiGraph) -> dict:
    indegree  = dict(G.in_degree())
    outdegree = dict(G.out_degree())
    pagerank  = nx.pagerank(G)
    eigen     = nx.eigenvector_centrality(G.to_undirected())
    features = {}
    for node in G.nodes():
        features[node] = {
            "indegree": indegree.get(node, 0),
            "outdegree": outdegree.get(node, 0),
            "pagerank": pagerank.get(node, 0.0),
            "eigenvector": eigen.get(node, 0.0),
        }
    return features

def make_feature_panel(frm_idx: pd.Series,
                       lambda_mat: pd.DataFrame,
                       sd_cent_df: pd.DataFrame,
                       cov_mes: pd.DataFrame,
                       lags: int = 5) -> pd.DataFrame:
    """
    Build feature panel X_panel indexed by date.

    Components:
      - lagged FRM index (t-1 ... t-lags)
      - raw lambda values at t (lambda_mat, one column per asset)
      - SD centralities at t (sd_cent_df, with prefixed column names)
      - CoVaR and MES controls (cov_mes, prefixed per asset)
    """
    # 1. Base DataFrame with index = FRM dates
    df = pd.DataFrame(index=frm_idx.index)

    # 2. Lagged FRM
    lag_cols = []
    for i in range(1, lags+1):
        col = f"frm_lag_{i}"
        df[col] = frm_idx.shift(i)
        lag_cols.append(col)

    # 3. Current lambda values (prefix to avoid conflicts)
    lamb = lambda_mat.copy().add_prefix("lambda_")
    df = df.join(lamb, how="left")

    # 4. SD centrality features (prefix)
    sd_feats = sd_cent_df.copy().add_prefix("sd_")
    df = df.join(sd_feats, how="left")

    # 5. CoVaR & MES controls: forward‐fill to daily frequency
    covmes = cov_mes.reindex(df.index).ffill()
    df = df.join(covmes, how="left")

    # 6. Drop only the initial rows where lagged FRM is NaN
    df = df.dropna(subset=lag_cols)

    return df
