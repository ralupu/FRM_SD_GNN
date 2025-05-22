"""
frm.py
------
Financial Risk Meter for equity returns (Härtle et al., 2019).
Core idea:
  • For each date t and each asset i
      – run an L1-penalised quantile regression (τ tail, rolling window)
      – choose the penalty λ_i,t via out-of-sample quantile loss
  • Node-level FRM_i,t  = chosen λ_i,t                 (tail vulnerability)
  • System FRM_t        = mean_i λ_i,t                (aggregate stress)
  • β̂_i,t              = sparse tail-dependence loadings → network edges
References: Mihoci et al. (2019) and Yu et al. (2020) :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit

# ────────────────────────────────────────────────────────────────────────────────
# 0.  Hyper-parameters (tweak later in config .yml or CLI)
# ────────────────────────────────────────────────────────────────────────────────
TAU           = 0.05          # tail level
WINDOW_DAYS   = 63            # three months (≈ FRM papers)
ALPHA_GRID    = np.logspace(-3, 1, 15)   # candidate λ (scikit alpha = λ / n)
N_JOBS        = -1            # use all CPUs


# ────────────────────────────────────────────────────────────────────────────────
# 1.  λ selection for ONE asset on ONE date
# ────────────────────────────────────────────────────────────────────────────────
def _choose_lambda(
    y: np.ndarray,
    X: np.ndarray,
    tau: float = TAU,
    alphas: np.ndarray = ALPHA_GRID,
) -> tuple[float, np.ndarray]:
    """
    Returns (best_alpha, coefficients) chosen by 3-fold expanding CV
    that minimises the average pinball (quantile) loss.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    cv_loss = []

    for a in alphas:
        qreg = QuantileRegressor(
            quantile=tau, alpha=a, solver="highs", fit_intercept=True
        )
        losses = []
        for train_idx, test_idx in tscv.split(X):
            qreg.fit(X[train_idx], y[train_idx])
            y_hat = qreg.predict(X[test_idx])
            e = y[test_idx] - y_hat
            loss = np.maximum(tau * e, (tau - 1) * e).mean()
            losses.append(loss)
        cv_loss.append(np.mean(losses))

    best_alpha = alphas[int(np.argmin(cv_loss))]
    final_model = QuantileRegressor(
        quantile=tau, alpha=best_alpha, solver="highs", fit_intercept=True
    ).fit(X, y)

    return best_alpha, final_model.coef_


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Daily FRM across all assets (parallel)
# ────────────────────────────────────────────────────────────────────────────────
def _compute_day(t_idx: int, ret_window: pd.DataFrame) -> Dict[str, object]:
    """
    Helper executed in parallel for one day (index t_idx refers to ending row).
    Returns dict with λ_i,t and β̂_i,t vectors.
    """
    tickers = ret_window.columns.to_list()
    Y = ret_window.values        # shape (window, N)
    results = {}

    # For each asset i, regress r_i on others
    for i, tgt in enumerate(tickers):
        y = Y[:, i]
        X = np.delete(Y, i, axis=1)  # drop target column
        alpha_i, coef_i = _choose_lambda(y, X)
        # Re-insert 0 for the self-coefficient to keep full length N
        full_beta = np.insert(coef_i, i, 0.0)
        results[tgt] = (alpha_i, full_beta)

    # Pack outputs
    lambdas = {k: v[0] for k, v in results.items()}
    betas   = {k: v[1] for k, v in results.items()}
    date    = ret_window.index[-1]
    return {"date": date, "lambdas": lambdas, "betas": betas}


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Public API
# ────────────────────────────────────────────────────────────────────────────────
def compute_frm(
    ret_log: pd.DataFrame,
    window: int = WINDOW_DAYS,
    tau: float = TAU,
    n_jobs: int = N_JOBS,
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    ret_log : Date-indexed, ticker-columns DataFrame of log-returns
    window   : rolling window length (business days)
    tau      : tail quantile level (e.g., 0.05)
    Returns
    -------
    dict with:
        • frm_index   (Series)
        • lambda_mat  (DataFrame: dates × tickers)
        • beta_panel  (dict[date] -> 2-D np.array of coefficients)
    """
    # Ensure chronological order
    ret_log = ret_log.sort_index()

    # Rolling start positions
    end_positions = range(window, len(ret_log) + 1)

    parallel_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_compute_day)(
            t_idx,
            ret_log.iloc[t_idx - window : t_idx, :],
        )
        for t_idx in end_positions
    )

    # Collect outputs
    lambda_records = []
    beta_panel = {}
    for d in parallel_results:
        lambda_records.append(pd.Series(d["lambdas"], name=d["date"]))
        beta_panel[d["date"]] = d["betas"]

    lambda_df = pd.DataFrame(lambda_records).sort_index()
    frm_index = lambda_df.mean(axis=1)

    return {
        "frm_index": frm_index,
        "lambda_mat": lambda_df,
        "beta_panel": beta_panel,
    }
