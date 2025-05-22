"""
frm_asgl.py
-----------
Financial Risk Meter (FRM) – FAST version using `asgl`'s adaptive sparse-group
Lasso quantile regression (cv_lqreg).  Speed-ups vs. scikit implementation:

    • C++ backend with OpenMP
    • Built-in K-fold CV path => no Python loop over λ
    • Optional tqdm bar for large runs

Returns
-------
dict with keys
    'frm_index'  : pd.Series   (system-wide mean λ̂_t)
    'lambda_mat' : pd.DataFrame (dates × tickers λ̂_{i,t})
    'beta_panel' : dict[date] -> np.ndarray (N×N sparse tail-betas)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from asgl import cv_lqreg          # pip install asgl>=0.4.0
from tqdm.auto import tqdm
from common.tqdm_joblib import tqdm_joblib   # tiny helper we created


# ╔═══════════════════════════════════════════════════════╗
# 0. Internal: single asset fit for one window
# ╚═══════════════════════════════════════════════════════╝
def _fit_one_asset(y: np.ndarray,
                   X: np.ndarray,
                   tau: float = 0.05,
                   nfolds: int = 3,
                   nlambda: int = 30) -> tuple[float, np.ndarray]:
    """
    Returns (best_lambda, coeff_vector)
    coeff_vector length == X.shape[1]  (no self, no intercept)
    """
    model = cv_lqreg(
        X, y, tau=tau,
        nfolds=nfolds,
        nlambda=nlambda,
        family="gaussian",        # returns are continuous
        parallel=False            # outer joblib already parallelised
    )
    lam_best = float(model.lambda_best_)
    coef = model.coef_           # shape (p,)
    return lam_best, coef


# ╔═══════════════════════════════════════════════════════╗
# 1. Helper: one FRM snapshot for a given day (rolling window)
# ╚═══════════════════════════════════════════════════════╝
def _compute_day(ret_window: pd.DataFrame,
                 tau: float) -> dict:
    tickers = ret_window.columns.to_list()
    Y = ret_window.values              # (W × N)
    lambdas, betas = {}, {}

    for i, tgt in enumerate(tickers):
        y = Y[:, i]
        X = np.delete(Y, i, axis=1)

        lam_i, coef_i = _fit_one_asset(y, X, tau=tau)
        full_beta = np.insert(coef_i, i, 0.0)  # align length to N

        lambdas[tgt] = lam_i
        betas[tgt]   = full_beta

    return dict(date=ret_window.index[-1],
                lambdas=lambdas,
                betas=betas)


# ╔═══════════════════════════════════════════════════════╗
# 2. Public API
# ╚═══════════════════════════════════════════════════════╝
def compute_frm(ret_log: pd.DataFrame,
                window: int = 63,
                tau: float = 0.05,
                n_jobs: int = -1,
                progress: bool = False) -> dict:
    """
    Parameters
    ----------
    ret_log : DataFrame  (Date × Ticker, NaN-free)
    window  : rolling look-back (business days)
    tau     : tail quantile level
    n_jobs  : joblib workers
    progress: if True, show tqdm bar over dates

    Returns same dict structure as legacy frm.py
    """
    ret_log = ret_log.sort_index()
    end_pos = range(window, len(ret_log) + 1)

    # Wrap Parallel in tqdm if requested
    iterator = end_pos
    if progress:
        iterator = tqdm(end_pos, desc="FRM windows")

    with tqdm_joblib(tqdm(iterator)) if progress else nullcontext():
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_day)(
                ret_log.iloc[t - window: t], tau
            ) for t in end_pos
        )

    # Collect outputs
    lambda_rows = []
    beta_panel  = {}
    for r in results:
        lambda_rows.append(pd.Series(r['lambdas'], name=r['date']))
        beta_panel[r['date']] = r['betas']

    lambda_df = pd.DataFrame(lambda_rows).sort_index()
    frm_idx   = lambda_df.mean(axis=1)

    return dict(frm_index=frm_idx,
                lambda_mat=lambda_df,
                beta_panel=beta_panel)


# small helper when tqdm disabled
from contextlib import nullcontext
