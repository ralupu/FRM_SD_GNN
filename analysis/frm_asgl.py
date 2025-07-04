from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from asgl import Regressor
from tqdm.auto import tqdm
from contextlib import nullcontext

"""
frm_asgl.py
-----------
Optimized Financial Risk Meter (FRM) with optional bootstrap for stochastic dominance analysis.

Public API
----------
compute_frm(
    ret_log: pd.DataFrame,
    window: int,
    tau: float,
    n_jobs: int,
    progress: bool,
    step: int,
    lambda_grid: list[float],
    n_folds: int,
    bootstrap: int = 0
) -> dict
"""

def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return np.maximum(tau * e, (tau - 1) * e).mean()

def _fit_one_asset(
    y: np.ndarray,
    X: np.ndarray,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
) -> tuple[float, np.ndarray]:
    cv = TimeSeriesSplit(n_splits=n_folds)
    best_lambda, best_loss, best_coef = None, np.inf, None
    for lam in lambda_grid:
        losses = []
        for train_idx, test_idx in cv.split(X):
            model = Regressor(
                model='qr',
                penalization='lasso',
                quantile=tau,
                lambda1=lam,
                fit_intercept=True,
            )
            model.fit(X[train_idx], y[train_idx])
            coef = getattr(model, "coef_", getattr(model, "coef", None))
            intercept = getattr(model, "intercept_", getattr(model, "intercept", 0.0))
            y_hat = X[test_idx].dot(coef) + intercept
            losses.append(_pinball_loss(y[test_idx], y_hat, tau))
        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lambda = lam
            best_coef = model.coef_.copy()
    return best_lambda, best_coef

def _bootstrap_one_asset(
    y: np.ndarray,
    X: np.ndarray,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
    n_boot: int,
    random_state: np.random.Generator = None
) -> np.ndarray:
    # Block bootstrap: sample rows with replacement
    n_obs = len(y)
    rng = random_state if random_state is not None else np.random.default_rng()
    lambdas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_obs, n_obs)
        y_b = y[idx]
        X_b = X[idx]
        lam, _ = _fit_one_asset(y_b, X_b, tau, lambda_grid, n_folds)
        lambdas.append(lam)
    return np.array(lambdas)

def _compute_period_scalar(
    ret_window: pd.DataFrame,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
) -> dict:
    tickers = ret_window.columns.to_list()
    Y = ret_window.values
    lambdas, betas = {}, {}
    for i, tgt in enumerate(tickers):
        y = Y[:, i]
        X = np.delete(Y, i, axis=1)
        lam_i, coef_i = _fit_one_asset(y, X, tau, lambda_grid, n_folds)
        full_beta = np.insert(coef_i, i, 0.0)
        lambdas[tgt] = lam_i
        betas[tgt] = full_beta
    return {'date': ret_window.index[-1], 'lambdas': lambdas, 'betas': betas}

def _compute_period_bootstrap(
    ret_window: pd.DataFrame,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
    n_boot: int,
    seed: int = 42
) -> dict:
    tickers = ret_window.columns.to_list()
    Y = ret_window.values
    lambdas, betas = {}, {}
    rng = np.random.default_rng(seed)
    for i, tgt in enumerate(tickers):
        y = Y[:, i]
        X = np.delete(Y, i, axis=1)
        lambdas[tgt] = _bootstrap_one_asset(y, X, tau, lambda_grid, n_folds, n_boot, rng)
        # For betas, just use fit on all data (optional)
        _, coef_i = _fit_one_asset(y, X, tau, lambda_grid, n_folds)
        full_beta = np.insert(coef_i, i, 0.0)
        betas[tgt] = full_beta
    return {'date': ret_window.index[-1], 'lambdas': lambdas, 'betas': betas}

def compute_frm(
    ret_log: pd.DataFrame,
    window: int = 12,
    tau: float = 0.05,
    n_jobs: int = -1,
    progress: bool = False,
    step: int = 1,
    lambda_grid: list[float] = None,
    n_folds: int = 3,
    bootstrap: int = 0,  # Number of bootstrap samples; if 0, use scalar mode
) -> dict:
    """
    Compute FRM (scalar or bootstrap) over log-returns.
    If bootstrap > 0, returns a lambda matrix of arrays (for SD).
    If bootstrap == 0, returns scalar matrix.
    """
    if n_folds < 2:
        print(f"[WARN] n_folds={n_folds} is too small; resetting to 2")
        n_folds = 2
    if lambda_grid is None:
        lambda_grid = list(10 ** np.linspace(-3, 1, 5))
    ret_log = ret_log.sort_index()
    dates = ret_log.index
    end_positions = range(window, len(dates) + 1, step)
    iterator = end_positions
    if progress:
        iterator = tqdm(end_positions, desc='FRM windows')
    results = []
    bar = tqdm(iterator, desc="FRM windows") if progress else iterator

    if bootstrap > 0:
        for pos in bar:
            res = _compute_period_bootstrap(
                ret_log.iloc[pos - window: pos, :],
                tau, lambda_grid, n_folds, bootstrap, seed=42+pos
            )
            results.append(res)
    else:
        for pos in bar:
            res = _compute_period_scalar(
                ret_log.iloc[pos - window: pos, :],
                tau, lambda_grid, n_folds
            )
            results.append(res)

    # Aggregate
    lambda_rows = []
    beta_panel = {}
    for r in results:
        lambda_rows.append(pd.Series(r['lambdas'], name=r['date']))
        beta_panel[r['date']] = r['betas']

    lambda_df = pd.DataFrame(lambda_rows).sort_index()
    frm_index = lambda_df.map(lambda x: np.mean(x) if isinstance(x, np.ndarray) else x).mean(axis=1)

    return {
        'frm_index': frm_index,
        'lambda_mat': lambda_df,
        'beta_panel': beta_panel
    }
