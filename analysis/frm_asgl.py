from __future__ import annotations
import numpy as np, pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import TimeSeriesSplit
from asgl import Regressor
from tqdm.auto import tqdm
from common.tqdm_joblib import tqdm_joblib
from contextlib import nullcontext

"""
frm_asgl.py
-----------
Optimized Financial Risk Meter (FRM) via `asgl`'s Regressor + manual CV.
Supports configurable grid, CV folds, window stepping, and tqdm bars.

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
    n_folds: int
) -> dict
"""

# ╔════════════════════════════════════════════════════════════════╗
# 0. Utility: pinball loss
# ╚════════════════════════════════════════════════════════════════╝
def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return np.maximum(tau * e, (tau - 1) * e).mean()


# ╔════════════════════════════════════════════════════════════════╗
# 1. Fit one asset over one window with manual CV
# ╚════════════════════════════════════════════════════════════════╝
def _fit_one_asset(
    y: np.ndarray,
    X: np.ndarray,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
) -> tuple[float, np.ndarray]:
    """
    Select best lambda by minimizing out-of-sample pinball loss over CV folds.
    Returns (best_lambda, best_coef) where coef length = X.shape[1]
    """
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
            # manual predict to avoid __sklearn_tags__ bug
            # Extract fitted parameters (robust to attribute naming)
            # manual predict: guard against None and avoid ambiguous truth-check
            coef_attr = getattr(model, "coef_", None)
            coef = coef_attr if coef_attr is not None else getattr(model, "coef", None)
            intercept_attr = getattr(model, "intercept_", None)
            intercept = intercept_attr if intercept_attr is not None else getattr(model, "intercept", 0.0)
            y_hat = X[test_idx].dot(coef) + intercept
            losses.append(_pinball_loss(y[test_idx], y_hat, tau))
        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lambda = lam
            best_coef = model.coef_.copy()

    return best_lambda, best_coef


# ╔════════════════════════════════════════════════════════════════╗
# 2. Compute FRM snapshot for one date
# ╚════════════════════════════════════════════════════════════════╝
def _compute_day(
    ret_window: pd.DataFrame,
    tau: float,
    lambda_grid: list[float],
    n_folds: int,
) -> dict:
    tickers = ret_window.columns.to_list()
    Y = ret_window.values  # (window × N)

    lambdas, betas = {}, {}
    for i, tgt in enumerate(tickers):
        y = Y[:, i]
        X = np.delete(Y, i, axis=1)
        lam_i, coef_i = _fit_one_asset(y, X, tau, lambda_grid, n_folds)
        full_beta = np.insert(coef_i, i, 0.0)
        lambdas[tgt] = lam_i
        betas[tgt] = full_beta

    return {'date': ret_window.index[-1], 'lambdas': lambdas, 'betas': betas}


# ╔════════════════════════════════════════════════════════════════╗
# 3. Public API
# ╚════════════════════════════════════════════════════════════════╝
def compute_frm(
    ret_log: pd.DataFrame,
    window: int = 63,
    tau: float = 0.05,
    n_jobs: int = -1,
    progress: bool = False,
    step: int = 1,
    lambda_grid: list[float] = None,
    n_folds: int = 3,
) -> dict:
    """
    Compute FRM over `ret_log` with: rolling window, manual CV, and optional skipping.

    Parameters
    ----------
    ret_log      : wide DataFrame (dates × tickers) of log-returns (NaN-free)
    window       : look-back length (days)
    tau          : tail quantile level
    n_jobs       : joblib parallel workers
    progress     : show tqdm bar over windows
    step         : compute only every `step` days
    lambda_grid  : list of λ candidates
    n_folds      : CV splits

    Returns
    -------
    dict containing:
        frm_index  : pd.Series  (date → mean λ)
        lambda_mat : pd.DataFrame (dates × tickers)
        beta_panel : dict[date] → {ticker: coef array}
    """

    # ─── Enforce at least 2 CV splits ─────────────────────
    if n_folds < 2:
        print(f"[WARN] n_folds={n_folds} is too small; resetting to 2")
        n_folds = 2
    # ────────────────────────────────────────────────────────

    # defaults for grid
    if lambda_grid is None:
        # coarse 5-point grid
        lambda_grid = list(10 ** np.linspace(-3, 1, 5))

    ret_log = ret_log.sort_index()
    dates = ret_log.index
    end_positions = range(window, len(dates) + 1, step)

    iterator = end_positions
    if progress:
        iterator = tqdm(end_positions, desc='FRM windows')

    # SERIAL fallback (avoids joblib pickling issues)
    results = []
    bar = tqdm(iterator, desc="FRM windows") if progress else iterator
    for pos in bar:
        res = _compute_day(
            ret_log.iloc[pos - window: pos, :],
            tau, lambda_grid, n_folds
            )
        results.append(res)

    # aggregate
    lambda_rows = []
    beta_panel = {}
    for r in results:
        lambda_rows.append(pd.Series(r['lambdas'], name=r['date']))
        beta_panel[r['date']] = r['betas']

    lambda_df = pd.DataFrame(lambda_rows).sort_index()
    frm_index = lambda_df.mean(axis=1)

    return {'frm_index': frm_index,
            'lambda_mat': lambda_df,
            'beta_panel': beta_panel}
