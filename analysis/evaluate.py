"""
evaluate.py
===========
Summarize and validate bootstrap-history outputs and create labels.

Functions
---------
summarize_frm_bootstrap(path, output_dir=None)
summarize_sd_centrality_bootstrap(path, output_dir=None)
plot_frm_fan_chart(fan_csv)
make_labels(frm_idx, h, q)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

def summarize_frm_bootstrap(
    path: str | Path,
    output_dir: str | Path | None = None
) -> pd.DataFrame:
    """
    Load frm_bootstrap.csv, compute mean and std of FRM across draws.

    Returns a DataFrame with columns ['mean', 'std'].

    If output_dir is given, saves a CSV and fan-chart data.
    """
    # 1. Read CSV
    df_full = pd.read_csv(path, index_col=0)
    # If single column, extract as Series
    if df_full.shape[1] == 1:
        df = df_full.iloc[:, 0]
    else:
        df = df_full

    # 2. Compute stats
    mean = df.mean()
    std  = df.std()
    summary = pd.DataFrame({'mean': mean, 'std': std}, index=['FRM'])

    # 3. Optionally save
    if output_dir:
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        summary.to_csv(out / "frm_bootstrap_summary.csv")
        fan = pd.Series({'lower': mean - 1.96*std, 'upper': mean + 1.96*std})
        fan.to_csv(out / "frm_bootstrap_fan.csv")

    return summary

def summarize_sd_centrality_bootstrap(
    path: str | Path,
    output_dir: str | Path | None = None
) -> pd.DataFrame:
    """
    Load sd_cent_bootstrap.csv, compute per-ticker mean and 5th/95th percentiles.

    Returns a DataFrame indexed by ticker, with columns ['mean','p5','p95'].
    """
    df = pd.read_csv(path, index_col=0)

    # Coerce all to numeric; drop columns that fail entirely
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    mean = df.mean(axis=0)
    p5   = df.quantile(0.05, axis=0)
    p95  = df.quantile(0.95, axis=0)
    summary = pd.DataFrame({'mean': mean, 'p5': p5, 'p95': p95})

    if output_dir:
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        summary.to_csv(out / "sd_cent_bootstrap_summary.csv")

    return summary


def plot_frm_fan_chart(fan_csv: str | Path):
    """
    Quick plot of the FRM fan chart from fan-chart data.
    """
    fan = pd.read_csv(fan_csv, index_col=0)
    plt.figure()
    fan.plot(y=['lower','upper'], kind='line', linewidth=1)
    plt.fill_between(
        fan.index, fan['lower'], fan['upper'], alpha=0.2,
        label='95% CI'
    )
    plt.title("Bootstrapped FRM Fan Chart")
    plt.legend()
    plt.tight_layout()
    plt.show()

def make_labels(
    frm_idx: pd.Series,
    h: int = 5,
    q: float = 0.90
) -> pd.Series:
    """
    Create binary FRM-jump labels.

    y_t = 1 if frm_idx[t + h] - frm_idx[t] > threshold,
    where threshold = (frm_idx.diff(h)).quantile(q).

    Returns a Series with the same index as frm_idx.
    """
    delta = frm_idx.shift(-h) - frm_idx
    thr = delta.quantile(q)
    y = (delta > thr).astype(int)
    return y
