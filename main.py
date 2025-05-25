#!/usr/bin/env python3
"""
main.py
--------
End-to-end driver for STOXX-600 FRM/SD pipeline with bootstrapped inference,
evaluation summaries, label creation, baseline models, and correct SD centrality flattening.
"""

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from analysis import prep
from analysis.aggregate     import aggregate_to_sector
from analysis.covar_mes     import load_controls
from analysis.cov_bootstrap import bootstrap_history
from analysis.frm_asgl      import compute_frm
from analysis.sd_utils      import sd_stat_pvalue
from analysis.sd_network    import dominance_graph_single
from analysis.features      import compute_graph_centralities, make_feature_panel
from analysis.evaluate      import (
    summarize_frm_bootstrap,
    summarize_sd_centrality_bootstrap,
    make_labels,
)
from analysis.model         import (
    train_logistic_lasso,
    eval_logistic,
    train_var,
    forecast_var
)

# ───────────────────────────────────────────────────────────────
# 1. CLI & config
# ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FRM/SD pipeline")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly keep only N assets (asset-level only)")
    p.add_argument("--n_jobs", type=int, default=None,
                   help="override number of parallel jobs for FRM")
    p.add_argument("--level", choices=["asset", "sector"], default="asset",
                   help="run at 'asset' or 'sector' aggregation level")
    p.add_argument("--config", type=Path, default=Path("config.yml"),
                   help="path to YAML config")
    return p.parse_args()

def load_cfg(path: Path) -> dict:
    defaults = dict(
        window_days    = 63,
        tau            = 0.05,
        step           = 1,
        lambda_grid    = [0.001,0.01,0.1,1.0,10.0],
        n_folds        = 2,
        n_jobs         = 4,
        block_len      = 10,
        n_boot         = 200,
        alpha_sd       = 0.05,
        ngrid          = 100,
        H              = 21,
        B              = 50,
        horizon        = 5,
        jump_quantile  = 0.90,
        var_lags       = 1,  # lags for VAR
    )
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            defaults.update(cfg)
        except yaml.YAMLError:
            print(f"[WARN] could not parse {path}; using defaults")
    return defaults

# ───────────────────────────────────────────────────────────────
# 2. Data loaders & cleaning
# ───────────────────────────────────────────────────────────────
def load_closing_prices(path: Path) -> pd.DataFrame:
    return (
        pd.read_excel(path, engine="openpyxl")
          .rename(columns=str.strip)
          .assign(Date=lambda d: pd.to_datetime(d["Date"], dayfirst=True))
          .set_index("Date")
          .sort_index()
    )

def load_company_meta(path: Path) -> pd.DataFrame:
    return (
        pd.read_excel(path, engine="openpyxl")
          .rename(columns=str.strip)
          .drop_duplicates(subset="Ticker")
          .set_index("Ticker")
    )

def initial_clean(prices: pd.DataFrame) -> pd.DataFrame:
    return (
        prices.apply(pd.to_numeric, errors="coerce")
              .dropna(axis=1, how="all")
              .ffill(limit=3)
    )

# ───────────────────────────────────────────────────────────────
# 3. Main workflow
# ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    root     = Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs  = root / "outputs"
    outputs.mkdir(exist_ok=True)

    # Paths to data files
    price_file = data_dir / "ClosingPrices.xlsx"
    meta_file  = data_dir / "names_sectors_28Mar25.xlsx"
    covar_file = data_dir / "covar.csv"
    mes_file   = data_dir / "mes.csv"

    # 3.1 Load & clean prices
    prices = initial_clean(load_closing_prices(price_file))
    meta   = load_company_meta(meta_file)
    print(f"[INFO] Loaded {prices.shape[0]} days × {prices.shape[1]} assets")

    # 3.2 Load CoVaR & MES panels
    cov_mes = load_controls(covar_file, mes_file)
    print(f"[INFO] Loaded CoVaR/MES from {cov_mes.index.min()} to {cov_mes.index.max()}")

    # 3.3 Optional sector aggregation
    if args.level == "sector":
        prices = aggregate_to_sector(prices, meta)
        print(f"[INFO] Aggregated to {prices.shape[1]} sectors")

    # 3.4 Optional asset subsample
    if args.level == "asset" and args.sample:
        prices = prices.sample(n=args.sample, axis=1, random_state=42)
        tickers = set(prices.columns)
        print(f"[INFO] Sub-sampled to {prices.shape[1]} assets")

        # Subset the controls to only these tickers
        # cov_mes has columns like "CoVaR_<TICKER>" and "MES_<TICKER>"
        keep = []
        for col in cov_mes.columns:
            try:
                _, t = col.split("_", 1)
                if t in tickers:
                    keep.append(col)
            except ValueError:
                continue
        cov_mes = cov_mes[keep]
        print(f"[INFO] Subset cov_mes to {len(keep)} control series")

    # 3.5 Compute log returns and prep
    datasets = prep.prepare_all(prices)
    ret_log  = datasets["ret_log"]

    # 3.6 Compute full FRM series + λ-matrix
    nj = args.n_jobs or cfg["n_jobs"]
    print(f"[INFO] Computing FRM (window={cfg['window_days']}, step={cfg['step']}) …")
    frm_out         = compute_frm(
        ret_log,
        window=cfg["window_days"],
        tau=cfg["tau"],
        n_jobs=nj,
        progress=True,
        step=cfg["step"],
        lambda_grid=cfg["lambda_grid"],
        n_folds=cfg["n_folds"],
    )
    frm_idx         = frm_out["frm_index"]
    full_lambda_mat = frm_out["lambda_mat"]  # preserve full series

    # 3.7 Save full λ-matrix & FRM index
    full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
    print(f"[INFO] Saved full lambda matrix {full_lambda_mat.shape}")
    frm_idx.to_csv(outputs / "frm_index_full.csv")
    print(f"[INFO] Saved full FRM index ({len(frm_idx)} rows)")

    # 3.8 Quick SD test pre/post-COVID
    cutoff = "2020-02-15"
    mask   = frm_idx.index < cutoff
    stat, p = sd_stat_pvalue(
        frm_idx[mask], frm_idx[~mask],
        s=2, nboot=cfg["n_boot"], ngrid=cfg["ngrid"], resampling="bootstrap"
    )
    print(f"[RESULT] 2nd-order SD stat={stat:.4f}, p={p:.4f}")

    # 3.9 Bootstrap-history draws
    recent = ret_log.iloc[-cfg["H"] :]
    frm_boot, sd_cent_boot = [], []
    print(f"[INFO] Running {cfg['B']} bootstrap-history draws …")
    for b in range(cfg["B"]):
        ret_b = bootstrap_history(recent, H=cfg["H"],
                                  block_len=cfg["block_len"],
                                  random_state=b)
        out_b = compute_frm(
            ret_b,
            window=cfg["H"],
            tau=cfg["tau"],
            n_jobs=1,
            progress=False,
            step=cfg["H"],
            lambda_grid=cfg["lambda_grid"],
            n_folds=cfg["n_folds"],
        )
        frm_boot.append(out_b["frm_index"].iloc[-1])
        lam_vec = out_b["lambda_mat"].iloc[-1]
        G_b     = dominance_graph_single(
            lam_vec, s=2, alpha=cfg["alpha_sd"],
            ngrid=cfg["ngrid"], nboot=cfg["n_boot"]
        )
        sd_cent_boot.append(compute_graph_centralities(G_b))

    pd.Series(frm_boot, name="frm").to_csv(outputs / "frm_bootstrap.csv")
    pd.DataFrame(sd_cent_boot).to_csv(outputs / "sd_cent_bootstrap.csv")
    print("[INFO] Saved bootstrap-history results")

    # 3.10 Summarize bootstrap distributions
    frm_summary = summarize_frm_bootstrap(outputs / "frm_bootstrap.csv", output_dir=outputs)
    print(f"[SUMMARY] Bootstrapped FRM:\n{frm_summary}\n")
    sd_summary  = summarize_sd_centrality_bootstrap(outputs / "sd_cent_bootstrap.csv", output_dir=outputs)
    print(f"[SUMMARY] Bootstrapped SD centralities:\n{sd_summary}\n")

    # 3.11 Create FRM-jump labels
    y = make_labels(frm_idx, h=cfg["horizon"], q=cfg["jump_quantile"])
    y.to_csv(outputs / "frm_jump_labels.csv", header=["jump"])
    print(f"[INFO] Generated labels: {y.sum()} jumps out of {len(y)} days")

    # 3.12 Build flattened SD-centrality panel over time
    dates       = full_lambda_mat.index
    sd_cent_list = []
    for t in dates:
        G_t  = dominance_graph_single(
            full_lambda_mat.loc[t],
            s=2, alpha=cfg["alpha_sd"],
            ngrid=cfg["ngrid"], nboot=cfg["n_boot"]
        )
        cent = compute_graph_centralities(G_t)
        flat = {
            f"{met}_{node}": val
            for node, mets in cent.items()
            for met, val in mets.items()
        }
        sd_cent_list.append(flat)
    sd_cent_df = pd.DataFrame(sd_cent_list, index=dates)

    # 3.13 Assemble feature panel
    X = make_feature_panel(frm_idx, full_lambda_mat, sd_cent_df, cov_mes, lags=cfg["horizon"])
    y = y.loc[X.index]
    print(f"[INFO] Feature panel shape: {X.shape}")
    print(f"[INFO] Labels shape:        {y.shape}")
    if X.shape[0] < 1:
        raise ValueError("No samples after alignment: adjust 'horizon' or lag settings.")

    # 3.14 Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # 3.15 Impute any remaining NaNs (mean imputation)
    from sklearn.impute import SimpleImputer

    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    # 3.16 Logistic-LASSO baseline
    log_mod = train_logistic_lasso(X_train, y_train)
    auc_log, pr_df = eval_logistic(log_mod, X_test, y_test)
    print(f"[LOGIT] AUC-ROC = {auc_log:.3f}")
    pr_df.to_csv(outputs / "pr_logistic.csv", index=False)

    # 3.16 VAR baseline: align and fill controls, then fit without dropna
    cov_mes_ff = cov_mes.reindex(frm_idx.index).ffill().bfill()
    print(f"[DEBUG] cov_mes_ff has shape {cov_mes_ff.shape} and nulls: {cov_mes_ff.isna().sum().sum()}")

    df_var = pd.concat([
        frm_idx.rename("FRM"),
        cov_mes_ff
    ], axis=1)
    print(f"[DEBUG] df_var shape before dropna: {df_var.shape}, nulls: {df_var.isna().sum().sum()}")

    # Check we have enough rows
    min_rows = cfg["var_lags"] + cfg["horizon"]
    if df_var.shape[0] < min_rows:
        raise ValueError(
            f"Not enough data ({df_var.shape[0]} rows) for VAR "
            f"with lags={cfg['var_lags']} and horizon={cfg['horizon']}"
        )

    # Fit VAR on the full daily panel
    var_res = train_var(df_var["FRM"], df_var.drop(columns="FRM"), lags=cfg["var_lags"])
    var_forecast = forecast_var(var_res, steps=cfg["horizon"])
    var_prob = var_forecast.reindex(y_test.index).ffill()

    # Align var_prob and y_test, then drop any NaNs
    aligned = pd.concat([y_test.rename("y"), var_prob.rename("score")], axis=1)
    aligned = aligned.dropna()

    if aligned.empty:
        print("[WARN] No overlapping dates for VAR forecast vs. test labels. Skipping VAR AUC.")
    else:
        y_aligned = aligned["y"]
        var_prob_aligned = aligned["score"]
        auc_var = roc_auc_score(y_aligned, var_prob_aligned)
        print(f"[VAR] AUC-ROC = {auc_var:.3f}")

    print(f"[DONE] {datetime.now():%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    main()
