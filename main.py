#!/usr/bin/env python3
"""
main.py
--------
End-to-end driver for crypto/FRM/SD pipeline with monthly or weekly data, config-driven!
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
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            print(f"[WARN] could not parse {path}; using defaults")
            cfg = {}
    else:
        cfg = {}
    # Provide minimal defaults, but all main params should be in config.yml now
    return cfg

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    root     = Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs  = root / "outputs"
    outputs.mkdir(exist_ok=True)

    # Use config-driven file paths for returns/volumes
    returns_file = Path(cfg['returns_path'])
    volumes_file = Path(cfg.get('volumes_path', ""))

    # 1. Load log-returns (monthly or weekly)
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    print(f"[INFO] Loaded {returns.shape[0]} periods × {returns.shape[1]} assets from {returns_file}")

    # 2. (Optional) Load CoVaR/MES if using (skip if only crypto returns for now)
    cov_mes = None
    if 'covar_path' in cfg and 'mes_path' in cfg:
        cov_mes = load_controls(cfg['covar_path'], cfg['mes_path'])
        print(f"[INFO] Loaded CoVaR/MES from {cov_mes.index.min()} to {cov_mes.index.max()}")

    # 3. (Optional) Sector aggregation (skip for crypto, but kept for full generality)
    #if args.level == "sector":
    #    returns = aggregate_to_sector(returns, meta)
    #    print(f"[INFO] Aggregated to {returns.shape[1]} sectors")

    # 4. (Optional) Asset subsample
    if args.sample:
        returns = returns.sample(n=args.sample, axis=1, random_state=42)
        print(f"[INFO] Sub-sampled to {returns.shape[1]} assets")

    # 5. Prepare (if you use extra prep: cleaning, imputation, etc.)
    datasets = prep.prepare_all(returns)
    ret_log  = datasets["ret_log"]

    # 6. Compute FRM/λ-matrix
    nj = args.n_jobs or cfg.get("n_jobs", 4)
    window = cfg.get("window", 12)
    print(f"[INFO] Computing FRM (window={window}, step={cfg['step']}, freq={cfg['frequency']}) …")
    frm_out = compute_frm(
        ret_log,
        window=window,
        tau=cfg['tau'],
        n_jobs=nj,
        progress=True,
        step=cfg['step'],
        lambda_grid=cfg['lambda_grid'],
        n_folds=cfg.get('n_folds', 2),
    )
    frm_idx         = frm_out["frm_index"]
    full_lambda_mat = frm_out["lambda_mat"]

    # 7. Save λ-matrix & FRM index
    full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
    frm_idx.to_csv(outputs / "frm_index_full.csv")
    print(f"[INFO] Saved lambda matrix {full_lambda_mat.shape} and FRM index ({len(frm_idx)} rows)")

    # (The rest: SD network, factor, etc. — as per your usual pipeline)
    # TODO: Insert additional steps for SD network, factor construction, regression as needed.
    print(f"[DONE] {datetime.now():%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    main()
