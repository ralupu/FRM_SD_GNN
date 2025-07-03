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

from analysis import prep
from analysis.frm_asgl      import compute_frm
from analysis.sd_network    import dominance_graph_single
from analysis.features      import compute_graph_centralities
from analysis.factor        import build_HL_factor

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

    # 1. Load log-returns (monthly or weekly)
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    print(f"[INFO] Loaded {returns.shape[0]} periods × {returns.shape[1]} assets from {returns_file}")

    # 2. (Optional) Asset subsample
    if args.sample:
        returns = returns.sample(n=args.sample, axis=1, random_state=42)
        print(f"[INFO] Sub-sampled to {returns.shape[1]} assets")

    # 3. Prepare returns for pipeline
    datasets = prep.prepare_all(returns)
    ret_log  = datasets["ret_log"]

    # 4. Compute FRM/λ-matrix
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

    # 5. Save λ-matrix & FRM index
    full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
    frm_idx.to_csv(outputs / "frm_index_full.csv")
    print(f"[INFO] Saved lambda matrix {full_lambda_mat.shape} and FRM index ({len(frm_idx)} rows)")

    # 6. Build scalar SD networks and extract centralities for each period
    print(f"[INFO] Building scalar SD networks and extracting centralities …")
    centralities = []
    for date, row in full_lambda_mat.iterrows():
        G = dominance_graph_single(row)  # Scalar network
        cent = compute_graph_centralities(G)
        # Flatten: {'metric': {asset: value}} to {'metric_asset': value}
        flat = {f"{metric}_{asset}": val for metric, asset_dict in cent.items() for asset, val in asset_dict.items()}
        flat['Date'] = date
        centralities.append(flat)
    centrality_df = pd.DataFrame(centralities).set_index('Date')
    centrality_df.to_csv(outputs / "centralities.csv")
    print(f"[INFO] Saved SD network centralities ({centrality_df.shape}) to outputs/centralities.csv")

    # 7. Build Network Risk factor (High-Low) and save
    print(f"[INFO] Building Network Risk factor (High minus Low portfolios) …")
    from analysis.factor import build_HL_factor
    factor_series = build_HL_factor(
        centrality_df=centrality_df,
        returns_df=returns,
        metric=cfg["centrality_metric"],
        top_n=cfg["factor_high"],
        bottom_n=cfg["factor_low"],
        out_path=cfg["factor_output"]
    )
    print(f"[INFO] Done. Network risk factor shape: {factor_series.shape}")

    print(f"[DONE] {datetime.now():%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    main()
