#!/usr/bin/env python3
"""
run_crypto.py
-------------
Full pipeline runner for FRM/SD/Factor analysis on cryptos (monthly/weekly).
Avoids recomputation: saves/loads lambda matrices and SD networks for fast reruns!
"""

from pathlib import Path
import yaml
import pickle
import numpy as np
import pandas as pd
import argparse
import os

def main():
    # --- CLI and config ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_network', action='store_true',
                        help="Recompute FRM/SD network (default: False, just reload pickles if present)")
    parser.add_argument('--config', type=str, default="config.yml", help="Path to config file")
    args = parser.parse_args()
    with_network = args.with_network

    # --- Load config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    # --- Data path setup ---
    networks_path = outputs / "networks.pkl"
    lambda_path = outputs / "lambda_mat_full.pkl"
    frm_idx_path = outputs / "frm_index_full.pkl"

    # --- 1. Data Prep (skip if already prepped) ---
    returns = pd.read_csv(cfg['returns_path'], index_col=0, parse_dates=True)
    ret_log = returns.copy()
    print("Data periods:", len(ret_log), "| Rolling window:", cfg["window"])
    print("ret_log shape:", ret_log.shape, ", window:", cfg["window"])

    # --- 2. Compute or load FRM/λ-matrix ---
    if with_network or not lambda_path.exists() or not frm_idx_path.exists():
        print("[INFO] Computing FRM/λ-matrix ...")
        from analysis.frm_asgl import compute_frm
        frm_out = compute_frm(
            ret_log,
            window=cfg["window"],
            tau=cfg["tau"],
            n_jobs=cfg.get("n_jobs", 4),
            progress=True,
            step=cfg["step"],
            lambda_grid=cfg["lambda_grid"],
            n_folds=cfg.get("n_folds", 2),
            bootstrap=cfg.get("bootstrap", 0)
        )
        frm_idx = frm_out["frm_index"]
        full_lambda_mat = frm_out["lambda_mat"]
        # Save outputs as pickle for fast reload
        full_lambda_mat.to_pickle(lambda_path)
        frm_idx.to_pickle(frm_idx_path)
        # Also save as CSV for reference
        full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
        frm_idx.to_csv(outputs / "frm_index_full.csv")
    else:
        print("[INFO] Loading FRM/λ-matrix from file ...")
        full_lambda_mat = pd.read_pickle(lambda_path)
        frm_idx = pd.read_pickle(frm_idx_path)
        print(f"[INFO] Loaded lambda_mat ({full_lambda_mat.shape}), frm_idx ({frm_idx.shape}) from pickles.")

    # --- 3. Compute or load networks ---
    if with_network or not networks_path.exists():
        print("[INFO] Computing and saving SD networks...")
        from analysis.sd_network import dominance_graph_single
        network_list = []
        for date, row in full_lambda_mat.iterrows():
            G = dominance_graph_single(row)
            network_list.append((date, G))
        with open(networks_path, "wb") as f:
            pickle.dump(network_list, f)
        print(f"[INFO] Saved networkx graphs for all periods to {networks_path}")
    else:
        print("[INFO] Loading precomputed networks from file...")
        with open(networks_path, "rb") as f:
            network_list = pickle.load(f)
        print(f"[INFO] Loaded {len(network_list)} networks from {networks_path}")

    # --- 4. Centralities ---
    from analysis.features import compute_graph_centralities
    centralities = []
    for date, G in network_list:
        cent = compute_graph_centralities(G)
        # flatten: {'metric': {asset: value}} to {'asset_metric': value}
        flat = {f"{asset}_{metric}": val for metric, asset_dict in cent.items() for asset, val in asset_dict.items()}
        flat['Date'] = date
        centralities.append(flat)
    centrality_df = pd.DataFrame(centralities).set_index("Date")
    centrality_df.to_csv(outputs / "centralities.csv")

    # --- 5. Factor construction ---
    from analysis.factor import build_HL_factor
    factor_series = build_HL_factor(
        centrality_df=centrality_df,
        returns_df=returns,
        metric=cfg["centrality_metric"],
        top_n=cfg["factor_high"],
        bottom_n=cfg["factor_low"],
        out_path=cfg["factor_output"]
    )

    # --- 6. Performance/Econ test ---
    from analysis.econ_test import run_econ_test
    run_econ_test(
        factor_path=cfg["factor_output"],
        returns_path=cfg["returns_path"],
        out_dir="outputs"
    )

    print("[PIPELINE DONE] All outputs saved in 'outputs/'.")

if __name__ == "__main__":
    main()
