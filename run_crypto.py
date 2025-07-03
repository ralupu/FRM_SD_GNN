#!/usr/bin/env python3
"""
run_crypto.py
-------------
Full pipeline runner for FRM/SD/Factor analysis on cryptos (monthly/weekly).
"""

from pathlib import Path
import yaml

def main():
    # --- Load config ---
    config_path = Path("config.yml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 1. Data Prep (skip if already prepped; optional) ---
    # If you want to rerun crypto_prep, uncomment:
    # from analysis.crypto_prep import get_returns_and_volumes
    # get_returns_and_volumes(
    #     top_n=cfg.get("top_n", 15),
    #     folder="data/cryptos/",
    #     start_date=cfg.get("start_date", "2019-01-01"),
    #     freq=cfg.get("frequency", "M")[0].upper(),
    # )

    # --- 2. Load data ---
    import pandas as pd
    returns = pd.read_csv(cfg['returns_path'], index_col=0, parse_dates=True)

    ret_log = returns.copy()

    print(f"Data periods: {len(ret_log)} | Rolling window: {cfg['window']}")
    print(f"ret_log shape: {ret_log.shape}, window: {cfg['window']}")

    # --- 4. Compute FRM/λ-matrix ---
    from analysis.frm_asgl import compute_frm
    frm_out = compute_frm(
        ret_log,
        window=cfg["window"],
        tau=cfg["tau"],
        n_jobs=cfg.get("n_jobs"),
        progress=True,
        step=cfg["step"],
        lambda_grid=cfg["lambda_grid"],
        n_folds=cfg.get("n_folds", 2),
        bootstrap=cfg.get("bootstrap", 0)
    )
    frm_idx = frm_out["frm_index"]
    full_lambda_mat = frm_out["lambda_mat"]
    print("Type/sample of one lambda entry:", type(full_lambda_mat.iloc[0, 0]),
          "size:", np.atleast_1d(full_lambda_mat.iloc[0, 0]).size)

    if full_lambda_mat.empty:
        raise ValueError(
            "FRM lambda matrix is empty! This usually means your sample has fewer rows than the rolling window size. "
            f"Returned shape: {full_lambda_mat.shape} | Data length: {len(ret_log)} | Window: {cfg['window']}")

    # Save outputs
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
    frm_idx.to_csv(outputs / "frm_index_full.csv")

    # --- 5. SD network & centralities ---
    from analysis.sd_network import dominance_graph_single
    from analysis.features import compute_graph_centralities
    centralities = []
    for date, row in full_lambda_mat.iterrows():
        G = dominance_graph_single(row)
        cent = compute_graph_centralities(G)
        flat = {f"{metric}_{asset}": val for metric, asset_dict in cent.items() for asset, val in asset_dict.items()}
        flat["Date"] = date
        centralities.append(flat)
    centrality_df = pd.DataFrame(centralities).set_index("Date")
    centrality_df.to_csv(outputs / "centralities.csv")

    # --- 6. Factor construction ---
    from analysis.factor import build_HL_factor
    factor_series = build_HL_factor(
        centrality_df=centrality_df,
        returns_df=returns,
        metric=cfg["centrality_metric"],
        top_n=cfg["factor_high"],
        bottom_n=cfg["factor_low"],
        out_path=cfg["factor_output"]
    )

    # --- 7. Performance/Econ test ---
    from analysis.econ_test import run_econ_test
    run_econ_test(
        factor_path=cfg["factor_output"],
        returns_path=cfg["returns_path"],
        out_dir="outputs"
    )

    print("[PIPELINE DONE] All outputs saved in 'outputs/'.")

if __name__ == "__main__":
    main()


# Use as:
# python run_crypto.py
