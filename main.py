#!/usr/bin/env python3
"""
main.py
--------
End–to–end driver for the STOXX-600 systemic-risk project.

Features added vs. original:
  • argparse  —  --sample N  to limit #assets
  • YAML cfg  —  --config path/to/cfg.yml
  • tqdm bars —  progress on FRM & bootstrap
  • hooks to   prep.prepare_all  +  frm_asgl.compute_frm + sd_utils
"""

# ╔════════════════════════════════════════════════════════╗
# 0. Imports & global config
# ╚════════════════════════════════════════════════════════╝
from pathlib import Path
import argparse, yaml, numpy as np, pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

# local modules
from analysis import prep, frm_asgl as frm, bootstrap, sd_utils as sd

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ╔════════════════════════════════════════════════════════╗
# 1. Helpers: CLI + YAML
# ╚════════════════════════════════════════════════════════╝
def parse_args():
    p = argparse.ArgumentParser(description="Run FRM/SD pipeline")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly keep only N assets for a fast run")
    p.add_argument("--config", type=str, default="config.yml",
                   help="YAML file with hyper-parameters")
    return p.parse_args()


def load_cfg(path) -> dict:
    cfg_file = Path(path)
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text())
    # defaults
    return dict(window_days=63, tau=0.05,
                block_len=10, n_boot=300, alpha_sd=0.05)


# ╔════════════════════════════════════════════════════════╗
# 2. Raw Excel loaders (unchanged)
# ╚════════════════════════════════════════════════════════╝
def load_closing_prices(path: Path) -> pd.DataFrame:
    df = (pd.read_excel(path, engine="openpyxl")
            .rename(columns=lambda c: c.strip())
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
            .set_index("Date")
            .sort_index())
    return df


def load_company_meta(path: Path) -> pd.DataFrame:
    meta = (pd.read_excel(path, engine="openpyxl")
              .rename(columns=lambda c: c.strip())
              .drop_duplicates(subset="Ticker")
              .set_index("Ticker"))
    return meta


def initial_clean(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.apply(pd.to_numeric, errors="coerce")
                  .dropna(axis=1, how="all")
                  .ffill(limit=3))


# ╔════════════════════════════════════════════════════════╗
# 3. Main workflow
# ╚════════════════════════════════════════════════════════╝
def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    price_file = DATA_DIR / "ClosingPrices.xlsx"
    meta_file  = DATA_DIR / "names_sectors_28Mar25.xlsx"

    # 3.1 Load & clean
    prices = initial_clean(load_closing_prices(price_file))
    meta   = load_company_meta(meta_file)

    # 3.2 Optional asset sampling
    if args.sample and args.sample < prices.shape[1]:
        keep = np.random.default_rng(42).choice(
            prices.columns, size=args.sample, replace=False)
        prices = prices[keep]
        print(f"[INFO] Sub-sampled to {args.sample} assets.")

    print(f"[INFO] {prices.shape[0]:,} days × {prices.shape[1]:,} assets ready.")

    # 3.3 Prepare returns etc.
    datasets  = prep.prepare_all(prices)
    ret_log   = datasets["ret_log"]

    # 3.4 Compute FRM with progress bar
    print("[INFO] Running FRM …")
    frm_out = frm.compute_frm(
        ret_log,
        window=cfg["window_days"],
        tau=cfg["tau"],
        n_jobs=-1,
        progress=True,      # tqdm inside frm_asgl
    )
    frm_idx    = frm_out["frm_index"]
    lambda_mat = frm_out["lambda_mat"]
    frm_idx.to_csv(OUTPUT_DIR / "frm_index.csv")

    # 3.5 Simple SD test pre- vs post-COVID as example
    covid_cut = frm_idx.index < "2020-02-15"
    stat, p   = sd.sd_stat_pvalue(
        frm_idx[covid_cut], frm_idx[~covid_cut],
        s=2, nboot=cfg["n_boot"])
    print(f"[RESULT] 2nd-order SD stat = {stat:.4f},  p = {p:.4f}")

    # 3.6 Moving-block bootstrap loop (optional; uncomment to run)
    """
    print("[INFO] Bootstrapping …")
    boot_stats = []
    for ret_b in bootstrap.moving_block_bootstrap(
            ret_log, block_len=cfg["block_len"],
            n_boot=cfg["n_boot"], progress=True):
        frm_b  = frm.compute_frm(ret_b, window=cfg["window_days"],
                                 tau=cfg["tau"], n_jobs=-1,
                                 progress=False)["frm_index"]
        s, _   = sd.sd_stat_pvalue(frm_b[covid_cut], frm_b[~covid_cut],
                                   s=2, nboot=0)
        boot_stats.append(s)
    np.save(OUTPUT_DIR / "sd_boot_stats.npy", np.array(boot_stats))
    """

    print(f"[DONE] {datetime.now():%Y-%m-%d %H:%M}")



# ╔════════════════════════════════════════════════════════╗
# 4. Entry guard
# ╚════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    main()
