"""
common/config.py
================
Centralised CLI and YAML configuration loader.

Functions
---------
parse_args()  -> argparse.Namespace
    • Recognises:
        --sample (int)    : max # of assets for a smoke test
        --n_jobs (int)    : override parallel workers for FRM
        --level {asset,sector} : aggregation level
        --config (str)    : path to YAML config

load_cfg(path) -> dict
    • Reads YAML into dict and merges with sensible defaults.
    • If file missing or malformed, uses built-in defaults.
"""

from pathlib import Path
import argparse, yaml
from typing import Union

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLI parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FRM/SD pipeline driver")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly keep only N assets (asset-level only)")
    p.add_argument("--n_jobs", type=int, default=None,
                   help="override number of parallel jobs for FRM")
    p.add_argument("--level", choices=["asset", "sector"], default="asset",
                   help="run at 'asset' or 'sector' aggregation level")
    p.add_argument("--config", type=str, default="config.yml",
                   help="YAML file with hyper-parameters")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Defaults + YAML loader
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    window_days=63,
    tau=0.05,
    step=1,
    lambda_grid=[0.001, 0.01, 0.1, 1.0, 10.0],
    n_folds=1,
    n_jobs=4,
    block_len=10,
    n_boot=100,
    alpha_sd=0.05,
    level="asset",
    sample=None,
)

def load_cfg(path: Union[str, Path]) -> dict:
    cfg = _DEFAULTS.copy()
    p = Path(path)
    if p.exists():
        try:
            # force UTF-8 decoding to avoid Windows ANSI issues
            text = p.read_text(encoding="utf-8")
            loaded = yaml.safe_load(text) or {}
            cfg.update(loaded)
        except yaml.YAMLError as e:
            print(f"[WARN] Failed parsing {path}: {e}; using defaults")
    return cfg
