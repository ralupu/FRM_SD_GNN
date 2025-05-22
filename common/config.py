"""
common/config.py
================
Small utility used by *main.py* (and any future CLI scripts).

Functions
---------
parse_args()  -> argparse.Namespace
    • Recognises --sample  (int)   max # of assets
    • Recognises --config  (str)   YAML file path

load_cfg(path) -> dict
    • Reads YAML into dict.
    • If file doesn’t exist, returns built-in defaults.

Defaults
--------
window_days : 63
tau         : 0.05
block_len   : 10
n_boot      : 300
alpha_sd    : 0.05
"""

from pathlib import Path
import argparse, yaml


# ╔═══════════════════════════════════════════════════════╗
# 1. CLI parser
# ╚═══════════════════════════════════════════════════════╝
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Systemic-risk FRM/SD driver")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly keep only N assets for a quick run")
    p.add_argument("--config", type=str, default="config.yml",
                   help="YAML file with hyper-parameters")
    return p.parse_args()


# ╔═══════════════════════════════════════════════════════╗
# 2. YAML loader with sane fall-back
# ╚═══════════════════════════════════════════════════════╝
_DEFAULTS = dict(
    window_days=63,
    tau=0.05,
    block_len=10,
    n_boot=300,
    alpha_sd=0.05,
)


def load_cfg(path: str | Path) -> dict:
    cfg_file = Path(path)
    if cfg_file.exists():
        try:
            return yaml.safe_load(cfg_file.read_text()) or _DEFAULTS.copy()
        except yaml.YAMLError as e:
            print(f"[WARN] YAML parse error: {e}. Using defaults.")
    return _DEFAULTS.copy()
