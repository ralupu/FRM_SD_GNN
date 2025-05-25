#!/usr/bin/env python3
"""
run_gnn.py
----------
Driver for dynamic-GNN early-warning model on SD networks.

This script loads pre-saved FRM index and lambda-matrix, reconstructs daily
SD graphs, builds a PyG dataset, trains a temporal GNN, and evaluates it.
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
from torch_geometric.data import DataLoader

from analysis.sd_network import dominance_graph_single
from analysis.covar_mes import load_controls
from analysis.evaluate import make_labels
from analysis.gnn_model import (
    FRMTGNN,
    build_dataset,
    train_tgat,
    evaluate_tgat
)


def parse_args():
    p = argparse.ArgumentParser("GNN driver")
    p.add_argument("--config", type=Path, default=Path("config.yml"),
                   help="path to YAML config")
    p.add_argument("--sample", type=int, default=None,
                   help="sub-sample N time-steps for quick runs")
    p.add_argument("--epochs", type=int, default=20,
                   help="training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate")
    return p.parse_args()


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs = root / "outputs"

    # 1. Load pre-saved data
    frm_idx = pd.read_csv(outputs / "frm_index_full.csv",
                          index_col=0, parse_dates=True)["frm_index"]
    lambda_mat = pd.read_csv(outputs / "lambda_mat_full.csv",
                             index_col=0, parse_dates=True)
    covar = pd.read_csv(data_dir / "covar.csv",
                        index_col=0, parse_dates=True)["CoVaR"]
    mes = pd.read_csv(data_dir / "mes.csv",
                      index_col=0, parse_dates=True)["MES"]
    cov_mes = pd.concat([covar, mes], axis=1).ffill()

    # 2. Create labels
    y = make_labels(
        frm_idx,
        h=cfg.get("horizon", 5),
        q=cfg.get("jump_quantile", 0.90)
    )

    # 3. Build PyG dataset of (graph, label) per day
    graphs, labels = build_dataset(
        lambda_mat=lambda_mat,
        cov_mes=cov_mes,
        cfg=cfg,
        make_labels_fn=make_labels
    )

    # 4. Optional sub-sample of time-steps
    if args.sample is not None:
        graphs = graphs[:args.sample]
        labels = labels[:args.sample]

    # 5. Split train/test (80/20)
    N = len(graphs)
    split = int(0.8 * N)
    train_graphs, test_graphs = graphs[:split], graphs[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # 6. DataLoaders
    train_loader = DataLoader(list(zip(train_graphs, train_labels)),
                              batch_size=1, shuffle=True)
    test_loader = DataLoader(list(zip(test_graphs, test_labels)),
                             batch_size=1)

    # 7. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRMTGNN(
        node_feature_dim=cfg.get("node_feat_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 32),
        num_layers=cfg.get("num_layers", 2)
    ).to(device)

    # 8. Train
    train_tgat(model, train_loader,
               epochs=args.epochs, lr=args.lr, device=device)

    # 9. Evaluate
    auc = evaluate_tgat(model, test_loader, device=device)
    print(f"[GNN] Test AUC-ROC = {auc:.3f}")


if __name__ == "__main__":
    main()
