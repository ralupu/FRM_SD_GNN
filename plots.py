import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ---- 1. FRM Index over time ----
frm_idx = pd.read_csv("outputs/frm_index_full.csv", index_col=0, parse_dates=True).squeeze("columns")
plt.figure(figsize=(10,4))
plt.plot(frm_idx)
plt.title("Evolution of FRM Index (Aggregate Lambda)")
plt.xlabel("Time")
plt.ylabel("FRM Index")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plot_frm_index.png")
plt.close()

# ---- 2. NetworkRisk factor over time ----
networkrisk = pd.read_csv("outputs/NetworkRisk.csv", index_col=0, parse_dates=True).squeeze("columns")
plt.figure(figsize=(10,4))
plt.plot(networkrisk)
plt.title("NetworkRisk Factor (High-Low Spread) Over Time")
plt.xlabel("Time")
plt.ylabel("NetworkRisk Factor")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plot_networkrisk_factor.png")
plt.close()

# ---- 3. Cumulative returns for High/Low centrality portfolios ----
centralities = pd.read_csv("outputs/centralities.csv", index_col=0, parse_dates=True)
eig_cols = [c for c in centralities.columns if c.endswith("_eigenvector")]
eig_matrix = centralities[eig_cols]
eig_matrix.columns = [c.replace("_eigenvector", "") for c in eig_cols]

mean_eig = eig_matrix.mean(axis=0)
high_assets = mean_eig.sort_values(ascending=False).head(3).index.tolist()
low_assets = mean_eig.sort_values(ascending=True).head(3).index.tolist()

returns = pd.read_csv("data/monthly_log_returns.csv", index_col=0, parse_dates=True)
cum_high = (1 + returns[high_assets].mean(axis=1)).cumprod()
cum_low = (1 + returns[low_assets].mean(axis=1)).cumprod()
cum_hl = cum_high / cum_low

plt.figure(figsize=(10,5))
plt.plot(cum_high, label="High Centrality Portfolio")
plt.plot(cum_low, label="Low Centrality Portfolio")
plt.plot(cum_hl, label="High/Low Spread")
plt.title("Cumulative Return of Network Portfolios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plot_cum_portfolios.png")
plt.close()

# ---- 4. Number of edges in the network over time ----
with open("outputs/networks.pkl", "rb") as f:
    network_list = pickle.load(f)
edge_counts = [len(G.edges()) for _, G in network_list]
dates = [date for date, _ in network_list]

plt.figure(figsize=(10,4))
plt.plot(dates, edge_counts, marker='o')
plt.title("Number of Edges in SD Network Over Time")
plt.xlabel("Time")
plt.ylabel("Edge Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plot_edge_count.png")
plt.close()

# ---- 5. Heatmap of centralities (Eigenvector) ----
if eig_matrix.shape[0] > 0 and eig_matrix.shape[1] > 0:
    plt.figure(figsize=(12,6))
    sns.heatmap(eig_matrix.T, cmap="YlOrRd", cbar_kws={"label": "Eigenvector Centrality"})
    plt.title("Heatmap of Eigenvector Centrality Over Time")
    plt.xlabel("Time")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig("outputs/plot_centrality_heatmap.png")
    plt.close()
else:
    print("Eigenvector centrality matrix is empty! Heatmap not plotted.")


print("All plots saved in outputs/ folder.")
