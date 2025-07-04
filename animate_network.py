import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ---- Config ----
NETWORKS_PATH = Path("outputs/networks.pkl")          # Where your network pickle is
OUTPUT_GIF = Path("outputs/sd_network_evolution.gif") # Output animation file
FRAME_INTERVAL = 700   # ms per frame

# ---- Load precomputed networks ----
with open(NETWORKS_PATH, "rb") as f:
    network_list = pickle.load(f)  # [(date, G), ...]

# ---- Compute consistent node layout ----
# Get union of all nodes across all periods
all_nodes = set()
for _, G in network_list:
    all_nodes.update(G.nodes)
all_nodes = sorted(all_nodes)
G_union = nx.DiGraph()
G_union.add_nodes_from(all_nodes)
for _, G in network_list:
    G_union.add_edges_from(G.edges())
# Fixed layout (spring layout can be replaced by circular for reproducibility)
pos = nx.spring_layout(G_union, seed=42)

# ---- Create animation ----
fig, ax = plt.subplots(figsize=(8, 8))
plt.close()  # Don't show static plot if running in notebook

def update(frame):
    ax.clear()
    date, G = network_list[frame]
    # Node/edge properties (customize as you wish!)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=800, edgecolors='k', linewidths=1)
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=15, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13)
    ax.set_title(f"SD Network - {date.strftime('%Y-%m-%d')}", fontsize=18)
    ax.axis('off')
    # Optional: show frame number for debugging
    # ax.text(0.05, 0.95, f"Frame: {frame+1}/{len(network_list)}", transform=ax.transAxes, fontsize=12, ha='left', va='top')

ani = animation.FuncAnimation(
    fig, update, frames=len(network_list), interval=FRAME_INTERVAL, repeat=True
)

# ---- Save as GIF ----
print(f"Saving SD network animation to {OUTPUT_GIF} ...")
ani.save(str(OUTPUT_GIF), writer="pillow")
print("Done! Animation saved.")

# ---- Optional: To view in Jupyter notebook (uncomment if using Jupyter) ----
# from IPython.display import HTML
# HTML(ani.to_jshtml())
