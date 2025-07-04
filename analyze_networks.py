import pickle
from pathlib import Path

NETWORKS_PATH = Path("outputs/networks.pkl")

# ---- Load precomputed networks ----
with open(NETWORKS_PATH, "rb") as f:
    network_list = pickle.load(f)  # [(date, G), ...]

print(f"Loaded {len(network_list)} networks.")

# ---- Print number of edges per network ----
print("\nEdge count by period:")
for date, G in network_list:
    print(f"{date}: {len(G.edges())} edges")

# ---- Print sample edges for a few periods ----
print("\nSample edges (with weights/p-values) for first 3 periods:")
for date, G in network_list[:3]:
    print(f"\n{date}:")
    for u, v, d in G.edges(data=True):
        print(f"  {u} → {v} | weight: {d.get('weight')}, pvalue: {d.get('pvalue')}")

# ---- Check how often the network changes ----
previous_edges = None
changes = 0
for date, G in network_list:
    current_edges = set(G.edges())
    if previous_edges is not None and current_edges != previous_edges:
        changes += 1
    previous_edges = current_edges
print(f"\nNumber of periods with a changed network structure: {changes} / {len(network_list)-1}")

# ---- Jaccard similarity between networks ----
from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

jaccard_scores = []
for (date1, G1), (date2, G2) in pairwise(network_list):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    if union > 0:
        score = intersection / union
    else:
        score = 1.0
    jaccard_scores.append(score)

if jaccard_scores:
    print(f"Mean Jaccard similarity between successive networks: {sum(jaccard_scores)/len(jaccard_scores):.3f}")
else:
    print("Not enough networks to compute Jaccard similarity.")

# ---- End ----
print("\nAnalysis complete.")
