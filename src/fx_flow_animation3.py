import os
import requests
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Layout: place all currencies evenly on a circle
# -----------------------------
def circular_layout(currencies, radius=1.0, center=(0, 0)):
    cx, cy = center
    n = len(currencies)
    pos = {}
    for i, c in enumerate(currencies):
        theta = 2 * np.pi * i / n
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        pos[c] = (x, y)
    return pos

# -----------------------------
# Fetch FX rates (base = USD) using exchangerate.host
# -----------------------------
def fetch_rates(base="USD", currencies=None, days=2):
    """
    Returns a DataFrame of FX rates vs base for the last `days` days.
    Columns = CURRENCIES, rows = dates (descending).
    """
    if currencies is None:
        currencies = CURRENCIES

    rates_list = []
    for delta in range(days):
        d = datetime.utcnow().date() - timedelta(days=delta)
        url = f"https://api.exchangerate.host/{d.isoformat()}"
        params = {
            "base": base,
            "symbols": ",".join(currencies),
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        if not data.get("success", True):  # some days may fail
            continue
        row = {c: data["rates"].get(c, np.nan) for c in currencies}
        row[base] = 1.0
        row["date"] = d
        rates_list.append(row)

    df = pd.DataFrame(rates_list)
    df = df.set_index("date").sort_index()
    # Ensure all currencies present
    for c in currencies:
        if c not in df.columns:
            df[c] = np.nan
    df = df[currencies]
    return df

# -----------------------------
# Build currency graph (edges)
# -----------------------------
def build_country_graph():
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    edges = [
        ("USD","EUR"), ("USD","JPY"), ("USD","KRW"),
        ("USD","GBP"), ("USD","SGD"), ("USD","HKD"),
        ("USD","AUD"), ("EUR","GBP"), ("EUR","JPY"),
        ("JPY","KRW"), ("SGD","HKD"), ("SGD","AUD"),
        ("KRW","HKD")
    ]
    G.add_edges_from(edges)
    return G

# -----------------------------
# Draw snapshot (nodes + arrows)
# -----------------------------
def draw_snapshot(G, rates, pos, filename, title="FX Flow Network"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    if len(rates) < 2:
        raise RuntimeError("Not enough data to compare two dates.")

    today = rates.iloc[-1]
    yesterday = rates.iloc[-2]

    dK_edges = {}
    node_change = {c: 0.0 for c in G.nodes()}

    # Compute dK for edges
    for u, v in G.edges():
        if pd.isna(today.get(u)) or pd.isna(today.get(v)):
            continue
        K_today = today[u] / today[v]
        K_prev = yesterday[u] / yesterday[v]
        dK = K_today - K_prev
        if dK == 0:
            continue
        start, end = (u, v) if dK < 0 else (v, u)
        dK_edges[(start, end)] = abs(dK)
        # accumulate net change to nodes
        node_change[u] += dK
        node_change[v] += dK

    # Compute node sizes
    max_abs = max(abs(val) for val in node_change.values()) + 1e-8
    node_sizes = [300 + 2000 * (abs(node_change[n]) / max_abs) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for (start, end), w in dK_edges.items():
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='gray',
                                linewidth=2 + 15*w,
                                mutation_scale=10 + 20*w,
                                connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    G = build_country_graph()
    pos = circular_layout(CURRENCIES, radius=1.5, center=(0, 0))

    rates = fetch_rates(base="USD", currencies=CURRENCIES, days=3)
    print("Rates:\n", rates.tail())

    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{now.strftime('%Y%m%d_%H%M%S')}.png")
    draw_snapshot(G, rates, pos, fname, title=f"FX Flow Network ({timestamp})")
    print("Saved:", fname)
