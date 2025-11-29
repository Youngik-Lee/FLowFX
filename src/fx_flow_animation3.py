import os
import requests
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FETCH FX DATA
# -----------------------------
def fetch_rates(base="USD", currencies=CURRENCIES, days=2):
    rates_list = []
    for delta in range(days):
        d = datetime.utcnow().date() - timedelta(days=delta)
        url = f"https://api.exchangerate.host/{d.isoformat()}"
        params = {"base": base, "symbols": ",".join(currencies)}
        resp = requests.get(url, params=params)
        data = resp.json()
        if not data.get("success", True):
            continue
        rates = data.get("rates", {})
        if not rates:
            continue
        row = {c: rates.get(c, np.nan) for c in currencies}
        row[base] = 1.0
        row["date"] = d
        rates_list.append(row)

    if not rates_list:
        raise RuntimeError("No valid FX data returned from API.")

    df = pd.DataFrame(rates_list)
    df = df.set_index("date").sort_index()
    df = df[currencies]
    return df

# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph():
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    # fully connect all currencies
    for i in range(len(CURRENCIES)):
        for j in range(i + 1, len(CURRENCIES)):
            G.add_edge(CURRENCIES[i], CURRENCIES[j])
    return G

# -----------------------------
# CIRCLE LAYOUT
# -----------------------------
def circle_layout():
    n = len(CURRENCIES)
    pos = {}
    for i, c in enumerate(CURRENCIES):
        angle = 2 * np.pi * i / n
        pos[c] = (np.cos(angle), np.sin(angle))
    return pos

# -----------------------------
# DRAW SNAPSHOT
# -----------------------------
def draw_snapshot(G, rates, pos, filename):
    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axis("off")

    # Compute dK/dt for all edges
    edge_weights = {}
    node_sums = {c: 0.0 for c in CURRENCIES}

    for u, v in G.edges():
        K_today = today_prices[u] / today_prices[v]
        K_yesterday = yesterday_prices[u] / yesterday_prices[v]
        dK = K_today - K_yesterday
        if dK == 0:
            continue
        # determine arrow direction
        start, end = (u, v) if dK < 0 else (v, u)
        edge_weights[(start, end)] = abs(dK)
        # sum for node size
        node_sums[u] += dK
        node_sums[v] += dK

    # Draw nodes
    sizes = [np.abs(node_sums[c])*5000 + 300 for c in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # Draw arrows
    for (u, v), w in edge_weights.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                mutation_scale=10 + 30*w,
                                linewidth=1 + 5*w,
                                color='gray',
                                connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plt.title(f"FX Flow Network ({timestamp} UTC)")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot: {filename}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    G = build_graph()
    pos = circle_layout()
    rates = fetch_rates(base="USD", currencies=CURRENCIES, days=2)

    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
    draw_snapshot(G, rates, pos, fname)
