import os
import requests
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# SETTINGS
# -----------------------------
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
        try:
            data = resp.json()
        except Exception:
            print(f"Skipping {d}: invalid JSON")
            continue
        if not data.get("success", True):
            print(f"Skipping {d}: API returned success=False")
            continue
        rates = data.get("rates")
        if not rates:
            print(f"Skipping {d}: no rates returned")
            continue
        row = {c: rates.get(c, np.nan) for c in currencies}
        row[base] = 1.0
        row["date"] = d
        rates_list.append(row)
        print(f"Fetched rates for {d}: {row}")

    if not rates_list:
        raise RuntimeError("No valid FX data returned from API.")

    df = pd.DataFrame(rates_list)
    df = df.set_index("date").sort_index()
    df = df[currencies]
    return df

# -----------------------------
# CIRCLE LAYOUT
# -----------------------------
def circular_layout(nodes):
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos

# -----------------------------
# DRAW SNAPSHOT
# -----------------------------
def draw_snapshot(G, rates, filename):
    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    plt.figure(figsize=(10,10))
    ax = plt.gca()

    pos = circular_layout(G.nodes())

    # Compute all dK/dt
    arrow_edges = []
    node_flow_sum = {c:0.0 for c in G.nodes()}

    for u, v in G.edges():
        K_today = today_prices[u] / today_prices[v]
        K_yesterday = yesterday_prices[u] / yesterday_prices[v]
        dK = K_today - K_yesterday
        if dK == 0:
            continue
        if dK > 0:
            start, end = v, u
        else:
            start, end = u, v
        width = abs(dK) * 5  # scaling factor for visibility
        arrow_edges.append((start, end, width))
        node_flow_sum[u] += dK
        node_flow_sum[v] += dK

    # Draw nodes with size proportional to |sum(dK/dt)|
    node_sizes = [50 + 2000 * abs(node_flow_sum[c]) for c in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for u, v, width in arrow_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='gray',
                                linewidth=width,
                                mutation_scale=10 + width*2,
                                connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plt.title(f"FX Flow Network ({timestamp} UTC)")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot: {filename}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    # Connect all currencies to each other
    for i, c1 in enumerate(CURRENCIES):
        for c2 in CURRENCIES[i+1:]:
            G.add_edge(c1, c2)

    rates = fetch_rates(base="USD", currencies=CURRENCIES, days=2)
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
    draw_snapshot(G, rates, fname)
