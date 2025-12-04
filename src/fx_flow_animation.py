import os
import yfinance as yf
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
# Yahoo Finance Tickers for USD-based pairs
TICKERS = {
    "EUR": "EURUSD=X", 
    "JPY": "JPY=X", 
    "KRW": "KRW=X",
    "GBP": "GBPUSD=X", 
    "SGD": "SGDUSD=X", 
    "HKD": "HKDUSD=X", 
    "AUD": "AUDUSD=X"
}

OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FETCH FX DATA
# -----------------------------
def fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7):
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
    end_date = datetime.utcnow().date().isoformat()

    data = yf.download(list(TICKERS.values()), start=start_date, end=end_date, progress=False)['Close']
    if data.empty:
        raise RuntimeError("No FX data returned.")

    df = pd.DataFrame(index=data.index)

    # Convert to USD-based rates
    for currency, ticker in TICKERS.items():
        if ticker not in data.columns:
            print(f"Warning: {ticker} missing.")
            continue
        df[currency] = 1.0 / data[ticker]   # convert EURUSD=X → USD/EUR

    df[base] = 1.0  # USD/USD = 1

    df = df[[c for c in currencies if c in df.columns]].dropna()

    if len(df) < 2:
        raise RuntimeError("Not enough days returned.")

    print(f"Fetched {len(df)} FX days.")
    return df

# -----------------------------
# CIRCULAR LAYOUT
# -----------------------------

def circular_layout(nodes):
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos

# -----------------------------
# DRAW SNAPSHOT (ONE ARROW PER EDGE)
# -----------------------------

def draw_snapshot(G, rates, filename):
    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    pos = circular_layout(G.nodes())
    node_flow_sum = {c: 0.0 for c in G.nodes()}
    arrow_edges = []

    for u, v in G.edges():
        K_today = today_prices[u] / today_prices[v]
        K_yest = yesterday_prices[u] / yesterday_prices[v]
        dK = K_today - K_yest

        if abs(dK) < 1e-6:
            continue

        start, end = (v, u) if dK > 0 else (u, v)

        width = max(min(abs(dK)*5, 3), 0.5)

        arrow_edges.append((start, end, width))
        node_flow_sum[u] += abs(dK)
        node_flow_sum[v] += abs(dK)

    # -------------------------
    # Draw nodes
    # -------------------------

    BASE_SIZE = 300
    MAX_NODE_SIZE = 1200
    SCALE = 5000

    node_sizes = [
        min(BASE_SIZE + node_flow_sum[c]*SCALE, MAX_NODE_SIZE)
        for c in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="skyblue",
        edgecolors="black",
        alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # -------------------------
    # ONE ARROW PER EDGE
    # Arrowhead at midpoint
    # -------------------------

    for u, v, width in arrow_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch(
            (x1, y1), (0.95*x2, 0.95*y2),          
            arrowstyle="-|>",
            color="darkred",
            linewidth=width,
            mutation_scale=10 + width * 2,
            connectionstyle="arc3,rad=0.15",
        )
        ax.add_patch(arrow)

    plt.title(f"FX Flow Network ({rates.index[-1].strftime('%Y-%m-%d')})")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", filename)

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    G = nx.Graph()
    for c1 in CURRENCIES:
        for c2 in CURRENCIES:
            if c1 < c2:   # avoid duplicates
                G.add_edge(c1, c2)

    try:
        rates = fetch_rates_yfinance(days=7)
        fname = os.path.join(OUTPUT_DIR, f"fxflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
        draw_snapshot(G, rates, fname)
    except RuntimeError as e:
        print("FATAL:", e)
