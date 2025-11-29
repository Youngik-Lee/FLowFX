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
    "EUR": "EURUSD=X", "JPY": "JPY=X", "KRW": "KRW=X",
    "GBP": "GBPUSD=X", "SGD": "SGDUSD=X", "HKD": "HKDUSD=X", "AUD": "AUDUSD=X"
}

OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FETCH FX DATA (YFINANCE)
# -----------------------------
def fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7):
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
    end_date = datetime.utcnow().date().isoformat()

    ticker_list = list(TICKERS.values())
    
    # Download data from yfinance
    data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)['Close']

    if data.empty:
        raise RuntimeError("No valid FX data returned from yfinance.")

    df = pd.DataFrame(index=data.index)

    # Convert Tickers back to Currency Names and reciprocal
    for currency, ticker in TICKERS.items():
        if ticker in data.columns:
            df[currency] = 1.0 / data[ticker]
        else:
            print(f"Warning: No data for ticker {ticker}.")

    df[base] = 1.0
    df = df[[c for c in currencies if c in df.columns]].dropna()

    if len(df) < 2:
        raise RuntimeError(f"Yfinance returned only {len(df)} days. Need at least 2.")

    print(f"Successfully fetched {len(df)} days of rates.")
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
# DRAW SNAPSHOT (STRAIGHT MIDPOINT ARROWS)
# -----------------------------
def draw_snapshot(G, rates, filename):
    if len(rates) < 2:
        print("Error: Not enough data points to calculate flow (need 2).")
        return

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    plt.figure(figsize=(10,10))
    ax = plt.gca()

    pos = circular_layout(G.nodes())
    node_flow_sum = {c:0.0 for c in G.nodes()}

    # Prepare midpoint arrows
    arrows = []
    for u, v in G.edges():
        K_today = today_prices[u] / today_prices[v]
        K_yesterday = yesterday_prices[u] / yesterday_prices[v]
        dK = K_today - K_yesterday
        if abs(dK) < 1e-6:
            continue

        # Determine direction: u -> v if dK > 0
        start, end = (v, u) if dK > 0 else (u, v)
        node_flow_sum[u] += abs(dK)
        node_flow_sum[v] += abs(dK)

        # Midpoint coordinates
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        xm, ym = (x1 + x2)/2, (y1 + y2)/2

        # Short arrow length (10% of edge length)
        sx, sy = (x2 - x1)*0.1, (y2 - y1)*0.1

        # Arrow width scaling
        width = max(min(abs(dK)*5, 3), 0.5)

        arrows.append((xm - sx, ym - sy, xm + sx, ym + sy, width))

    # Node size scaling
    BASE_SIZE = 300
    MAX_NODE_SIZE = 1200
    NODE_SCALE_FACTOR = 5000
    node_sizes = [min(BASE_SIZE + abs(node_flow_sum[c])*NODE_SCALE_FACTOR, MAX_NODE_SIZE) for c in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9, edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw straight midpoint arrows
    for x_start, y_start, x_end, y_end, width in arrows:
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                arrowstyle='-|>',
                                color='darkred',
                                linewidth=width,
                                mutation_scale=6 + width*2,
                                connectionstyle="arc3,rad=0.0",  # straight line
                                zorder=3)
        ax.add_patch(arrow)

    plt.title(f"FX Flow Network (Data Date: {rates.index[-1].strftime('%Y-%m-%d')})")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot: {filename}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Initialize complete graph
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    for i, c1 in enumerate(CURRENCIES):
        for c2 in CURRENCIES[i+1:]:
            G.add_edge(c1, c2)

    try:
        # Fetch FX rates
        rates = fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7)

        # Draw snapshot
        fname = os.path.join(OUTPUT_DIR, f"fx_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
        draw_snapshot(G, rates, fname)
    except RuntimeError as e:
        print(f"FATAL ERROR: {e}")
