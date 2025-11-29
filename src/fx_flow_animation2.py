import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Fixed node positions
# -----------------------------
def fixed_layout():
    return {
        "USD": (0.0, 0.0),
        "EUR": (1.0, -0.3),
        "JPY": (0.6, -1.0),
        "KRW": (-0.6, -1.0),
        "HKD": (-1.0, -0.3),
        "SGD": (-1.0, 0.5),
        "AUD": (0.0, 1.0),
        "GBP": (1.0, 0.5),
    }

# -----------------------------
# Fetch FX prices vs USD
# -----------------------------
def fetch_rates(base="USD", days=30):
    """Fetch FX prices vs USD for the last 'days' days"""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    tickers = [f"{c}{base}=X" for c in CURRENCIES if c != base]

    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if 'Adj Close' in df:
        df = df['Adj Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df[base] = 1.0  # USD = 1
    for c in CURRENCIES:
        if c not in df.columns:
            df[c] = np.nan

    df = df[CURRENCIES].ffill().bfill()  # fill missing prices
    return df

# -----------------------------
# Build graph
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
# Draw FX snapshot
# -----------------------------
def draw_snapshot(G, rates, pos, filename, title="FX Flow Network"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    if len(rates) < 2:
        raise RuntimeError("Not enough data to compare today and yesterday.")

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]
    usd_idx = CURRENCIES.index("USD")

    # Draw nodes
    node_colors = np.abs(today_prices / yesterday_prices - 1)
    norm_colors = node_colors / (np.nanmax(node_colors) + 1e-8)
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color=norm_colors, cmap="coolwarm",
                           edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for u, v in G.edges():
        idx_u = CURRENCIES.index(u)
        idx_v = CURRENCIES.index(v)

        # Skip edge if a currency is NaN
        if np.isnan(today_prices[idx_u]) or np.isnan(today_prices[idx_v]) \
           or np.isnan(yesterday_prices[idx_u]) or np.isnan(yesterday_prices[idx_v]):
            continue

        # Compute K via USD as pivot
        K_today = (today_prices[idx_u] / today_prices[usd_idx]) / \
                  (today_prices[idx_v] / today_prices[usd_idx])
        K_yesterday = (yesterday_prices[idx_u] / yesterday_prices[usd_idx]) / \
                      (yesterday_prices[idx_v] / yesterday_prices[usd_idx])
        dK = K_today - K_yesterday

        if dK == 0:
            continue  # no arrow

        start, end = (u, v) if dK < 0 else (v, u)
        weight = abs(dK)

        linewidth = max(2, 15 * weight)
        mutation_scale = 15 + 30 * weight

        x1, y1 = pos[start]
        x2, y2 = pos[end]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='gray',
                                linewidth=linewidth,
                                mutation_scale=mutation_scale,
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
    pos = fixed_layout()

    rates = fetch_rates(base="USD", days=30)
    if len(rates) < 2:
        raise RuntimeError("Not enough valid FX data returned.")

    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{now.strftime('%Y%m%d_%H%M%S')}.png")

    draw_snapshot(G, rates, pos, fname, title=f"FX Flow Network ({timestamp} UTC)")
    print(f"Saved snapshot: {fname}")
