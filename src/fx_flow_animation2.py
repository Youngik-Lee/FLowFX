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
# Fetch FX prices vs USD safely
# -----------------------------
def fetch_rates(base="USD", days=30):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    tickers = [f"{c}{base}=X" for c in CURRENCIES if c != base]

    df = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle multi-level
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df:
            df = df['Adj Close']

    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Map tickers to currency codes
    ticker_to_currency = {t: t.replace(f"{base}=X", "") for t in df.columns if t.endswith(f"{base}=X")}
    df = df.rename(columns=ticker_to_currency)

    # Keep only available currencies + USD
    available = list(ticker_to_currency.values())
    df = df[available]
    df[base] = 1.0
    if base not in df.columns:
        df[base] = 1.0

    # Reorder, placing USD first
    df = df[[base] + [c for c in CURRENCIES if c in df.columns and c != base]]

    # Fill missing data
    df = df.ffill().bfill()
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

    # Draw nodes as circles
    node_colors = np.abs(today_prices / yesterday_prices - 1)
    norm_colors = node_colors / (np.nanmax(node_colors) + 1e-8)
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color=norm_colors, cmap="coolwarm",
                           edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows based on dK/dt rule
    for u, v in G.edges():
        if u not in rates.columns or v not in rates.columns:
            continue

        K_today = (today_prices[u] / today_prices["USD"]) / (today_prices[v] / today_prices["USD"])
        K_yesterday = (yesterday_prices[u] / yesterday_prices["USD"]) / (yesterday_prices[v] / yesterday_prices["USD"])
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
