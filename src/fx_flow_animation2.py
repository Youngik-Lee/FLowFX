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

    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df:
            df = df['Adj Close']

    if isinstance(df, pd.Series):
        df = df.to_frame()

    ticker_to_currency = {}
    for t in df.columns:
        col_name = t[-1] if isinstance(t, tuple) else t
        if col_name.endswith(f"{base}=X"):
            ticker_to_currency[t] = col_name.replace(f"{base}=X", "")

    df = df.rename(columns=ticker_to_currency)
    available = list(ticker_to_currency.values())
    df = df[available]
    df[base] = 1.0
    df = df[[c for c in CURRENCIES if c in df.columns]]  # reorder
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
# Draw FX snapshot with arrow thickness and node size
# -----------------------------
def draw_snapshot(G, rates, pos, filename, title="FX Flow Network"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    if len(rates) < 2:
        raise RuntimeError("Not enough data to compare today and yesterday.")

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    # Compute dK/dt for all edges
    dK_dict = {}
    node_weights = {c: 0.0 for c in G.nodes()}

    for u, v in G.edges():
        if u not in rates.columns or v not in rates.columns:
            continue

        K_today = (today_prices[u] / today_prices["USD"]) / (today_prices[v] / today_prices["USD"])
        K_yesterday = (yesterday_prices[u] / yesterday_prices["USD"]) / (yesterday_prices[v] / yesterday_prices["USD"])
        dK = K_today - K_yesterday

        if dK == 0:
            continue

        start, end = (u, v) if dK < 0 else (v, u)
        dK_dict[(start, end)] = abs(dK)

        # accumulate node weights
        node_weights[u] += abs(dK)
        node_weights[v] += abs(dK)

    # Normalize node sizes
    max_weight = max(node_weights.values()) + 1e-8
    node_sizes = [800 + 2000 * (node_weights[n] / max_weight) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for (start, end), weight in dK_dict.items():
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='gray',
                                linewidth=3 + 20*weight,  # proportional to abs(dK/dt)
                                mutation_scale=15 + 20*weight,
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
    pos = fixed_layout()  # you can shuffle or change positions if you want

    rates = fetch_rates(base="USD", days=30)
    if len(rates) < 2:
        raise RuntimeError("Not enough valid FX data returned.")

    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{now.strftime('%Y%m%d_%H%M%S')}.png")

    draw_snapshot(G, rates, pos, fname, title=f"FX Flow Network ({timestamp} UTC)")
    print(f"Saved snapshot: {fname}")
