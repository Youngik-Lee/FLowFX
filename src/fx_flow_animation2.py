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
# Fetch last 2 trading days of FX rates
# -----------------------------
def fetch_rates(base="USD", days=5):
    """
    Fetch the last few days to ensure we have 2 valid trading days.
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    tickers = [f"{c}{base}=X" for c in CURRENCIES if c != base]

    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if 'Adj Close' in df:
        df = df['Adj Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df[base] = 1.0
    for c in CURRENCIES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CURRENCIES].ffill().bfill()
    df = df.dropna()
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

    today_prices = rates.iloc[-1].values
    yesterday_prices = rates.iloc[-2].values
    ratios = today_prices / yesterday_prices

    # DEBUG: print data
    print("=== FX Data ===")
    print(rates.iloc[-2:])
    print("Ratios today/yesterday:", ratios)

    # Draw nodes
    node_colors = np.abs(ratios - 1)
    norm_colors = node_colors / (np.max(node_colors) + 1e-8)
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color=norm_colors, cmap="coolwarm",
                           edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Determine max difference for scaling
    max_diff = max(abs(ratios[CURRENCIES.index(u)] - ratios[CURRENCIES.index(v)])
                   for u, v in G.edges()) + 1e-8

    # Draw arrows for all edges
    for u, v in G.edges():
        r_u = ratios[CURRENCIES.index(u)]
        r_v = ratios[CURRENCIES.index(v)]

        start, end = (u, v) if r_u < r_v else (v, u)
        weight = abs(r_v - r_u)

        # Scale arrows for visibility
        linewidth = max(2, 8 * weight / max_diff)
        mutation_scale = 15 + 25 * weight / max_diff

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

    rates = fetch_rates(base="USD", days=5)

    # Ensure at least 2 rows
    if len(rates) < 2:
        raise RuntimeError("Not enough valid FX data returned.")

    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{now.strftime('%Y%m%d_%H%M%S')}.png")

    draw_snapshot(G, rates, pos, fname, title=f"FX Flow Network ({timestamp} UTC)")
    print(f"Saved snapshot: {fname}")
