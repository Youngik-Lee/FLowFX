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
# Fetch FX rates from Yahoo directly
# -----------------------------
def fetch_rates(days=30):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    tickers = [f"USD{c}=X" for c in CURRENCIES if c != "USD"]
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Select adjusted close
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            df = df['Adj Close']
        else:
            df = df.iloc[:, :len(tickers)]

    # Rename columns to currency codes like USDEUR=X -> EUR
    new_cols = {}
    for t in df.columns:
        if isinstance(t, str) and t.startswith("USD") and t.endswith("=X"):
            new_cols[t] = t[3:6]  # 'USDEUR=X' -> 'EUR'
        elif isinstance(t, tuple) and t[0].startswith("USD") and t[0].endswith("=X"):
            new_cols[t] = t[0][3:6]
    df = df.rename(columns=new_cols)

    # Add USD as 1.0
    df["USD"] = 1.0
    df = df.ffill().bfill()

    # Ensure all currencies are present
    for c in CURRENCIES:
        if c not in df.columns:
            df[c] = np.nan

    df = df[CURRENCIES]
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
# Draw snapshot
# -----------------------------
def draw_snapshot(G, rates, pos, filename, title="FX Flow Network"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    if len(rates) < 2:
        raise RuntimeError("Not enough data to compare today and yesterday.")

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    dK_dict = {}
    node_weights = {c: 0.0 for c in G.nodes()}

    for u, v in G.edges():
        if u not in rates.columns or v not in rates.columns:
            continue

        # Convert to scalar float to avoid Series issues
        K_today = float(today_prices[u] / today_prices[v])
        K_yesterday = float(yesterday_prices[u] / yesterday_prices[v])
        dK = K_today - K_yesterday

        if dK == 0:
            continue

        start, end = (u, v) if dK < 0 else (v, u)
        dK_dict[(start, end)] = abs(dK)

        # accumulate node weights as sum(dK/dt)
        node_weights[u] += dK
        node_weights[v] += dK

    max_weight = max(abs(w) for w in node_weights.values()) + 1e-8
    node_sizes = [800 + 2000 * (abs(node_weights[n]) / max_weight) for n in G.nodes()]

    # Draw nodes
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
                                linewidth=3 + 20*weight,
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
    pos = fixed_layout()

    rates = fetch_rates(days=30)
    if len(rates) < 2:
        raise RuntimeError("Not enough valid FX data returned.")

    now = datetime.utcnow()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{now.strftime('%Y%m%d_%H%M%S')}.png")

    draw_snapshot(G, rates, pos, fname, title=f"FX Flow Network ({timestamp} UTC)")
    print(f"Saved snapshot: {fname}")
