import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio.v2 as imageio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch
CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
# -----------------------------
# Directories
# -----------------------------
OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FIXED NODE POSITIONS
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
# FETCH FX DATA
# -----------------------------
def fetch_rates(base="USD", days=60):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    tickers = [f"{c}{base}=X" for c in CURRENCIES if c != base]
    df = yf.download(tickers, start=start_date, end=end_date)
    if 'Adj Close' in df:
        df = df['Adj Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df[base] = 1.0
    for c in CURRENCIES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CURRENCIES].ffill().bfill()
    return df

# -----------------------------
# COMPUTE FLOWS
# -----------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# -----------------------------
# BUILD NETWORK GRAPH
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
# DRAW FRAME
# -----------------------------
def draw_frame(G, flow, pos, filename, title="FX Flow Network"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Directed graph for arrows
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())

    # Build directed edges based on flow differences
    max_diff = 0
    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue
        # Direction: from weaker → stronger
        start, end = (u, v) if fu < fv else (v, u)
        mag = abs(fv - fu)
        DG.add_edge(start, end, weight=mag)
        if mag > max_diff:
            max_diff = mag

    # Draw nodes with color reflecting daily change
    norm_flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow) + 1e-8)
    nx.draw_networkx_nodes(DG, pos, node_size=1500,
                           node_color=norm_flow, cmap="coolwarm")
    nx.draw_networkx_labels(DG, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for u, v, data in DG.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Normalize weight for consistent scaling
        norm_weight = data['weight'] / (max_diff + 1e-8)

        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='gray',
                                linewidth=2 + 6*norm_weight,
                                mutation_scale=15 + 30*norm_weight,
                                connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    G = build_country_graph()
    pos = fixed_layout()

    rates = fetch_rates(base="USD")
    flows = compute_flows(rates)

    if flows.empty:
        raise RuntimeError("No FX flow data to visualize")

    # Pick latest flow for snapshot
    latest_flow = flows.iloc[-1].values
    timestamp = flows.index[-1].strftime("%Y-%m-%d")
    fname = os.path.join(OUTPUT_DIR, f"fx_flow_{timestamp}.png")
    draw_frame(G, latest_flow, pos, fname, title=f"FX Flow Network ({timestamp})")

    print(f"Saved snapshot: {fname}")
