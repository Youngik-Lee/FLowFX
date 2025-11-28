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
DT = 1.0

BASE_OUTPUT = "output"
ANIM_DIR = os.path.join(BASE_OUTPUT, "animation")  # NEW
os.makedirs(ANIM_DIR, exist_ok=True)              # NEW

# -------------------------------------------------------
# FETCH FX DATA
# -------------------------------------------------------
def fetch_rates_for_animation(base="USD", start_date=None, end_date=None):
    today = datetime.utcnow().date()
    if end_date is None or end_date > today:
        end_date = today
    if start_date is None:
        start_date = end_date - timedelta(days=60)

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

# -------------------------------------------------------
# COMPUTE FLOWS
# -------------------------------------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# -------------------------------------------------------
# COUNTRY GRAPH
# -------------------------------------------------------
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

# -------------------------------------------------------
# DRAW ONE FRAME
# -------------------------------------------------------
def draw_frame(G, flow, pos, filename):
    plt.figure(figsize=(10,7))
    ax = plt.gca()

    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())

    # Build directional edges (lower → higher)
    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue
        start, end = (u, v) if fu < fv else (v, u)
        mag = abs(fv - fu)
        DG.add_edge(start, end, weight=mag)

    # Node color normalization
    norm_flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow) + 1e-8)

    # Draw arrows
    for u, v, data in DG.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>',
            color='gray',
            linewidth=1.5 + 3*data['weight'],
            mutation_scale=15 + 25*data['weight'],
            connectionstyle="arc3,rad=0.07",
            zorder=1,
        )
        ax.add_patch(arrow)

    # Draw nodes
    nx.draw_networkx_nodes(
        DG, pos,
        node_size=1500,
        node_color=norm_flow,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=2,
        zorder=2,
    )

    # Draw labels
    nx.draw_networkx_labels(
        DG, pos,
        font_size=12,
        font_weight="bold",
        zorder=3
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# -------------------------------------------------------
# MAIN ANIMATION
# -------------------------------------------------------
if __name__ == "__main__":
    G = build_country_graph()
    rates = fetch_rates_for_animation(base="USD")
    flows = compute_flows(rates)

    if flows.empty:
        raise RuntimeError("No FX flow data to animate")

    pos = nx.spring_layout(G, seed=42)

    frames = []
    for i in range(len(flows)):
        f = flows.iloc[i].values
        fname = os.path.join(ANIM_DIR, f"frame_{i}.png")  # NEW
        draw_frame(G, f, pos, fname)
        frames.append(imageio.imread(fname))

    gif_path = os.path.join(ANIM_DIR, "flow_animation.gif")  # NEW
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"Saved animation: {gif_path}")
