import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio.v2 as imageio  # Use v2 to avoid ImageIO v3 warning
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
DT = 1.0

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FX DATA FETCH FOR ANIMATION
# -----------------------------
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

# -----------------------------
# FLOW CALCULATION
# -----------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# -----------------------------
# COUNTRY GRAPH
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
def draw_frame(G, flow, pos, filename):
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=(flow-1), cmap="coolwarm")
    nx.draw_networkx_labels(G,pos)

    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue
        start, end = (u, v) if fu < fv else (v, u)
        mag = abs(fv - fu)
        nx.draw_networkx_edges(G, pos, edgelist=[(start,end)], width=3*mag, edge_color='gray')

    plt.axis("off")
    plt.savefig(filename)
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    G = build_country_graph()
    rates = fetch_rates_for_animation(base="USD")
    flows = compute_flows(rates)

    if flows.empty:
        raise RuntimeError("No FX flow data to animate")

    pos = nx.spring_layout(G, seed=2)

    frames = []
    for i in range(len(flows)):
        f = flows.iloc[i].values
        fname = os.path.join(OUTPUT_DIR, f"frame_{i}.png")
        draw_frame(G, f, pos, fname)
        frames.append(imageio.imread(fname))

    gif_path = os.path.join(OUTPUT_DIR, "flow_animation.gif")
    imageio.mimsave(gif_path, frames, fps=2)
    print(f"Saved animation: {gif_path}")
