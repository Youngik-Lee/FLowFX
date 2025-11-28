import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
DT = 1.0

# -------------------------------
# FETCH FX RATES FROM YAHOO
# -------------------------------
def fetch_rates_yahoo(base="USD", start_date=None, end_date=None):
    """
    Fetch historical FX rates from Yahoo Finance.
    Returns a DataFrame with index=dates, columns=currencies relative to base.
    """
    if start_date is None:
        start_date = pd.Timestamp.today() - pd.Timedelta(days=60)
    if end_date is None:
        end_date = pd.Timestamp.today()

    df_all = pd.DataFrame()
    for c in CURRENCIES:
        if c == base:
            continue
        ticker = f"{c}{base}=X"  # e.g., EURUSD=X
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            print(f"Warning: No data for {ticker}")
            continue
        df_all[c] = data['Close']  # use 'Close' as adjusted price
    df_all[base] = 1.0
    df_all = df_all.ffill().bfill()  # fill missing
    return df_all[CURRENCIES]

# -------------------------------
# BUILD COUNTRY GRAPH
# -------------------------------
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

# -------------------------------
# COMPUTE FLOWS
# -------------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# -------------------------------
# NAVIER-STOKES SYSTEM
# -------------------------------
def laplacian(G):
    return nx.laplacian_matrix(G, nodelist=CURRENCIES).toarray()

def advective(u, A):
    return u * (A @ u)

def simulate_step(u, A, L, nu, gamma, forcing):
    return u + DT * (-advective(u, A) + nu * (L @ u) - gamma*u + forcing)

def loss(params, flows, A, L):
    nu, gamma, f = params
    total = 0
    for t in range(len(flows)-1):
        pred = simulate_step(
            flows.iloc[t].values, A, L, nu, gamma, f*np.ones(len(CURRENCIES))
        )
        total += np.mean((pred - flows.iloc[t+1].values)**2)
    return total

def calibrate(flows, G):
    A = nx.to_numpy_array(G, nodelist=CURRENCIES)
    A = A / A.sum(axis=1, keepdims=True)
    L = laplacian(G)

    res = minimize(
        lambda x: loss(x, flows, A, L),
        x0=[0.1, 0.1, 0.0],
        bounds=[(0,1),(0,1),(-0.5,0.5)]
    )
    return (*res.x, A, L)

# -------------------------------
# DRAW FLOW
# -------------------------------
def draw_flow(G, flow):
    """
    Draw FX flow network.
    - G: networkx Graph (or DiGraph)
    - flow: np.array or list of flows (same order as CURRENCIES)
    """
    # Convert to DiGraph for directional arrows
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    
    # Add edges in direction: lower → higher
    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue  # skip equal flows
        start, end = (u, v) if fu < fv else (v, u)
        mag = abs(fv - fu)
        DG.add_edge(start, end, weight=mag)
    
    pos = nx.spring_layout(DG, seed=42)
    plt.figure(figsize=(10, 7))
    
    # Draw nodes colored by flow magnitude
    node_colors = flow - np.min(flow)
    nx.draw_networkx_nodes(DG, pos, node_size=1500, node_color=node_colors, cmap="coolwarm")
    nx.draw_networkx_labels(DG, pos, font_size=12, font_weight="bold")
    
    # Draw edges with width proportional to flow difference
    for u, v, data in DG.edges(data=True):
        nx.draw_networkx_edges(
            DG, pos,
            edgelist=[(u, v)],
            width=3 + 5*data['weight'],  # scale width by magnitude
            alpha=0.7,
            arrowstyle='-|>',
            arrowsize=15 + 30*data['weight'],  # scale arrow size
            edge_color='gray'
        )
    
    plt.title("FX Flow Network: Lower → Higher")
    plt.axis('off')
    plt.show()
