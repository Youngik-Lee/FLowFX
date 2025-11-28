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
import yfinance as yf
import pandas as pd
import numpy as np

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]

def fetch_rates_yahoo(base="USD", start_date=None, end_date=None):
    tickers = []
    for c in CURRENCIES:
        if c == base:
            continue
        tickers.append(f"{c}{base}=X")  # e.g., EURUSD=X

    # Download data
    df_all = yf.download(tickers, start=start_date, end=end_date)
    if df_all.empty:
        raise RuntimeError("No FX data fetched from Yahoo")
    
    # Yahoo returns multiindex ['Adj Close']
    if 'Adj Close' in df_all:
        df_all = df_all['Adj Close']

    # Add base currency as 1
    df_all[base] = 1.0

    # Ensure all currencies present
    for c in CURRENCIES:
        if c not in df_all.columns:
            df_all[c] = np.nan

    df_all = df_all[CURRENCIES]  # reorder
    df_all = df_all.ffill().bfill()  # fill missing
    return df_all

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
    pos = nx.spring_layout(G, seed=3)
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color=(flow-1), cmap="coolwarm")
    nx.draw_networkx_labels(G,pos)

    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue
        start, end = (u, v) if fu < fv else (v, u)
        mag = abs(fv - fu)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(start, end)],
            width=3*mag,
            arrowstyle="->", arrowsize=20+80*mag
        )

    plt.title("FX Flow: Lower → Higher")
    plt.axis("off")
    plt.show()
