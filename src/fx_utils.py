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
