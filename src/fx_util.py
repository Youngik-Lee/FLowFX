import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import yfinance as yf

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
DT = 1.0

# -------------------------
# FX YAHOO TICKERS
# -------------------------
def get_yahoo_tickers(base="USD"):
    tickers = {
        "USD":"USDX=X", "EUR":"EURUSD=X", "JPY":"JPY=X", "KRW":"KRW=X",
        "GBP":"GBPUSD=X", "SGD":"SGD=X", "HKD":"HKD=X", "AUD":"AUDUSD=X"
    }
    return tickers

# -------------------------
# FETCH RATES
# -------------------------
def fetch_rates_yahoo(base="USD", start_date=None, end_date=None):
    if end_date is None:
        end_date = datetime.utcnow().date()
    if start_date is None:
        start_date = end_date - timedelta(days=60)

    tickers = get_yahoo_tickers(base)
    df_all = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

    for cur, ticker in tickers.items():
        if cur == base:
            df_all[cur] = 1.0
        else:
            data = yf.download(ticker, start=start_date, end=end_date)
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            else:
                data = data.iloc[:,0]
            data.name = cur
            df_all = df_all.join(data, how='left')

    df_all = df_all.fillna(method='ffill').fillna(method='bfill')
    return df_all

# -------------------------
# BUILD GRAPH
# -------------------------
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

# -------------------------
# FLOWS
# -------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# -------------------------
# NAVIER-STOKES
# -------------------------
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
        pred = simulate_step(flows.iloc[t].values, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
        total += np.mean((pred - flows.iloc[t+1].values)**2)
    return total

def calibrate(flows, G):
    A = nx.to_numpy_array(G, nodelist=CURRENCIES)
    A = A / A.sum(axis=1, keepdims=True)
    L = laplacian(G)
    res = minimize(lambda x: loss(x, flows, A, L), x0=[0.1, 0.1, 0.0], bounds=[(0,3),(0,3),(-0.05,0.05)])
    return (*res.x, A, L)

# -------------------------
# DRAW FLOW
# -------------------------
def draw_flow(G, flow):
    pos = nx.spring_layout(G, seed=3)
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=(flow-1), cmap="coolwarm")
    nx.draw_networkx_labels(G,pos)
    for u, v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCIES.index(v)]
        if fu == fv:
            continue
        if fu < fv:
            start, end = u, v
            mag = fv - fu
        else:
            start, end = v, u
            mag = fu - fv
        nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], width=3*mag,
                               arrowstyle="->", arrowsize=20+80*mag)
    plt.title("FX Flow: Lower → Higher")
    plt.axis("off")
    plt.show()
