import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import yfinance as yf

# ---------------------------------------
# CONFIG
# ---------------------------------------
CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
DT = 1.0

# Mapping currencies to Yahoo tickers for USD base
# Yahoo FX tickers format: CURUSD=X
def get_yahoo_tickers(base="USD"):
    tickers = {}
    for cur in CURRENCIES:
        if cur == base:
            tickers[cur] = None  # base currency
        else:
            tickers[cur] = f"{cur}{base}=X"
    return tickers

# ---------------------------------------
# FETCH FX RATES
# ---------------------------------------
def fetch_rates_yahoo(base="USD", start_date=None, end_date=None):
    if end_date is None:
        end_date = datetime.utcnow().date()
    if start_date is None:
        start_date = end_date - timedelta(days=60)

    tickers = get_yahoo_tickers(base)
    df_all = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

    for cur, ticker in tickers.items():
        if cur == base:
            df_all[cur] = 1.0  # base currency always 1
        else:
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            data.name = cur
            df_all = df_all.join(data, how='left')

    df_all = df_all.fillna(method='ffill')  # fill missing data
    df_all = df_all.fillna(method='bfill')
    return df_all

# ---------------------------------------
# GRAPH
# ---------------------------------------
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

# ---------------------------------------
# FLOW CALCULATION
# ---------------------------------------
def compute_flows(rates: pd.DataFrame):
    return (rates / rates.shift(1)).iloc[1:]

# ---------------------------------------
# NAVIER–STOKES SYSTEM
# ---------------------------------------
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
        bounds=[(0,3),(0,3),(-0.05,0.05)]
    )
    return (*res.x, A, L)

# ---------------------------------------
# DRAW FLOW ARROWS
# ---------------------------------------
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
        if fu < fv:
            start, end = u, v
            mag = fv - fu
        else:
            start, end = v, u
            mag = fu - fv

        nx.draw_networkx_edges(
            G, pos, edgelist=[(start, end)],
            width=3*mag,
            arrowstyle="->", arrowsize=20+80*mag
        )

    plt.title("FX Flow: Lower → Higher")
    plt.axis("off")
    plt.show()

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=60)
    end_date = today

    # fetch FX rates using Yahoo Finance
    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    flows = compute_flows(rates)

    if flows.empty:
        raise RuntimeError("Not enough historical FX data to compute flows")

    G = build_country_graph()

    nu, gamma, f, A, L = calibrate(flows, G)
    print("NAVIER SYSTEM:")
    print("nu:", nu, " gamma:", gamma, " f:", f)

    last = flows.iloc[-1].values
    pred = simulate_step(last, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))

    print("\nPREDICTION:")
    print(pd.DataFrame({
        "currency": CURRENCIES,
        "flow_today": last,
        "flow_pred": pred,
        "pred_%": (pred - 1) * 100
    }))

    draw_flow(G, last)
