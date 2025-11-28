import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import requests

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]
DT = 1.0

# ---------------------------------------
# FX RATES API — historical data (keyless)
# ---------------------------------------
def fetch_rates_history(currencies=CURRENCIES, base="USD",
                        start_date=None, end_date=None):
    """
    Fetch historical FX rates from exchangerate.host between start_date and end_date (inclusive).
    Returns a DataFrame with index = dates, columns = currencies (relative to base).
    """
    if end_date is None:
        end_date = datetime.utcnow().date()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    all_rates = []
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    for d in dates:
        url = f"https://api.exchangerate.host/{d.strftime('%Y-%m-%d')}"
        params = {
            "base": base,
            "symbols": ",".join(currencies)
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if "rates" not in data:
                print("Warning: no rates for", d, data)
                continue
            rates = data["rates"]
            row = {c: rates.get(c, np.nan) for c in currencies}
            row["date"] = d
            all_rates.append(row)
        except Exception as e:
            print("Error fetching for date", d, ":", e)
            continue

    df = pd.DataFrame(all_rates)
    if df.empty:
        raise RuntimeError("No FX data could be fetched")
    df = df.set_index("date").sort_index()
    return df

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
    # fetch past 60 days FX rates
    rates = fetch_rates_history(CURRENCIES, base="USD",
                                start_date=datetime.utcnow().date() - timedelta(days=60),
                                end_date=datetime.utcnow().date())

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
