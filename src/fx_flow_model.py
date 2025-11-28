import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fx_flow_api import fetch_rates_real
import requests

CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]

def fetch_rates_real(currencies=CURRENCIES, base="USD"):
    """
    Fetch real FX rates from a free API.
    Returns a DataFrame with 1 row and columns = currencies.
    """
    # Example: ExchangeRate API endpoint (replace with your API key)
    API_KEY = "YOUR_API_KEY_HERE"
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{base}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Check which key exists
        if "conversion_rates" in data:
            rates_data = data["conversion_rates"]
        elif "rates" in data:
            rates_data = data["rates"]
        else:
            raise ValueError(f"No FX rates found in API response: {data}")

        # Keep only requested currencies
        rates = {c: rates_data[c] for c in currencies if c in rates_data}
        missing = [c for c in currencies if c not in rates]
        if missing:
            print(f"Warning: missing rates for {missing}")

        return pd.DataFrame([rates])

    except requests.RequestException as e:
        print("Network/API error:", e)
        return pd.DataFrame(columns=currencies)
    except ValueError as e:
        print("Data error:", e)
        return pd.DataFrame(columns=currencies)
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
def compute_flows(rates):
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
        x0=[0.1,0.1,0.01],
        bounds=[(0,3),(0,3),(-0.05,0.05)]
    )
    nu, gamma, f = res.x
    return nu, gamma, f, A, L

# ---------------------------------------
# DRAW FLOW ARROWS
# ---------------------------------------
def draw_flow(G, flow):
    pos = nx.spring_layout(G, seed=3)
    plt.figure(figsize=(10,7))
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color=(flow-1), cmap="coolwarm")
    nx.draw_networkx_labels(G,pos)

    for u,v in G.edges():
        fu = flow[CURRENCIES.index(u)]
        fv = flow[CURRENCORIES.index(v)]
        if fu==fv: continue
        if fu < fv:
            start,end = u,v; mag = fv-fu
        else:
            start,end = v,u; mag = fu-fv

        nx.draw_networkx_edges(
            G,pos,edgelist=[(start,end)],width=3*mag,
            arrowstyle="->",arrowsize=20+80*mag
        )

    plt.title("FX Flow: Lower → Higher")
    plt.axis("off")
    plt.show()

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # load rates from real API
    rates = fetch_rates_real(CURRENCIES)
    flows = compute_flows(rates)

    G = build_country_graph()

    nu, gamma, f, A, L = calibrate(flows, G)
    print("NAVIER SYSTEM:")
    print("nu:",nu," gamma:",gamma," f:",f)

    last = flows.iloc[-1].values
    pred = simulate_step(last, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))

    print("\nPREDICTION:")
    print(pd.DataFrame({
        "currency": CURRENCIES,
        "flow_today": last,
        "flow_pred": pred,
        "pred_%": (pred-1)*100
    }))

    draw_flow(G, last)
