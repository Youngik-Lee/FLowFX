import os
import yfinance as yf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# SETTINGS
# -----------------------------
CURRENCIES = ["USD", "EUR", "JPY", "KRW", "GBP", "SGD", "HKD", "AUD"]

# Yahoo Finance Tickers for USD-based pairs (e.g., EURUSD=X is 1 EUR in USD)
TICKERS = {
    "EUR": "EURUSD=X", "JPY": "JPY=X", "KRW": "KRW=X",
    "GBP": "GBPUSD=X", "SGD": "SGDUSD=X", "HKD": "HKDUSD=X",
    "AUD": "AUDUSD=X"
}

OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FETCH FX DATA (YFINANCE)
# -----------------------------
def fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7):
    """
    Fetches historical FX rates using yfinance, converting the F/X=USD rate
    to the required USD/X=F rate (USD in foreign currency) via the reciprocal.
    """
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
    end_date = datetime.utcnow().date().isoformat()
    
    ticker_list = list(TICKERS.values())
    
    # Download data from yfinance and take the closing price
    data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)['Close']
    
    if data.empty:
        raise RuntimeError("No valid FX data returned from yfinance.")

    df = pd.DataFrame(index=data.index)

    # Convert Tickers back to Currency Names and apply the reciprocal
    for currency, ticker in TICKERS.items():
        if ticker in data.columns:
            # Reciprocal: 1 / (Foreign_Currency in USD) = USD in Foreign_Currency
            df[currency] = 1.0 / data[ticker]
        else:
            print(f"Warning: No data found for ticker {ticker}.")
            
    # Add the base currency (USD) which is always 1.0
    df[base] = 1.0
    
    # Select and clean the data
    df = df[[c for c in currencies if c in df.columns]].dropna()

    # Need at least two days (today's rate and yesterday's rate)
    if len(df) < 2:
        raise RuntimeError(f"Yfinance returned only {len(df)} days of valid data. Need at least 2.")
        
    print(f"Successfully fetched {len(df)} days of rates.")
    return df

# -----------------------------
# CIRCLE LAYOUT
# -----------------------------
def circular_layout(nodes):
    """Generates a circular layout for the network nodes."""
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos

# -----------------------------
# DRAW SNAPSHOT
# -----------------------------
def draw_snapshot(G, rates, filename):
    """Draws the currency flow network based on daily cross-rate changes."""
    if len(rates) < 2:
        print("Error: Not enough data points to calculate flow (need 2).")
        return

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]

    plt.figure(figsize=(10,10))
    ax = plt.gca()

    pos = circular_layout(G.nodes())

    arrow_edges = []
    node_flow_sum = {c:0.0 for c in G.nodes()} # Measures total volatility (Sum |dK|)

    for u, v in G.edges():
        # K_u/v: The cross-rate of u in terms of v 
        K_today = today_prices[u] / today_prices[v]
        K_yesterday = yesterday_prices[u] / yesterday_prices[v]
        
        dK = K_today - K_yesterday
        
        if abs(dK) < 1e-6:
            continue
        
        # Determine flow direction
        if dK > 0:
            start, end = v, u
        else:
            start, end = u, v
            
        # Arrow thickness: Linear scale of volatility (50 is the multiplier)
        width = abs(dK) * 50  
        arrow_edges.append((start, end, width))
        
        # Aggregate flow for node sizing
        node_flow_sum[u] += abs(dK)
        node_flow_sum[v] += abs(dK)

    # ----------------------------------------
    # MODIFIED: Linear Scaling for Node Size 
    # Node Size = Base Size + (Sum of all arrow widths attached to it) * Scaling Factor
    # ----------------------------------------
    BASE_SIZE = 300  # Minimum size for any node
    # We use a reduced multiplier (50000) to keep the max size manageable.
    # The original arrow scale factor was 50. Summing 7 pairs * 50 = max 350.
    # We need a much larger factor for node size.
    NODE_SCALE_FACTOR = 50000 

    node_sizes = [BASE_SIZE + abs(node_flow_sum[c]) * NODE_SCALE_FACTOR for c in G.nodes()] 
    
    # Apply a hard cap to prevent extreme sizes (e.g., if JPY or KRW were exceptionally volatile)
    MAX_NODE_SIZE = 2000
    node_sizes = [min(size, MAX_NODE_SIZE) for size in node_sizes]
    # ----------------------------------------

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9, edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw arrows
    for u, v, width in arrow_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                color='darkred',
                                linewidth=width,
                                mutation_scale=10 + width*2,
                                connectionstyle="arc3,rad=0.1",
                                zorder=2)
        ax.add_patch(arrow)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plt.title(f"FX Flow Network (Data Date: {rates.index[-1].strftime('%Y-%m-%d')})")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot: {filename}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # 1. Initialize the complete graph (all currencies connected)
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    for i, c1 in enumerate(CURRENCIES):
        for c2 in CURRENCIES[i+1:]:
            G.add_edge(c1, c2)

    try:
        # 2. Fetch the rates
        rates = fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7)

        # 3. Draw the resulting snapshot
        fname = os.path.join(OUTPUT_DIR, f"fx_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
        draw_snapshot(G, rates, fname)

    except RuntimeError as e:
        print(f"FATAL ERROR: {e}")
