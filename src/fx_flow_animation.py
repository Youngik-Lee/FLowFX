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

# Yahoo Finance Tickers for USD-based pairs
TICKERS = {
"EUR": "EURUSD=X", "JPY": "JPY=X", "KRW": "KRW=X",
"GBP": "GBPUSD=X", "SGD": "SGDUSD=X", "HKD": "HKDUSD=X", "AUD": "AUDUSD=X"
}

OUTPUT_DIR = "output/animation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FETCH FX DATA (YFINANCE)
# -----------------------------

def fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7):
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
    end_date = datetime.utcnow().date().isoformat()

    # Download data from yfinance
    ticker_list = list(TICKERS.values())
    # Silence download progress output for a cleaner script run
    data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)['Close']
    
    if data.empty:
        raise RuntimeError("No valid FX data returned from yfinance.")

    df = pd.DataFrame(index=data.index)
    
    # Convert Tickers back to Currency Names and calculate rate against the Base (USD)
    for currency, ticker in TICKERS.items():
        if ticker in data.columns:
            # EURUSD=X is EUR/USD. 1.0 / (EUR/USD) gives USD/EUR, which is the USD rate.
            # This logic needs adjustment based on convention. Assuming Tickers are QUOTE/BASE (e.g., EURUSD=X means 1 EUR = X USD), 
            # we want the inverse to get USD's value in other currencies, which is standard for a USD-based analysis.
            df[currency] = 1.0 / data[ticker]
        else:
            print(f"Warning: No data for ticker {ticker}.")
            
    df[base] = 1.0 # Base currency rate against itself is 1.0
    
    # Filter columns and drop any remaining rows with missing data
    df = df[[c for c in currencies if c in df.columns]].dropna()
    
    if len(df) < 2:
        raise RuntimeError(f"Yfinance returned only {len(df)} days. Need at least 2.")
        
    print(f"Successfully fetched {len(df)} days of rates.")
    return df 

# -----------------------------
# CIRCLE LAYOUT
# -----------------------------

def circular_layout(nodes):
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos

# -----------------------------
# DRAW SNAPSHOT (CORRECTED FOR SINGLE MIDPOINT ARROW)
# -----------------------------

def draw_snapshot(G, rates, filename):
    if len(rates) < 2:
        print("Error: Not enough data points to calculate flow (need 2).")
        return

    today_prices = rates.iloc[-1]
    yesterday_prices = rates.iloc[-2]
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    pos = circular_layout(G.nodes())
    arrow_data = [] # Stores (start, end, width)
    node_flow_sum = {c:0.0 for c in G.nodes()}

    for u, v in G.edges():
        # K is the rate of u per v: (u/USD) / (v/USD)
        K_today = today_prices[u] / today_prices[v]
        K_yesterday = yesterday_prices[u] / yesterday_prices[v]
        dK = K_today - K_yesterday 

        if abs(dK) < 1e-6:
            continue

        # If dK > 0, u got stronger relative to v (u/v increased), so flow is from v to u.
        start, end = (v, u) if dK > 0 else (u, v) 
        
        # Arrow width: minimum 0.5, max 3
        width = max(min(abs(dK)*5, 3), 0.5)
        arrow_data.append((start, end, width))

        node_flow_sum[u] += abs(dK)
        node_flow_sum[v] += abs(dK)

    # Node size scaling
    BASE_SIZE = 300
    MAX_NODE_SIZE = 1200
    NODE_SCALE_FACTOR = 5000
    node_sizes = [min(BASE_SIZE + abs(node_flow_sum[c])*NODE_SCALE_FACTOR, MAX_NODE_SIZE) for c in G.nodes()]

    # Draw Nodes and Labels
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9, edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw the single line and the separate midpoint marker
    for u, v, width in arrow_data:
        x1, y1 = pos[u] # Start coordinates
        x2, y2 = pos[v] # End coordinates
        
        # Calculate straight-line midpoint (where the marker will be placed)
        xm, ym = (x1 + x2)/2, (y1 + y2)/2
        
        # 1. Draw the full curved line (no arrow head)
        line = FancyArrowPatch(
            (x1, y1), (x2, y2), 
            arrowstyle='-',  # Simple line style
            color='darkred', 
            linewidth=width, 
            mutation_scale=1, # No arrowhead needed
            connectionstyle="arc3,rad=0.1", 
            zorder=2
        )
        ax.add_patch(line)
        
        # 2. Place a directional marker at the midpoint
        
        # Calculate angle of the straight line segment (for rotation)
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Plot a small triangle marker at the midpoint (xm, ym)
        ax.plot(
            xm, ym, 
            marker='>',  # Use a triangle marker
            markersize=3 + width*2, # Size proportional to line width
            color='darkred',
            linestyle='', # Do not draw a line
            transform=ax.transData,
            rotation=angle_deg, # Rotate to align with the flow direction
            zorder=4
        )
        
    plt.title(f"FX Flow Network (Data Date: {rates.index[-1].strftime('%Y-%m-%d')})")
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved snapshot: {filename}")

# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":
    # Initialize complete graph
    G = nx.Graph()
    for c in CURRENCIES:
        G.add_node(c)
    # Add edges between all unique pairs of currencies
    for i, c1 in enumerate(CURRENCIES):
        for c2 in CURRENCIES[i+1:]:
            G.add_edge(c1, c2)

    try:
        # Fetch FX rates 
        # (It will print an error if run without internet access or if yfinance is down)
        rates = fetch_rates_yfinance(base="USD", currencies=CURRENCIES, days=7)
        
        # Draw snapshot
        fname = os.path.join(OUTPUT_DIR, f"fx_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
        draw_snapshot(G, rates, fname)
        
    except RuntimeError as e:
        print(f"FATAL ERROR: {e}")
