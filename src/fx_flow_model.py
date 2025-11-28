import numpy as np
from datetime import datetime, timedelta
from fx_utils import *
from timeseries_tools import add_timeseries_features
from covariance_model import compute_covariance, compute_correlation
from alpha_model import compute_alpha_signals
from regression_model import run_linear_regression
from ml_model import train_ml_model, predict_next_day
from slippage import apply_slippage

if __name__ == "__main__":
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=60)
    end_date = today

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
    import pandas as pd
    print(pd.DataFrame({
        "currency": CURRENCIES,
        "flow_today": last,
        "flow_pred": pred,
        "pred_%": (pred - 1) * 100
    }))

    draw_flow(G, last)
