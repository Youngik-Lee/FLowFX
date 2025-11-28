import numpy as np
import pandas as pd
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

    # --- fetch FX rates ---
    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    flows = compute_flows(rates)

    if flows.empty:
        raise RuntimeError("Not enough historical FX data to compute flows")

    # --- time series & quant features ---
    rates_ts = add_timeseries_features(rates)
    cov = compute_covariance(rates)
    corr = compute_correlation(rates)
    alpha_df = compute_alpha_signals(rates_ts)

    # --- regression & ML models ---
    lin_model, coefs, intercept = run_linear_regression(rates)
    ml_model = train_ml_model(rates)

    X_last = rates.drop(columns=["USD"]).pct_change().iloc[-1].fillna(0)
    ml_pred = predict_next_day(ml_model, X_last)
    slippage_pred = apply_slippage(ml_pred, volume=5_000_000)

    print("ML next-day prediction:", ml_pred)
    print("After slippage:", slippage_pred)

    # --- Navier-Stokes simulation ---
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
