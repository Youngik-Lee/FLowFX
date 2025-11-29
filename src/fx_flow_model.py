7import numpy as np
import pandas as pd
from fx_utils import *
from timeseries_tools import add_timeseries_features
from covariance_model import compute_covariance, compute_correlation
from alpha_model import compute_alpha_signals
from regression_model import run_linear_regression
from ml_model import train_ml_model, predict_next_day
from slippage import apply_slippage
import os
OUTPUT_DIR = "output/model"
def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def save_text(name: str, content: str):
    """Save text data into output/model directory."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Saved] {path}")
    
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
    save_text("ml_prediction.txt",
              f"ML next-day prediction:\n{ml_pred}\n\nAfter slippage:\n{slippage_pred}\n")
    # --- Navier-Stokes simulation ---
    G = build_country_graph()
    nu, gamma, f, A, L = calibrate(flows, G)

    navier_str = (
    "NAVIER SYSTEM PARAMETERS\n"
    f"nu: {nu}\n"
    f"gamma: {gamma}\n"
    f"f: {f}\n"
    "------------------------------\n"
    )
    save_text("navier_parameters.txt", navier_str)
    print("NAVIER SYSTEM:")
    print("nu:", nu, " gamma:", gamma, " f:", f)

    last = flows.iloc[-1].values
    pred = simulate_step(last, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
       
    pred_df = pd.DataFrame({
        "currency": CURRENCIES,
        "flow_today": last,
        "flow_pred": pred,
        "pred_%": (pred - 1) * 100
    })       
    print("\nPREDICTION:")
    print(pred_df)
    save_text("flow_prediction.txt", pred_df.to_string(index=False))
  
    draw_flow(G, last)
