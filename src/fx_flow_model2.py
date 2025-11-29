import numpy as np
import pandas as pd
from fx_utils import *
from timeseries_tools import add_timeseries_features
from covariance_model import compute_covariance, compute_correlation
from alpha_model import compute_alpha_signals
from regression_model import run_linear_regression
from ml_model import train_ml_model, predict_next_day
from slippage import apply_slippage
from scipy.optimize import minimize
import os

OUTPUT_DIR = "output/model"

# --- helper functions ---
def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def save_summary_file(content: str):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Saved] {path}")

def predict_with_confidence(model, X):
    # Random Forest: mean + std across trees
    preds = np.array([tree.predict(X.values.reshape(1, -1)) for tree in model.estimators_])
    mean_pred = preds.mean()
    std_pred = preds.std()
    return mean_pred, std_pred

# --- Navier-Stokes calibration using multiple sources ---
def calibrate_navier(last_flows, combined_target, G):
    """Optimize NS parameters to fit combined target flows."""
    def loss(params):
        nu, gamma, f = params
        pred = simulate_step(last_flows, *build_navier_matrices(G), nu=nu, gamma=gamma, f=f*np.ones(len(last_flows)))
        return np.sum((pred - combined_target)**2)

    # initial guess
    init = [0.1, 0.1, 0.5]
    bounds = [(0, 1), (0, 1), (0, 2)]
    result = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
    nu_opt, gamma_opt, f_opt = result.x
    return nu_opt, gamma_opt, f_opt

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
    ml_mean, ml_std = predict_with_confidence(ml_model, X_last)
    slippage_pred = apply_slippage(ml_mean, volume=5_000_000)

    # --- combine predictions for NS calibration ---
    # Option 1: weighted average of regression, ML, and alpha
    reg_pred = lin_model.predict(rates.drop(columns=["USD"]))[-1]
    alpha_pred = alpha_df.iloc[-1].values
    combined_target = 0.5 * ml_mean + 0.3 * reg_pred.mean() + 0.2 * alpha_pred.mean()

    # --- Navier-Stokes simulation ---
    G = build_country_graph()
    nu, gamma, f = calibrate_navier(flows.iloc[-1].values, combined_target*np.ones(len(CURRENCIES)), G)
    pred = simulate_step(flows.iloc[-1].values, *build_navier_matrices(G), nu=nu, gamma=gamma, f=f*np.ones(len(CURRENCIES)))

    # --- build summary content ---
    summary_lines = []
    summary_lines.append("==== NAVIER SYSTEM PARAMETERS ====")
    summary_lines.append(f"nu: {nu:.6f}")
    summary_lines.append(f"gamma: {gamma:.6f}")
    summary_lines.append(f"f: {f:.6f}\n")

    summary_lines.append("==== ML PREDICTION ====")
    summary_lines.append(f"Predicted flow: {ml_mean:.6f}")
    summary_lines.append(f"Reliability (std across trees): {ml_std:.6f}")
    summary_lines.append(f"After slippage: {slippage_pred:.6f}\n")

    summary_lines.append("==== CURRENCY PREDICTIONS ====")
    for cur, flow_today, flow_pred in zip(CURRENCIES, flows.iloc[-1].values, pred):
        reliability = max(0, 1 - ml_std)  # simple reliability metric
        summary_lines.append(f"{cur}: today={flow_today:.6f}, predicted={flow_pred:.6f}, reliability={reliability:.3f}")

    save_summary_file("\n".join(summary_lines))

    # --- print to console ---
    print("\n".join(summary_lines))

    # --- optional visualization ---
    draw_flow(G, flows.iloc[-1].values)
