import numpy as np
import pandas as pd
from fx_utils import *
from timeseries_tools import add_timeseries_features
from covariance_model import compute_covariance, compute_correlation
from alpha_model import compute_alpha_signals
from regression_model import run_linear_regression  # will override for multi-output
from ml_model import train_ml_model, predict_next_day
from slippage import apply_slippage
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import os

OUTPUT_DIR = "output/model"

# -----------------------------
# Helper functions
# -----------------------------
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
    """ML prediction with mean and std across trees for confidence."""
    preds = np.array([tree.predict(X.values.reshape(1, -1)) for tree in model.estimators_])
    mean_pred = preds.mean(axis=0)   # multi-output mean
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred

# -----------------------------
# Multi-output Linear Regression
# -----------------------------
def run_linear_regression_multi(X_df, y_df):
    """
    Fits multi-output linear regression to predict all currencies' dK/dt
    """
    X = X_df.pct_change().fillna(0).values
    y = y_df.values
    model = MultiOutputRegressor(LinearRegression()).fit(X, y)
    coefs = np.array([est.coef_ for est in model.estimators_])
    intercepts = np.array([est.intercept_ for est in model.estimators_])
    return model, coefs, intercepts

# -----------------------------
# Navier-Stokes calibration
# -----------------------------
def calibrate_navier(K_last, combined_target, G):
    """Optimize NS parameters to fit dK/dt targets."""
    _, _, _, A, L = calibrate(pd.DataFrame([K_last], columns=CURRENCIES), G)

    def loss(params):
        nu, gamma, f = params
        dK_pred = simulate_step(K_last, A, L, nu, gamma, f*np.ones(len(K_last)))
        return np.sum((dK_pred - combined_target)**2)

    init = [0.1, 0.1, 0.5]
    bounds = [(0, 1), (0, 1), (0, 2)]
    result = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
    nu_opt, gamma_opt, f_opt = result.x
    return nu_opt, gamma_opt, f_opt, A, L

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=60)
    end_date = today

    # --- fetch FX rates ---
    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    K_matrix = rates[CURRENCIES].copy()       # K = FX rates
    dK_dt = K_matrix.diff().fillna(0)         # flow speed

    # --- features and targets ---
    K_features = K_matrix.shift(1).dropna()
    dK_targets = dK_dt.iloc[1:]

    # --- time series & quant features ---
    rates_ts = add_timeseries_features(K_matrix)
    cov = compute_covariance(K_matrix)
    corr = compute_correlation(K_matrix)
    alpha_df = compute_alpha_signals(rates_ts)

    # --- regression & ML models ---
    lin_model, coefs, intercepts = run_linear_regression_multi(K_features, dK_targets)
    ml_model = train_ml_model(K_features, dK_targets)

    # --- predict dK/dt for last day ---
    X_last = K_matrix.iloc[-1].drop("USD")
    ml_mean, ml_std = predict_with_confidence(ml_model, X_last)
    slippage_pred = apply_slippage(ml_mean, volume=5_000_000)

    # --- combine targets for NS calibration ---
    reg_pred = lin_model.predict(K_matrix.drop(columns=["USD"]))[-1]
    alpha_pred = alpha_df.iloc[-1].values
    combined_target = 0.5*ml_mean + 0.3*reg_pred.mean() + 0.2*alpha_pred
    combined_target = apply_slippage(combined_target, volume=5_000_000)

    # --- NS simulation ---
    G = build_country_graph(covariance=cov)  # optionally pass covariance for edge weights
    nu, gamma, f, A, L = calibrate_navier(K_matrix.iloc[-1].values, combined_target, G)
    dK_pred = simulate_step(K_matrix.iloc[-1].values, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
    K_next = K_matrix.iloc[-1].values + dK_pred

    # --- summary file ---
    summary_lines = []
    summary_lines.append("==== NAVIER SYSTEM PARAMETERS ====")
    summary_lines.append(f"nu: {nu:.6f}")
    summary_lines.append(f"gamma: {gamma:.6f}")
    summary_lines.append(f"f: {f:.6f}\n")

    summary_lines.append("==== ML + Regression + Alpha ====")
    summary_lines.append(f"Predicted flow speed (dK/dt): {ml_mean}")
    summary_lines.append(f"Regression contribution: {reg_pred}")
    summary_lines.append(f"Alpha contribution: {alpha_pred}")
    summary_lines.append(f"After slippage: {slippage_pred}\n")

    summary_lines.append("==== CURRENCY PREDICTIONS ====")
    for cur, K_today, dK in zip(CURRENCIES, K_matrix.iloc[-1].values, dK_pred):
        reliability = max(0, 1 - ml_std.mean())  # simple reliability
        summary_lines.append(f"{cur}: K_today={K_today:.6f}, dK/dt_pred={dK:.6f}, reliability={reliability:.3f}")

    save_summary_file("\n".join(summary_lines))
    print("\n".join(summary_lines))

    # --- optional visualization ---
    # draw_flow(G, K_matrix.iloc[-1].values, dK_pred)
