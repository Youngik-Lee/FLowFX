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
    preds = np.array([tree.predict(X.values.reshape(1, -1)) for tree in model.estimators_])
    mean_pred = preds.mean()
    std_pred = preds.std()
    return mean_pred, std_pred

# --- Navier-Stokes calibration using dK/dt ---
def calibrate_navier(K_last, combined_target, G):
    """Optimize NS parameters to fit dK/dt targets using original calibrate function."""
    # Use same calibration to build matrices
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

if __name__ == "__main__":
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=60)
    end_date = today

    # --- fetch FX rates ---
    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)

    # --- define flows as K ---
    K_matrix = rates[CURRENCIES].copy()  # flows = K
    dK_dt = K_matrix.diff().fillna(0)    # discrete derivative = flow speed

    # --- features and targets ---
    K_features = K_matrix.shift(1).dropna()          # yesterday's K
    dK_targets = dK_dt.iloc[1:]                      # today's ΔK

    # --- time series & quant features ---
    rates_ts = add_timeseries_features(K_matrix)
    cov = compute_covariance(K_matrix)
    corr = compute_correlation(K_matrix)
    alpha_df = compute_alpha_signals(rates_ts)

    # --- regression & ML models ---
    lin_model, coefs, intercept = run_linear_regression(K_features, dK_targets)
    ml_model = train_ml_model(K_features, dK_targets)

    X_last = K_matrix.iloc[-1].drop("USD")           # latest K
    ml_mean, ml_std = predict_with_confidence(ml_model, X_last)
    slippage_pred = apply_slippage(ml_mean, volume=5_000_000)

    # --- combined target for NS calibration ---
    reg_pred = lin_model.predict(K_matrix.drop(columns=["USD"]))[-1]
    alpha_pred = alpha_df.iloc[-1].values
    combined_target = 0.5*ml_mean + 0.3*reg_pred.mean() + 0.2*alpha_pred.mean()

    # --- NS simulation ---
    G = build_country_graph()
    nu, gamma, f, A, L = calibrate_navier(K_matrix.iloc[-1].values, combined_target*np.ones(len(CURRENCIES)), G)
    dK_pred = simulate_step(K_matrix.iloc[-1].values, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
    K_next = K_matrix.iloc[-1].values + dK_pred    # update K

    # --- build summary ---
    summary_lines = []
    summary_lines.append("==== NAVIER SYSTEM PARAMETERS ====")
    summary_lines.append(f"nu: {nu:.6f}")
    summary_lines.append(f"gamma: {gamma:.6f}")
    summary_lines.append(f"f: {f:.6f}\n")

    summary_lines.append("==== ML PREDICTION ====")
    summary_lines.append(f"Predicted flow speed (dK/dt): {ml_mean:.6f}")
    summary_lines.append(f"Reliability (std across trees): {ml_std:.6f}")
    summary_lines.append(f"After slippage: {slippage_pred:.6f}\n")

    summary_lines.append("==== CURRENCY PREDICTIONS ====")
    for cur, K_today, dK in zip(CURRENCIES, K_matrix.iloc[-1].values, dK_pred):
        reliability = max(0, 1 - ml_std)
        summary_lines.append(f"{cur}: K_today={K_today:.6f}, dK/dt_pred={dK:.6f}, reliability={reliability:.3f}")

    save_summary_file("\n".join(summary_lines))
    print("\n".join(summary_lines))

    # --- optional visualization ---
    draw_flow(G, K_matrix.iloc[-1].values, dK_pred)
