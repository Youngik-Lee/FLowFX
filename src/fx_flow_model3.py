import numpy as np
import pandas as pd
from fx_utils import *
from timeseries_tools import add_timeseries_features
from covariance_model import compute_covariance, compute_correlation
from alpha_model import compute_alpha_signals
from slippage import apply_slippage
from scipy.optimize import minimize

# -----------------------------
# ML / Regression imports
# -----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
    """ML prediction with mean and std across trees for confidence (multi-output)."""
    # X must be a 2D array or DataFrame/Series converted to (1, n_features)
    if isinstance(X, pd.Series):
        X_input = X.values.reshape(1, -1)
    elif isinstance(X, pd.DataFrame):
        X_input = X.values
    else:
        # Fallback for numpy array, ensuring 2D structure
        X_input = np.array(X).reshape(1, -1)

    preds = np.array([est.predict(X_input) for est in model.estimators_])
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred[0], std_pred[0] # Return 1D arrays for prediction/std

# -----------------------------
# Multi-output Linear Regression
# -----------------------------
def run_linear_regression_multi(X_df, y_df):
    # FIX 1: Drop "USD" from features before calculating percentage change
    X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values 
    y = y_df.values
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    coefs = np.array([est.coef_ for est in model.estimators_])
    intercepts = np.array([est.intercept_ for est in model.estimators_])
    return model, coefs, intercepts

# -----------------------------
# Multi-output ML (Random Forest)
# -----------------------------
def train_ml_model_multi(X_df, y_df):
    # FIX 1: Drop "USD" from features before calculating percentage change
    X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values 
    y = y_df.values
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)) 
    model.fit(X, y)
    return model

# -----------------------------
# Navier-Stokes calibration
# -----------------------------
def calibrate_navier(K_last, combined_target, G):
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
    K_matrix = rates[CURRENCIES].copy()       
    dK_dt = K_matrix.diff().fillna(0)         

    # --- features and targets ---
    K_features = K_matrix.shift(1).dropna()
    # FIX 2: Drop "USD" target for consistent 7-target prediction
    dK_targets_non_usd = dK_dt.iloc[1:].drop(columns=["USD"], errors='ignore')

    # --- time series & quant features ---
    rates_ts = add_timeseries_features(K_matrix)
    cov = compute_covariance(K_matrix)
    corr = compute_correlation(K_matrix)
    alpha_df = compute_alpha_signals(rates_ts)

    # --- regression & ML models ---
    lin_model, coefs, intercepts = run_linear_regression_multi(K_features, dK_targets_non_usd)
    ml_model = train_ml_model_multi(K_features, dK_targets_non_usd)

    # --- predict dK/dt for last day ---
    
    # FIX 3: Prepare the single feature vector X_last consistently (7 non-USD features)
    K_last_2_rows = K_matrix.iloc[-2:].copy() 
    X_features_last = K_last_2_rows.drop(columns=["USD"]).pct_change().fillna(0)
    X_last_row_df = X_features_last.iloc[-1].to_frame().T 

    # Predict using the 7-feature vector
    ml_mean, ml_std = predict_with_confidence(ml_model, X_last_row_df)
    slippage_pred = apply_slippage(ml_mean, volume=5_000_000)

    # --- combine targets for NS calibration ---
    # FIX 4: Use .values to remove feature names and suppress UserWarning
    reg_pred = lin_model.predict(X_last_row_df.values)[0] 
    
    # If alpha_df has all CURRENCIES, drop USD to match ml_mean and reg_pred (size 7)
    alpha_pred_non_usd = alpha_df.iloc[-1].drop(labels=["USD"], errors='ignore').values 
    
    # combined_target is size 7 (non-USD currencies)
    combined_target = 0.5*ml_mean + 0.3*reg_pred.mean() + 0.2*alpha_pred_non_usd
    combined_target = apply_slippage(combined_target, volume=5_000_000)

    # --- NS simulation ---
    # Need to pad combined_target back to 8 currencies for NS
    combined_target_full = np.insert(combined_target, CURRENCIES.index("USD"), 0)
    K_last_full = K_matrix.iloc[-1].values 
    
    # FIX 5: Corrected argument for build_country_graph to use 'correlation' instead of 'covariance'
    G = build_country_graph(correlation=corr)
    
    nu, gamma, f, A, L = calibrate_navier(K_last_full, combined_target_full, G)
    dK_pred = simulate_step(K_last_full, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
    K_next = K_last_full + dK_pred

    # --- summary file ---
    summary_lines = []
    summary_lines.append("==== NAVIER SYSTEM PARAMETERS ====")
    summary_lines.append(f"nu: {nu:.6f}")
    summary_lines.append(f"gamma: {gamma:.6f}")
    summary_lines.append(f"f: {f:.6f}\n")

    summary_lines.append("==== ML + Regression + Alpha (Non-USD Currencies) ====")
    summary_lines.append(f"ML Predicted dK/dt: {ml_mean}")
    summary_lines.append(f"Regression contribution: {reg_pred}")
    summary_lines.append(f"Alpha contribution: {alpha_pred_non_usd}")
    summary_lines.append(f"After slippage: {slippage_pred}\n")

    summary_lines.append("==== CURRENCY PREDICTIONS ====")
    # Simple reliability calculation using the mean standard deviation of the ML prediction
    reliability = max(0, 1 - ml_std.mean()) 
    for cur, K_today, dK in zip(CURRENCIES, K_matrix.iloc[-1].values, dK_pred):
        summary_lines.append(f"{cur}: K_today={K_today:.6f}, dK/dt_pred={dK:.6f}, reliability={reliability:.3f}")

    save_summary_file("\n".join(summary_lines))
    print("\n".join(summary_lines))

    # --- optional visualization ---
    # draw_flow(G, K_matrix.iloc[-1].values, dK_pred)
