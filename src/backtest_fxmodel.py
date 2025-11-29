import pandas as pd
import numpy as np
from fx_flow_model import predict_with_confidence, run_linear_regression_multi, train_ml_model_multi, apply_slippage, calibrate_navier, CURRENCIES, add_timeseries_features, compute_covariance, compute_correlation, compute_alpha_signals, build_country_graph, simulate_step

# -----------------------------
# Load historical FX rates
# -----------------------------
rates = pd.read_csv("historical_rates.csv", index_col=0, parse_dates=True)
K_matrix = rates[CURRENCIES].copy()
dK_actual = K_matrix.diff().fillna(0)

# -----------------------------
# Backtest parameters
# -----------------------------
window = 60  # rolling window size (days)
results = []

for t in range(window, len(K_matrix)-1):
    K_window = K_matrix.iloc[t-window:t]
    K_last = K_matrix.iloc[t]
    K_next_actual = K_matrix.iloc[t+1]

    # --- features ---
    K_features = K_window.shift(1).dropna()
    dK_targets_non_usd = K_window.diff().iloc[1:].drop(columns=["USD"], errors='ignore')
    
    # --- train models ---
    lin_model, coefs, intercepts = run_linear_regression_multi(K_features, dK_targets_non_usd)
    ml_model = train_ml_model_multi(K_features, dK_targets_non_usd)

    # --- ML prediction ---
    X_last_row = K_window.iloc[-2:].drop(columns=["USD"]).pct_change().fillna(0).iloc[-1].to_frame().T
    ml_mean, ml_std = predict_with_confidence(ml_model, X_last_row)
    slippage_pred = apply_slippage(ml_mean, volume=5_000_000)

    # --- regression & alpha ---
    reg_pred = lin_model.predict(X_last_row.values)[0]
    rates_ts = add_timeseries_features(K_window)
    alpha_pred_non_usd = compute_alpha_signals(rates_ts).iloc[-1].values[:7]

    combined_target = 0.5*ml_mean + 0.3*reg_pred.mean() + 0.2*alpha_pred_non_usd
    combined_target = apply_slippage(combined_target, volume=5_000_000)
    combined_target_full = np.insert(combined_target, CURRENCIES.index("USD"), 0)
    
    # --- Navier-Stokes simulation ---
    G = build_country_graph()
    nu, gamma, f, A, L = calibrate_navier(K_last.values, combined_target_full, G)
    dK_pred = simulate_step(K_last.values, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))

    # --- collect results ---
    for i, cur in enumerate(CURRENCIES):
        actual = K_next_actual[cur] - K_last[cur]
        pred = dK_pred[i]
        results.append({
            "date": K_matrix.index[t],
            "currency": cur,
            "actual_change": actual,
            "predicted_change": pred,
            "direction_correct": np.sign(actual) == np.sign(pred)
        })

# -----------------------------
# Convert to DataFrame and compute metrics
# -----------------------------
results_df = pd.DataFrame(results)
directional_accuracy = results_df.groupby("currency")["direction_correct"].mean() * 100
mean_error = results_df.groupby("currency").apply(lambda x: np.mean(x["predicted_change"] - x["actual_change"]))

print("==== Backtest Results ====")
print("Directional Accuracy (%) per currency:")
print(directional_accuracy)
print("\nMean Error per currency:")
print(mean_error)

# Optional: save results
results_df.to_csv("backtest_results.csv", index=False)
print("\n[Saved backtest results] backtest_results.csv")
