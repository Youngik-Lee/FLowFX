import pandas as pd
import numpy as np
import os
from fx_flow_model import (
    predict_with_confidence, run_linear_regression_multi, train_ml_model_multi,
    apply_slippage, calibrate_navier, CURRENCIES, add_timeseries_features,
    compute_alpha_signals, build_country_graph, simulate_step, fetch_rates_yahoo
)
# -----------------------------
# Fetch historical FX rates
# -----------------------------
today = pd.Timestamp.today()
start_date = today - pd.Timedelta(days=365)  # last 1 year
rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=today)
K_matrix = rates[CURRENCIES].copy()
dK_actual = K_matrix.diff().fillna(0)

# -----------------------------
# Backtest parameters
# -----------------------------
window = 60
trade_volume = 1_000_000
results = []

# Ensure output folder exists
output_dir = "output/backtest"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Rolling window backtest
# -----------------------------
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
    slippage_pred = apply_slippage(ml_mean, volume=trade_volume)

    # --- regression & alpha ---
    reg_pred = lin_model.predict(X_last_row.values)[0]
    rates_ts = add_timeseries_features(K_window)
    alpha_pred_non_usd = compute_alpha_signals(rates_ts).iloc[-1].values[:7]

    combined_target = 0.5*ml_mean+0.3*reg_pred+0.2*alpha_pred_non_usd
    combined_target = apply_slippage(combined_target, volume=trade_volume)
    combined_target_full = np.insert(combined_target, CURRENCIES.index("USD"), 0)
    combined_target_full -= combined_target_full.mean()
    # --- Navier-Stokes simulation ---
    G = build_country_graph()
    nu, gamma, f, A, L = calibrate_navier(K_last.values, combined_target_full, G)
    # dK_pred = simulate_step(K_last.values, A, L, nu, gamma, f*np.ones(len(CURRENCIES))) // when there is a macro signals (eg. news) 
    dK_pred = simulate_step(K_last.values, A, L, nu, gamma, f*np.zeros(len(CURRENCIES)))

    # --- collect results ---
    for i, cur in enumerate(CURRENCIES):
        actual = K_next_actual[cur] - K_last[cur]
        pred = dK_pred[i]
        direction = np.sign(pred)
        pnl = direction * actual * trade_volume

        results.append({
            "date": K_matrix.index[t],
            "currency": cur,
            "actual_change": actual,
            "predicted_change": pred,
            "direction": direction,
            "direction_correct": direction == np.sign(actual),
            "pnl": pnl
        })

# -----------------------------
# Convert to DataFrame and compute metrics
# -----------------------------
results_df = pd.DataFrame(results)

directional_accuracy = results_df.groupby("currency")["direction_correct"].mean() * 100
mean_error = results_df.groupby("currency").apply(lambda x: np.mean(x["predicted_change"] - x["actual_change"]))
total_pnl = results_df.groupby("currency")["pnl"].sum()
mean_pnl_per_trade = results_df.groupby("currency")["pnl"].mean()
sharpe_ratio = results_df.groupby("currency")["pnl"].apply(lambda x: x.mean() / x.std() * np.sqrt(252))

# Save full backtest results CSV
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(os.path.join(output_dir, f"backtest_results_{timestamp}.csv"), index=False)

print("==== Backtest Summary ====")
print("\nDirectional Accuracy (%):")
print(directional_accuracy)
print("\nTotal P&L:")
print(total_pnl)
print("\nSharpe Ratio:")
print(sharpe_ratio)

# -----------------------------
# Save final day summary (like summary.txt)
# -----------------------------
last_row = K_matrix.iloc[-1]
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
reliability = max(0, 1 - ml_std.mean())
for cur, K_today, dK in zip(CURRENCIES, last_row.values, dK_pred):
    summary_lines.append(f"{cur}: K_today={K_today:.6f}, dK/dt_pred={dK:.6f}, reliability={reliability:.3f}")

# Create timestamp string
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# -----------------------------
# Save final day summary (like summary.txt) with timestamp
# -----------------------------
summary_filename = f"summary_{timestamp}.txt"
summary_path = os.path.join(output_dir, summary_filename)
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\n[Saved final summary] {summary_path}")

# ============================================================
# Save individual currency summaries with timestamp
# ============================================================
for cur in CURRENCIES:
    cur_dir_acc = directional_accuracy.get(cur, np.nan)
    cur_total_pnl = total_pnl.get(cur, np.nan)
    cur_mean_pnl = mean_pnl_per_trade.get(cur, np.nan)
    cur_sharpe = sharpe_ratio.get(cur, np.nan)

    summary_text = f"""
========================================
        BACKTEST PERFORMANCE SUMMARY
========================================
Currency: {cur}
Directional Accuracy (%): {cur_dir_acc:.6f}
Total P&L (Unit Trade): {cur_total_pnl:.6f}
Mean P&L per Trade: {cur_mean_pnl:.6f}
Sharpe Ratio (Approx): {cur_sharpe:.6f}
========================================
"""
    # Add timestamp to filename
    file_path = os.path.join(output_dir, f"{cur}_summary_{timestamp}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
