import pandas as pd
import numpy as np
from model import (
    run_linear_regression_multi,
    train_ml_model_multi,
    predict_with_confidence,
    apply_slippage,
    calibrate_navier,
    CURRENCIES,
    fetch_rates_yahoo,
    add_timeseries_features,
    compute_covariance,
    compute_correlation,
    compute_alpha_signals,
    build_country_graph,
    simulate_step
)

def backtest(start_date, end_date):
    # Fetch data
    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    K = rates[CURRENCIES].copy()

    # Backtest storage
    preds = []
    reals = []
    dates = []

    G = build_country_graph()

    # Walk-forward loop
    for i in range(20, len(K)-1):
        K_train = K.iloc[:i]
        K_today = K.iloc[i]
        K_next_real = K.iloc[i+1]

        # --- Features / Targets ---
        K_features = K_train.shift(1).dropna()
        dK_dt = K_train.diff().fillna(0)
        dK_targets_non_usd = dK_dt.iloc[1:].drop(columns=["USD"], errors="ignore")

        # --- Models ---
        lin_model, _, _ = run_linear_regression_multi(K_features, dK_targets_non_usd)
        ml_model = train_ml_model_multi(K_features, dK_targets_non_usd)

        # --- Predict ---
        X_t = K_features.drop(columns=["USD"]).pct_change().fillna(0).iloc[-1]

        ml_mean, ml_std = predict_with_confidence(ml_model, X_t)
        reg_pred = lin_model.predict(X_t.values.reshape(1, -1))[0]

        # Alpha
        rates_ts = add_timeseries_features(K_train)
        alpha_df = compute_alpha_signals(rates_ts)
        alpha_pred = alpha_df.iloc[-1].values[:7]

        # Weighted combination
        combined = 0.5*ml_mean + 0.3*reg_pred.mean() + 0.2*alpha_pred
        combined = apply_slippage(combined, volume=5_000_000)

        # Pad USD=0
        combined_full = np.insert(combined, CURRENCIES.index("USD"), 0)

        # Navier-Stokes step
        nu, gamma, f, A, L = calibrate_navier(K_today.values, combined_full, G)
        dK_pred = simulate_step(K_today.values, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
        K_pred = K_today.values + dK_pred

        preds.append(K_pred)
        reals.append(K_next_real.values)
        dates.append(K.index[i+1])

    # Results DataFrame
    preds = pd.DataFrame(preds, index=dates, columns=CURRENCIES)
    reals = pd.DataFrame(reals, index=dates, columns=CURRENCIES)

    return preds, reals
