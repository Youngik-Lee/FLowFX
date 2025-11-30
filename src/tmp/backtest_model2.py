import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION AND MOCK DATA
# ==============================================================================
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

def fetch_rates_yahoo(base, start_date, end_date):
    """Generates synthetic FX rate data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    num_days = len(dates)
    np.random.seed(42)
    
    data = {"USD": np.ones(num_days)}
    for i, cur in enumerate(CURRENCIES):
        if cur == "USD":
            continue
        start_val = 1.0 + i * 0.1
        noise = np.random.randn(num_days) * 0.005
        data[cur] = start_val + np.cumsum(noise)
    
    return pd.DataFrame(data, index=dates)

# Mock feature and model functions
def add_timeseries_features(df): return df.copy()
def compute_alpha_signals(df): 
    return pd.DataFrame(np.random.rand(len(df), len(CURRENCIES)), index=df.index, columns=CURRENCIES)
def apply_slippage(pred, volume): return pred * 0.999
def build_country_graph(): return np.eye(len(CURRENCIES))
def simulate_step(K_last, A, L, nu, gamma, f): 
    np.random.seed(42)
    dK = np.zeros(len(CURRENCIES))
    dK[1:] = np.random.randn(len(CURRENCIES)-1) * 0.001
    return dK
def calibrate_navier(K_last, combined_target, G):
    return 0.5, 0.5, 1.0, np.eye(8), np.eye(8)

# ==============================================================================
# MODEL TRAINING
# ==============================================================================
def run_linear_regression_multi(X_df, y_df):
    X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values
    y = y_df.values
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    return model

def train_ml_model_multi(X_df, y_df):
    X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values
    y = y_df.values
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
    model.fit(X, y)
    return model

def predict_with_confidence(model, X):
    if isinstance(X, pd.Series):
        X_input = X.values.reshape(1, -1)
    elif isinstance(X, pd.DataFrame):
        X_input = X.values
    else:
        X_input = np.array(X).reshape(1, -1)
    preds = np.array([est.predict(X_input) for est in model.estimators_])
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred[0], std_pred[0]

# ==============================================================================
# BACKTESTING
# ==============================================================================
def run_backtest(rates: pd.DataFrame, train_window_days: int, test_window_days: int):
    rates = rates.ffill().bfill()
    rates_dates = rates.index
    
    results = {'Date': [], 'Currency': [], 'Actual_Rate': [], 'Predicted_Rate': [],
               'Predicted_dK_dt': [], 'Reliability': []}
    
    print(f"Starting backtest for {len(rates)} days...")
    min_idx = train_window_days
    
    for i in range(min_idx, len(rates) - test_window_days):
        train_end_date = rates_dates[i]
        train_start_date = rates_dates[i] - pd.Timedelta(days=train_window_days)
        predict_date = rates_dates[i + test_window_days]

        rates_train = rates.loc[train_start_date:train_end_date].copy()
        K_actual_next = rates.loc[predict_date].values

        print(f"Training until {train_end_date.date()} | Predicting {predict_date.date()}")

        K_matrix = rates_train[CURRENCIES].copy()
        dK_dt = K_matrix.diff().fillna(0)
        K_features = K_matrix.shift(1).dropna()
        dK_targets_non_usd = dK_dt.iloc[1:].drop(columns=["USD"], errors='ignore')

        if len(K_features) < 10:
            continue

        rates_ts = add_timeseries_features(K_matrix)
        alpha_df = compute_alpha_signals(rates_ts)

        lin_model = run_linear_regression_multi(K_features, dK_targets_non_usd)
        ml_model = train_ml_model_multi(K_features, dK_targets_non_usd)

        K_last_2_rows = K_matrix.iloc[-2:].copy()
        X_features_last = K_last_2_rows.drop(columns=["USD"]).pct_change().fillna(0)
        X_last_row_df = X_features_last.iloc[-1].to_frame().T

        ml_mean, ml_std = predict_with_confidence(ml_model, X_last_row_df)
        reg_pred = lin_model.predict(X_last_row_df.values)[0]
        alpha_pred_non_usd = alpha_df.iloc[-1].values[1:]

        combined_target = 0.5 * ml_mean + 0.3 * reg_pred + 0.2 * alpha_pred_non_usd
        combined_target = apply_slippage(combined_target, volume=5_000_000)

        combined_target_full = np.insert(combined_target, CURRENCIES.index("USD"), 0)
        K_last_full = K_matrix.iloc[-1].values
        G = build_country_graph()

        nu, gamma, f, A, L = calibrate_navier(K_last_full, combined_target_full, G)
        dK_pred = simulate_step(K_last_full, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
        K_next_pred = K_last_full + dK_pred

        reliability = max(0, 1 - ml_std.mean())
        for j, cur in enumerate(CURRENCIES):
            results['Date'].append(predict_date)
            results['Currency'].append(cur)
            results['Actual_Rate'].append(K_actual_next[j])
            results['Predicted_Rate'].append(K_next_pred[j])
            results['Predicted_dK_dt'].append(dK_pred[j])
            results['Reliability'].append(reliability)

    return pd.DataFrame(results)

# ==============================================================================
# METRICS
# ==============================================================================
def calculate_metrics(results_df: pd.DataFrame):
    metrics = {}
    results_df['Actual_dK_dt'] = results_df['Actual_Rate'].diff().fillna(0)
    non_usd_results = results_df[results_df['Currency'] != 'USD'].copy()

    non_usd_results['Actual_Direction'] = np.sign(non_usd_results['Actual_dK_dt'])
    non_usd_results['Predicted_Direction'] = np.sign(non_usd_results['Predicted_dK_dt'])

    correct_predictions = (non_usd_results['Actual_Direction'] == non_usd_results['Predicted_Direction'])
    metrics['Directional Accuracy (%)'] = correct_predictions.mean() * 100

    non_usd_results['Profit_Loss'] = non_usd_results['Predicted_Direction'] * non_usd_results['Actual_dK_dt']
    metrics['Total P&L (Unit Trade)'] = non_usd_results['Profit_Loss'].sum()
    metrics['Mean P&L per Trade'] = non_usd_results['Profit_Loss'].mean()

    daily_returns = non_usd_results.groupby('Date')['Profit_Loss'].sum()
    metrics['Sharpe Ratio (Approx)'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    return metrics, non_usd_results

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_and_save_backtest_results(results_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for currency in CURRENCIES:
        cur_data = results_df[results_df['Currency'] == currency].set_index('Date')
        plt.figure(figsize=(12,6))
        plt.plot(cur_data.index, cur_data['Actual_dK_dt'], label='Actual dK/dt', alpha=0.7)
        plt.plot(cur_data.index, cur_data['Predicted_dK_dt'], label='Predicted dK/dt', linestyle='--')
        plt.title(f'Actual vs. Predicted Rate Changes for {currency}')
        plt.xlabel('Date')
        plt.ylabel('Rate Change (dK/dt)')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f"{currency}_backtest_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot → {plot_path}")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    total_days = 252 + 60
    start_date = pd.Timestamp.today() - pd.Timedelta(days=total_days*7/5)
    end_date = pd.Timestamp.today()
    TRAIN_WINDOW = 60
    TEST_WINDOW = 1

    historical_rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    backtest_results = run_backtest(historical_rates, TRAIN_WINDOW, TEST_WINDOW)

    if not backtest_results.empty:
        metrics, full_results = calculate_metrics(backtest_results)

        print("\n" + "="*40)
        print("         BACKTEST PERFORMANCE SUMMARY")
        print("="*40)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("="*40)

        # Save metrics
        output_dir = "output/backtest"
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "backtest_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("========================================\n")
            f.write("        BACKTEST PERFORMANCE SUMMARY\n")
            f.write("========================================\n")
            for name, value in metrics.items():
                f.write(f"{name}: {value:.6f}\n")
            f.write("========================================\n")
        print(f"Saved metrics → {metrics_path}")

        # Save plots for all currencies
        plot_dir = os.path.join(output_dir, "plots")
        plot_and_save_backtest_results(full_results, plot_dir)
    else:
        print("Backtest failed to generate results.")
