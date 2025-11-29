import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt

# ==============================================================================
# MOCK IMPORTS AND CONFIGURATION (Replace with actual imports in a real system)
# ==============================================================================
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

def fetch_rates_yahoo(base, start_date, end_date):
    """MOCK: Generates synthetic FX rate data."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    num_days = len(dates)
    
    # Create synthetic rate paths (starting around 1.1 for EUR, 1.3 for GBP, etc.)
    data = {}
    np.random.seed(42)
    
    # USD is always 1
    data["USD"] = np.ones(num_days)
    
    # Other currencies: start value + cumulative noise
    for i, cur in enumerate(CURRENCIES):
        if cur == "USD": continue
        start_val = 1.0 + (i * 0.1)
        noise = np.random.randn(num_days) * 0.005
        data[cur] = start_val + np.cumsum(noise)

    df = pd.DataFrame(data, index=dates)
    return df

def add_timeseries_features(df): return df.copy()
def compute_covariance(df): return df.iloc[-1:].T
def compute_correlation(df): return df.iloc[-1:].T
def compute_alpha_signals(df): 
    # Return 8 columns, one for each currency, based on latest features
    return pd.DataFrame(np.random.rand(len(df), len(CURRENCIES)), index=df.index, columns=CURRENCIES)
def apply_slippage(pred, volume): return pred * 0.999 # Simple mock slippage
def build_country_graph(): return np.eye(len(CURRENCIES)) # Mock adjacency matrix
def calibrate(K_df, G): return 0, 0, 0, np.eye(8), np.eye(8) # Mock A, L
def simulate_step(K_last, A, L, nu, gamma, f): 
    # Mock simulation: just return a small random change
    np.random.seed(42)
    dK = np.zeros(len(CURRENCIES))
    # Predict non-USD changes
    dK[1:] = np.random.randn(len(CURRENCIES)-1) * 0.001
    return dK

# ==============================================================================
# CORE MODEL FUNCTIONS (Copied from model.py)
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
    # Note: Setting n_estimators lower for faster mock runs
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42)) 
    model.fit(X, y)
    return model

def predict_with_confidence(model, X):
    """ML prediction with mean and std across trees for confidence (multi-output)."""
    # X must be a 2D array or DataFrame/Series converted to (1, n_features)
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

def calibrate_navier(K_last, combined_target, G):
    """Navier-Stokes calibration using minimization (Mocked implementation)."""
    # Mocked to return fixed parameters for demonstration
    nu_opt, gamma_opt, f_opt = 0.5, 0.5, 1.0 
    A, L = np.eye(8), np.eye(8)
    # In a real system, the minimize function would run here:
    # result = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
    return nu_opt, gamma_opt, f_opt, A, L

# ==============================================================================
# BACKTESTING DRIVER
# ==============================================================================

def run_backtest(rates: pd.DataFrame, train_window_days: int, test_window_days: int):
    """
    Performs a rolling-window backtest of the multi-model forecasting system.

    :param rates: Historical FX rate DataFrame.
    :param train_window_days: Size of the training window (e.g., 60 days).
    :param test_window_days: Size of the prediction step (typically 1 day).
    :return: DataFrame of predictions vs. actuals.
    """
    
    # Ensure data starts with a full week to calculate features properly
    rates = rates.ffill().bfill() 
    rates_dates = rates.index
    
    # Store results
    results = {
        'Date': [],
        'Currency': [],
        'Actual_Rate': [],
        'Predicted_Rate': [],
        'Predicted_dK_dt': [],
        'Reliability': [],
    }

    print(f"Starting backtest with {len(rates)} total days...")
    
    # Calculate the minimum required index for training
    min_idx = train_window_days 
    
    # The backtest rolls from the end of the training window to the end of data
    for i in range(min_idx, len(rates) - test_window_days):
        
        # --- 1. Define Windows ---
        train_end_date = rates_dates[i]
        train_start_date = rates_dates[i] - pd.Timedelta(days=train_window_days)
        predict_date = rates_dates[i + test_window_days]

        # Use historical data for training/feature generation
        rates_train = rates.loc[train_start_date:train_end_date].copy()
        
        # Actual target rates for comparison
        K_actual_next = rates.loc[predict_date].values
        
        print(f"--- Training until: {train_end_date.date()} | Predicting: {predict_date.date()} ---")

        # --- 2. Data Preparation ---
        K_matrix = rates_train[CURRENCIES].copy()
        dK_dt = K_matrix.diff().fillna(0)
        K_features = K_matrix.shift(1).dropna()
        dK_targets_non_usd = dK_dt.iloc[1:].drop(columns=["USD"], errors='ignore')

        if len(K_features) < 10: # Safety check
            print(f"Skipping prediction due to insufficient training data.")
            continue
            
        # --- 3. Feature Generation ---
        rates_ts = add_timeseries_features(K_matrix)
        # cov = compute_covariance(K_matrix) # Not used directly in prediction
        # corr = compute_correlation(K_matrix) # Not used directly in prediction
        alpha_df = compute_alpha_signals(rates_ts)

        # --- 4. Model Training ---
        lin_model = run_linear_regression_multi(K_features, dK_targets_non_usd)
        ml_model = train_ml_model_multi(K_features, dK_targets_non_usd)

        # --- 5. Prediction (for the next step) ---
        K_last_2_rows = K_matrix.iloc[-2:].copy() 
        X_features_last = K_last_2_rows.drop(columns=["USD"]).pct_change().fillna(0)
        X_last_row_df = X_features_last.iloc[-1].to_frame().T 
        
        # ML Prediction
        ml_mean, ml_std = predict_with_confidence(ml_model, X_last_row_df)
        reg_pred = lin_model.predict(X_last_row_df.values)[0]
        
        # Alpha Signal
        alpha_pred_non_usd = alpha_df.iloc[-1].values[1:] # Drop USD alpha if present
        
        # Combined Target
        combined_target = 0.5 * ml_mean + 0.3 * reg_pred + 0.2 * alpha_pred_non_usd
        combined_target = apply_slippage(combined_target, volume=5_000_000)

        # --- 6. NS Simulation ---
        combined_target_full = np.insert(combined_target, CURRENCIES.index("USD"), 0)
        K_last_full = K_matrix.iloc[-1].values 
        G = build_country_graph()
        
        nu, gamma, f, A, L = calibrate_navier(K_last_full, combined_target_full, G)
        dK_pred = simulate_step(K_last_full, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
        K_next_pred = K_last_full + dK_pred
        
        # --- 7. Store Results ---
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
# PERFORMANCE METRICS AND PLOTTING
# ==============================================================================

def calculate_metrics(results_df: pd.DataFrame):
    """Calculates key trading metrics from the backtest results."""
    
    metrics = {}
    
    # Calculate Prediction Accuracy (Directional Accuracy)
    # Prediction is correct if predicted dK/dt and actual dK/dt have the same sign
    results_df['Actual_dK_dt'] = results_df['Actual_Rate'].diff()
    results_df['Actual_dK_dt'] = results_df['Actual_dK_dt'].fillna(0)

    # Calculate actual movement vs predicted movement direction (excluding USD)
    non_usd_results = results_df[results_df['Currency'] != 'USD'].copy()
    
    # Calculate the sign of the change
    non_usd_results['Actual_Direction'] = np.sign(non_usd_results['Actual_dK_dt'])
    non_usd_results['Predicted_Direction'] = np.sign(non_usd_results['Predicted_dK_dt'])

    # Directional Accuracy (DA)
    correct_predictions = (non_usd_results['Actual_Direction'] == non_usd_results['Predicted_Direction'])
    metrics['Directional Accuracy (%)'] = correct_predictions.mean() * 100
    
    # Calculate simple P&L for a unit trade based on predicted direction
    # If prediction is positive (buy), profit is (Actual_Rate_t+1 - Actual_Rate_t)
    # If prediction is negative (sell), profit is (Actual_Rate_t - Actual_Rate_t+1)
    non_usd_results['Profit_Loss'] = non_usd_results['Predicted_Direction'] * non_usd_results['Actual_dK_dt']
    metrics['Total P&L (Unit Trade)'] = non_usd_results['Profit_Loss'].sum()
    metrics['Mean P&L per Trade'] = non_usd_results['Profit_Loss'].mean()
    
    # Sharpe Ratio (Requires risk-free rate, which we'll ignore for simplicity)
    # Assumes no correlation between trades for a simple approximation
    daily_returns = non_usd_results.groupby('Date')['Profit_Loss'].sum()
    metrics['Sharpe Ratio (Approx)'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annulize by sqrt(252)
    
    return metrics, non_usd_results

def plot_backtest_results(results_df: pd.DataFrame, currency: str):
    """Plots the actual vs. predicted rate changes for a specific currency."""
    
    cur_data = results_df[results_df['Currency'] == currency].set_index('Date')
    
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Rate Change
    plt.plot(cur_data.index, cur_data['Actual_dK_dt'], label='Actual dK/dt', alpha=0.7)
    
    # Plot Predicted Rate Change
    plt.plot(cur_data.index, cur_data['Predicted_dK_dt'], label='Predicted dK/dt', linestyle='--')
    
    plt.title(f'Actual vs. Predicted Rate Changes for {currency}')
    plt.xlabel('Date')
    plt.ylabel('Rate Change (dK/dt)')
    plt.legend()
    plt.grid(True)
    plt.show() # In a real environment, this would save to a file/display inline


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # --- Configuration ---
    # Fetch 252 business days (approx 1 year) + 60 training days
    total_days = 252 + 60 
    start_date = pd.Timestamp.today() - pd.Timedelta(days=total_days * 7/5) 
    end_date = pd.Timestamp.today()
    TRAIN_WINDOW = 60 # days
    TEST_WINDOW = 1 # day

    # --- 1. Fetch All Data ---
    historical_rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)

    # --- 2. Run Backtest ---
    backtest_results = run_backtest(historical_rates, TRAIN_WINDOW, TEST_WINDOW)

    # --- 3. Calculate and Display Metrics ---
    if not backtest_results.empty:
        metrics, full_results = calculate_metrics(backtest_results)
        
        print("\n" + "="*40)
        print("         BACKTEST PERFORMANCE SUMMARY")
        print("="*40)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("="*40)
        
        # --- 4. Plot Example (Requires Matplotlib) ---
        # NOTE: In a text-based environment, this won't show the plot.
        # plot_backtest_results(full_results, 'EUR')
        # print("Plot generated for EUR (requires graphical environment to display).")
    else:
        print("Backtest failed to generate results.")
