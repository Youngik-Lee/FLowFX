import pandas as pd
import numpy as np
import random
from typing import Tuple, Dict, List

# -----------------------------
# ML / Regression imports
# -----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# SciPy for optimization in the NS model
from scipy.optimize import minimize

# -----------------------------
# CONSTANTS & MOCK UTILITIES (To make the code runnable)
# -----------------------------

# List of currencies used in the original model
CURRENCIES = ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'JPY', 'USD']
NON_USD_CURRENCIES = CURRENCIES[:-1]
DAYS_IN_YEAR = 252

def fetch_rates_yahoo(base, start_date, end_date):
    """MOCK: Generates simulated FX data (OHLCV structure is not needed for K_matrix)."""
    days = (end_date - start_date).days
    np.random.seed(42)
    
    data = {}
    for i, cur in enumerate(CURRENCIES):
        # Generate slightly different random walks for each currency
        start_price = 1.0 + (i * 0.1)
        price_changes = np.random.normal(0, 0.001, days)
        prices = start_price + np.cumsum(price_changes)
        data[cur] = prices

    return pd.DataFrame(data, index=pd.to_datetime(pd.date_range(start=start_date, periods=days, freq='D')))

# --- Mocks for unprovided feature engineering functions ---
def add_timeseries_features(K_matrix: pd.DataFrame) -> pd.DataFrame:
    """MOCK: Adds mock features like RSI, momentum, etc."""
    return K_matrix.shift(1).fillna(K_matrix.iloc[0]).add_suffix("_MOM") 

def compute_covariance(K_matrix: pd.DataFrame) -> pd.DataFrame:
    """MOCK: Returns a mock covariance matrix."""
    return K_matrix.cov()

def compute_correlation(K_matrix: pd.DataFrame) -> pd.DataFrame:
    """MOCK: Returns a mock correlation matrix."""
    return K_matrix.corr()

def compute_alpha_signals(rates_ts: pd.DataFrame) -> pd.DataFrame:
    """MOCK: Returns mock alpha signals for non-USD currencies."""
    df = pd.DataFrame(np.random.uniform(-0.001, 0.001, (len(rates_ts), len(NON_USD_CURRENCIES))), 
                      index=rates_ts.index, columns=[f"{c}_Alpha" for c in NON_USD_CURRENCIES])
    return df

def apply_slippage(dK_pred: np.ndarray, volume: int) -> np.ndarray:
    """MOCK: Applies a small slippage penalty to the prediction."""
    penalty = 0.05 # 5% penalty on the magnitude
    return dK_pred * (1 - penalty * np.sign(dK_pred))

# --- Mocks for Navier-Stokes (NS) related functions ---
def build_country_graph():
    """MOCK: Returns a placeholder for the country graph structure."""
    return {'nodes': CURRENCIES, 'edges': {}}

def calibrate(K_df: pd.DataFrame, G: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MOCK: Simulates the calibration step for NS, returning placeholders."""
    # Assuming the original NS setup returns: [V_mat, W_mat, E_mat, A_mat, L_mat]
    n = len(CURRENCIES)
    V, W, E, A, L = [np.random.rand(n, n) for _ in range(5)]
    return V, W, E, A, L

def simulate_step(K_last: np.ndarray, A: np.ndarray, L: np.ndarray, nu: float, gamma: float, f: np.ndarray) -> np.ndarray:
    """MOCK: Simulates the dK/dt prediction step using NS parameters."""
    # Simple linear combination of inputs to simulate a step prediction
    dK_pred = nu * np.dot(A, K_last) + gamma * np.dot(L, f)
    # Ensure the shape is (8,)
    return dK_pred

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, annualization_factor: int = DAYS_IN_YEAR) -> float:
    """Calculates the Annualized Sharpe Ratio."""
    daily_risk_free_rate = risk_free_rate / annualization_factor
    excess_returns = returns - daily_risk_free_rate
    
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0:
        return 0.0
        
    sharpe_ratio = (excess_returns.mean() / std_dev_excess_return) * np.sqrt(annualization_factor)
    
    return sharpe_ratio

# -----------------------------
# QUANT MODEL CLASS
# -----------------------------

class QuantModel:
    """
    Encapsulates the complex modeling logic from the user's model.py, 
    allowing for training and prediction steps.
    """
    def __init__(self):
        self.lin_model = None
        self.ml_model = None
        self.G = build_country_graph()

    def run_linear_regression_multi(self, X_df, y_df):
        """Trains Multi-Output Linear Regression."""
        X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values 
        y = y_df.values
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X, y)
        return model

    def train_ml_model_multi(self, X_df, y_df):
        """Trains Multi-Output Random Forest Regressor."""
        X = X_df.drop(columns=["USD"]).pct_change().fillna(0).values 
        y = y_df.values
        # Reduced estimators for speed in backtest simulation
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)) 
        model.fit(X, y)
        return model

    def predict_with_confidence(self, model, X):
        """ML prediction with mean and std across trees."""
        if isinstance(X, pd.Series):
            X_input = X.values.reshape(1, -1)
        elif isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = np.array(X).reshape(1, -1)
            
        preds = np.array([est.predict(X_input) for est in model.estimators_])
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        return mean_pred[0], std_pred[0] # Return 1D arrays for prediction/std

    def calibrate_navier(self, K_last: np.ndarray, combined_target: np.ndarray):
        """Calibrates Navier-Stokes parameters."""
        
        # Need to ensure the calibration step receives the correct shapes (8 currencies)
        
        # 1. NS internal calibration (A and L matrices)
        K_df = pd.DataFrame([K_last], columns=CURRENCIES)
        _, _, _, A, L = calibrate(K_df, self.G)

        def loss(params):
            nu, gamma, f_scalar = params
            f = f_scalar * np.ones(len(K_last))
            dK_pred = simulate_step(K_last, A, L, nu, gamma, f)
            # Loss is MSE between NS prediction and the combined target (size 8)
            return np.sum((dK_pred - combined_target)**2)

        init = [0.1, 0.1, 0.5]
        bounds = [(0, 1), (0, 1), (0, 2)]
        result = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
        
        nu_opt, gamma_opt, f_opt = result.x
        return nu_opt, gamma_opt, f_opt, A, L

    def train(self, K_matrix_train: pd.DataFrame):
        """
        Train all models (ML, Regression) on the training window data.
        """
        K_features = K_matrix_train.shift(1).dropna()
        dK_dt = K_matrix_train.diff().dropna()
        dK_targets_non_usd = dK_dt.drop(columns=["USD"], errors='ignore')

        # 1. Train Regression and ML
        self.lin_model = self.run_linear_regression_multi(K_features, dK_targets_non_usd)
        self.ml_model = self.train_ml_model_multi(K_features, dK_targets_non_usd)

    def predict_next(self, K_matrix_day_T: pd.DataFrame, alpha_df_day_T: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Predict dK/dt for day T+1 using the latest data from day T.
        
        Args:
            K_matrix_day_T: Last two rows of K_matrix (T-1 and T).
            alpha_df_day_T: Last row of Alpha signals (T).
            
        Returns:
            Tuple of predicted dK/dt (size 8) and predicted reliability (1 - mean_std).
        """
        
        # 1. Prepare Features for Prediction (Uses Pct Change between T-1 and T)
        K_last_2_rows = K_matrix_day_T.iloc[-2:].copy() 
        X_features_last = K_last_2_rows.drop(columns=["USD"]).pct_change().fillna(0)
        X_last_row_df = X_features_last.iloc[-1].to_frame().T 

        # 2. Predict with ML and Regression
        ml_mean, ml_std = self.predict_with_confidence(self.ml_model, X_last_row_df)
        reg_pred = self.lin_model.predict(X_last_row_df.values)[0] 
        
        # 3. Get Alpha Signals (Only non-USD, size 7)
        alpha_pred_non_usd = alpha_df_day_T.iloc[-1].values 
        
        # 4. Combine Signals (size 7)
        combined_target_non_usd = 0.5*ml_mean + 0.3*reg_pred + 0.2*alpha_pred_non_usd
        
        # 5. Apply Slippage to the combined target
        slippage_applied_target_non_usd = apply_slippage(combined_target_non_usd, volume=5_000_000)

        # 6. Prepare for NS: Pad target (size 7) back to full 8 currencies
        combined_target_full = np.insert(slippage_applied_target_non_usd, CURRENCIES.index("USD"), 0)
        K_last_full = K_matrix_day_T.iloc[-1].values # K at day T (size 8)
        
        # 7. Navier-Stokes Calibration and Prediction
        nu, gamma, f, A, L = self.calibrate_navier(K_last_full, combined_target_full)
        dK_pred_full = simulate_step(K_last_full, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
        
        # Reliability is inversely proportional to ML prediction uncertainty
        reliability = max(0, 1 - ml_std.mean())
        
        return dK_pred_full, reliability


# -----------------------------
# WALK-FORWARD BACKTESTING FRAMEWORK
# -----------------------------

def walk_forward_backtest(K_matrix_full: pd.DataFrame, 
                          train_window_size: int = DAYS_IN_YEAR * 3, # 3 years for training
                          test_window_size: int = 1,                 # 1 day for testing
                          step_size: int = 1,                        # Roll forward 1 day at a time
                         ) -> Dict[str, float]:
    """
    Performs a walk-forward analysis on the complex quantitative model.
    """
    
    print("--- Starting Walk-Forward Backtest (ML + NS) ---")
    
    returns_list = []
    
    # Start loop after the initial training window is available
    i = train_window_size
    while i < len(K_matrix_full) - test_window_size:
        
        # 1. Define Windows
        train_start_idx = i - train_window_size
        train_end_idx = i - 1
        
        test_day_idx = i # The day we predict
        
        K_matrix_train = K_matrix_full.iloc[train_start_idx:train_end_idx + 1]
        
        # Data needed for prediction (day T-1, day T)
        # We need the last two bars of the train window to calculate the feature vector pct_change
        K_matrix_pred_data = K_matrix_full.iloc[train_end_idx - 1:train_end_idx + 1]
        
        K_actual_price_t_plus_1 = K_matrix_full.iloc[test_day_idx]
        
        # 2. Extract Features/Signals for the Prediction Step
        rates_ts_train = add_timeseries_features(K_matrix_train)
        alpha_df_t = compute_alpha_signals(K_matrix_full.iloc[train_end_idx - len(K_matrix_train):train_end_idx+1])
        alpha_df_day_T = alpha_df_t.iloc[[-1]] # Only the last day's alpha signal

        # 3. Model Training
        model = QuantModel()
        model.train(K_matrix_train)
        
        # 4. Out-of-Sample Prediction
        dK_pred_full, reliability = model.predict_next(K_matrix_pred_data, alpha_df_day_T)
        
        # 5. Trading Signal Generation & P&L Calculation
        
        # The predicted dK/dt for the non-USD currencies (dK_dt is size 8, USD is 0)
        dK_pred_non_usd = np.delete(dK_pred_full, CURRENCIES.index("USD"))
        
        # Trade 1 unit for every predicted basis point move, scaled by reliability
        # Signal: 1 if prediction is positive, -1 if negative, 0 if near zero.
        signal = np.sign(dK_pred_non_usd) * reliability 
        
        # Actual price change over the predicted period (T to T+1)
        dK_actual = K_actual_price_t_plus_1[NON_USD_CURRENCIES] - K_matrix_pred_data.iloc[-1][NON_USD_CURRENCIES]
        
        # Daily Return = Sum of (Signal * Actual Price Change)
        # This is a simplified P&L calculation
        daily_return = (signal * dK_actual.values).sum()
        
        returns_list.append(daily_return)
        
        print(f"Day {i - train_window_size}: Train End: {K_matrix_train.index[-1].date()} | Test Day: {K_matrix_full.index[test_day_idx].date()} | Daily Return: {daily_return:.6f}")
        
        i += step_size
        
    # 6. Final Reporting
    if not returns_list:
        return {"Error": "Insufficient data to run backtest."}
        
    returns_series = pd.Series(returns_list)
    final_sharpe = calculate_sharpe_ratio(returns_series)
    total_return = (1 + returns_series).prod() - 1
    
    return {
        "Total_Testing_Days": len(returns_list),
        "Annualized_Sharpe_Ratio": final_sharpe,
        "Total_Compounded_Return": total_return,
        "Average_Daily_Return": returns_series.mean(),
        "Max_Daily_Drawdown": returns_series.min(),
    }


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":
    
    print("Initializing FX Model Backtest...")
    
    # 1. Fetch/Generate Full Data
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=DAYS_IN_YEAR * 5) # 5 years of data
    end_date = today - pd.Timedelta(days=1) # End yesterday to ensure we have a full target bar
    K_matrix_full = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    
    print(f"Data range: {K_matrix_full.index[0].date()} to {K_matrix_full.index[-1].date()} ({len(K_matrix_full)} days)")
    print("Model relies on retraining (walk-forward optimization) every step. ")
    
    # 2. Run the Backtest
    # We use a large train window (3 years) and step 1 day at a time
    results = walk_forward_backtest(K_matrix_full)
    
    # 3. Print Summary
    print("\n==============================================")
    print("COMPLEX QUANT MODEL BACKTEST SUMMARY (MOCK RUN)")
    print("==============================================")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\nNote: This simulation uses mock data and mock helper functions.")
