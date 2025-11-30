# FLowFX: Network-Based Currency Flow Prediction

FLowFX is a network dynamics project that models currency movements as flows, constructs a currency network, applies a **Navier–Stokes-inspired network dynamics** engine to predict 1-day flow changes, and visualizes the results.

The core approach is to model the change in currency rates (relative to USD) as a fluid velocity vector **u**, subject to momentum transfer, diffusion, and external market forces.



## 🧠 Architecture

The prediction system is a hybrid model where **Machine Learning (ML)** provides the external force input to the **Navier-Stokes (NS) primary engine**.

```
FX Data (Yahoo)
      ↓
Time Series Stats (vol, MA)
      ↓
Covariance Matrix
      ↓
Alpha Factors  → modifies forcing term
      ↓
Regression Model → auxiliary predictor
      ↓
ML Model → auxiliary predictor
      ↓
Navier–Stokes Simulation (primary engine)
      ↓
Slippage Model  → adjusts final output
      ↓
Flow Visualization (network arrows)
```

**Data Flow**
* **FX Data (Yahoo Finance)** -> **Time Series Features/Stats**
* **Time Series Features** feeds:
    * **Covariance Matrix**
    * **Alpha Factors**
    * **Regression Model**
    * **ML Model**
* **Alpha, Regression, and ML Models** combine into the **Combined External Forcing (f)**
* **Combined External Forcing (f)** -> **Navier-Stokes Network Engine** (Primary Model)
* **Navier-Stokes Network Engine** -> **Slippage Model**
* **Slippage Model** -> **Final 1-Day Flow Prediction**
* **Final Prediction** -> **Flow Visualization**

The **Combined External Forcing** (f) is a **weighted average** of the auxiliary models, injected into the NS simulation:

**f = 0.5 * ML_pred + 0.3 * Reg_pred + 0.2 * alpha_pred**



## ✨ Features

* **Data Source:** Fetches live FX rates (e.g., USD/EUR, USD/JPY) from **Yahoo Finance**.
* **Network Dynamics:** Models currency flows using a **Navier–Stokes (NS) equation analogy** on a country/currency network.
* **NS Calibration:** Calibrates two key parameters: viscosity (`nu`) and slippage/friction (`gamma`) calibration using optimization (scipy.optimize.minimize).
* **Hybrid Prediction:** Uses **Random Forest (ML)**, **Linear Regression**, and proprietary **Alpha Signals** to compute the external forcing term for the NS model.
* **Output:** Provides 1-day flow prediction for each currency pair.
* **Visualization:** Generates static and animated visualizations of the flow network (arrow thickness = flow magnitude).


## 🚀 How It Works

### 1) Compute Flow Velocity (u)
The currency flow velocity u[c] is the relative change in the FX rate for currency c (relative to the base currency, USD):

**u[c] = (rate_today[c] / rate_yesterday[c]) - 1**

### 2) Network and Dynamics

The network adjacency matrix is constructed based on the **currency correlation matrix**. The next-day flow velocity (u_next) is calculated by solving the discretized NS-like equation:

**u_next = u - (u * (A_norm @ u)) + nu * L @ u - gamma * u + f**

* **u * (A_norm @ u)**: Convection (momentum transfer)
* **nu * L @ u**: Diffusion/Viscosity (nu is viscosity, L is the Laplacian)
* **gamma * u**: Friction/Damping (gamma is friction coefficient)
* **f**: External Forcing (input from the hybrid ML/Alpha model)



## 📁 Repository Structure
```
FLowFX/
│── README.md
│── requirements.txt
│── Dockerfile
│── src/
│    ├── fx_flow_model.py         # Main execution script
│    ├── fx_flow_animation.py     # Animation script
│    ├── fx_utils.py              # CORE: Contains build_country_graph, calibrate, simulate_step
│    ├── backtest_fxmodel.py
│    ├── alpha_model.py
│    ├── covariance_model.py
│    ├── regression_model.py
│    ├── ml_model.py
│    ├── slippage.py
│    ├── timeseries_tools.py
│── data/
│    ├── sample_fx_data.csv      
│── output/
│    ├── animation
│    ├── model
│    ├── backtest
```

## 📦 Installation & Run
### 1. Clone the repo:
```bash
git clone https://github.com/Youngik-Lee/FLowFX.git
cd FLowFX
```

### 2. Installation
```
pip install -r requirements.txt
```

### 3. Run Model (Prediction)
```
python3 src/fx_flow_model.py
```
Result
```
==== NAVIER SYSTEM PARAMETERS ====
nu: 0.133591
gamma: 0.598976
f: 0.000000

==== ML + Regression + Alpha (Non-USD Currencies) ====
ML Predicted dK/dt: -0.00035200150399136227
Regression contribution: [-1.42551479e-04 -7.09626830e-06 -2.41911307e-06 -1.87805586e-03
 -2.15161472e-04 -5.17546575e-06 -2.03828937e-03]
Alpha contribution: [1.00000000e+00 1.15998518e+00 6.39860798e-03 6.84275350e-04
 1.32411754e+00 7.71176517e-01 1.28556192e-01]
After slippage: -0.5005520015039914

==== CURRENCY PREDICTIONS ====
USD: K_today=1.000000, dK/dt_pred=0.218068, reliability=0.999
EUR: K_today=1.159985, dK/dt_pred=-0.282384, reliability=0.999
JPY: K_today=0.006399, dK/dt_pred=-0.288125, reliability=0.999
KRW: K_today=0.000684, dK/dt_pred=-0.151330, reliability=0.999
GBP: K_today=1.324118, dK/dt_pred=-0.833808, reliability=0.999
SGD: K_today=0.771177, dK/dt_pred=-0.077864, reliability=0.999
HKD: K_today=0.128556, dK/dt_pred=-0.209557, reliability=0.999
AUD: K_today=0.653580, dK/dt_pred=-0.378690, reliability=0.999
```

### 4. Run Animation
```
python3 src/fx_flow_animation.py
```
Result

### 5. Backtest
```
python3 src/backtest_fxmodel.py
```
Result
```
========================================
        BACKTEST PERFORMANCE SUMMARY
========================================
Currency: EUR
Directional Accuracy (%): 50.000000
Total P&L (Unit Trade): 113850.712776
Mean P&L per Trade: 580.870984
Sharpe Ratio (Approx): 1.564125
========================================
```

6. Run all process
```
chmod +x run_all.sh
./run_all.sh
``` 
