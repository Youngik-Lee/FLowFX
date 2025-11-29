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
```

## 📦 Installation & Run
1. Clone the repo:
```bash
git clone https://github.com/Youngik-Lee/FLowFX.git
cd FLowFX
```

2. Installation
```
pip install -r requirements.txt
```

3. Run Model (Prediction)
```
python3 src/fx_flow_model.py
```

4. Run Animation
```
python3 src/fx_flow_animation.py
```
