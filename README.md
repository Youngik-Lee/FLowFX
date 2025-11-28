# FlowFX

FlowFX is a network-based FX prediction and visualization project. It models currency flows as `rate_today / rate_yesterday`, constructs a country/currency network, applies a Navier–Stokes-inspired dynamics on that network, and produces 1-day flow predictions.

## Features

- Fetch live FX rates from exchangerate.host
- Compute per-currency flow index (today / yesterday)
- Build a country/currency network and compute directed flows (lower→higher)
- Navier–Stokes inspired model with viscosity (`nu`) and slippage/friction (`gamma`) calibration
- 1-day prediction for each currency's flow
- Static and animated visualizations (arrow thickness = flow magnitude)
- Dockerfile for easy deployment

## Quick start

1. Clone the repo:

```bash
git clone https://github.com/Youngik-Lee/FLowFX.git
cd FLowFX
```

## 📁 Repository Structure

```
FlowFX/
│── README.md
│── requirements.txt
│── Dockerfile
│── src/
│    ├── fx_flow_model.py
│    ├── fx_flow_api.py
│    ├── fx_flow_animation.py
│── data/
│── output/
│── figures/
```


## 🚀 How It Works

### 1) Compute flow index  
```
flow[c] = today_rate[c] / yesterday_rate[c]
```

### 2) Flow direction rule  
```
Arrow from lower flow → higher flow  
Magnitude = |flow_high - flow_low|
```

### 3) Navier–Stokes style prediction  
```
u_next = u - (u * (A_norm @ u)) + nu * L u - gamma u + forcing
```

### 4) Animate result  
Animated country flow changes over time.


## 📦 Installation

```
pip install -r requirements.txt
```

## ▶ Run Model

```
python3 src/fx_flow_model.py
```

## ▶ Run Animation

```
python3 src/fx_flow_animation.py
```
