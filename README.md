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
git clone https://github.com/your-org/FlowFX.git
cd FlowFX
