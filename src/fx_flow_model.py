from datetime import datetime, timedelta
import pandas as pd
from fx_utils import CURRENCIES, fetch_rates_yahoo, build_country_graph, compute_flows, calibrate, simulate_step, draw_flow

if __name__ == "__main__":
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=60)
    end_date = today

    rates = fetch_rates_yahoo(base="USD", start_date=start_date, end_date=end_date)
    flows = compute_flows(rates)
    if flows.empty:
        raise RuntimeError("Not enough FX data to compute flows")

    G = build_country_graph()
    nu, gamma, f, A, L = calibrate(flows, G)
    print("NAVIER SYSTEM:")
    print("nu:", nu, " gamma:", gamma, " f:", f)

    last = flows.iloc[-1].values
    pred = simulate_step(last, A, L, nu, gamma, f*np.ones(len(CURRENCIES)))
    print("\nPREDICTION:")
    print(pd.DataFrame({
        "currency": CURRENCIES,
        "flow_today": last,
        "flow_pred": pred,
        "pred_%": (pred - 1) * 100
    }))

    draw_flow(G, last)
