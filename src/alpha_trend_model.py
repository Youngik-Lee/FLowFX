import pandas as pd
import numpy as np

def compute_alpha_signals(df, currency="USD"):
    df = df.copy()
    ret = df[currency].pct_change()

    df["alpha_momentum"] = df[currency].pct_change(5)
    df["alpha_meanrev"] = -ret
    df["alpha_trend_voladj"] = df["alpha_momentum"] / (ret.rolling(20).std() + 1e-9)

    df["signal_long"] = (df["alpha_trend_voladj"] > 0).astype(int)
    df["signal_short"] = (df["alpha_trend_voladj"] < 0).astype(int)

    return df
