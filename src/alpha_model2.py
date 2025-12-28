import pandas for pd
import numpy for np

def compute_alpha_dK(df, window=20):
    """
    Returns alpha-based dK estimate (same unit as dK)
    """
    df = df.copy()
    # 1️⃣ returns
    ret = df.pct_change()
    # 2️⃣ momentum signal
    momentum = df.pct_change(5)
    # 3️⃣ volatility (scale)
    vol = ret.rolling(window).std()
    # 4️⃣ normalized trend signal
    trend_strength = momentum / (vol + 1e-9)
    # 5️⃣ direction only
    direction = np.sign(trend_strength)
    # 6️⃣ convert to dK scale
    dK_scale = df.diff().rolling(window).std()
    dK_alpha = direction * dK_scale
    return dK_alpha
