import pandas as pd
import numpy as np

def add_timeseries_features(df):
    df = df.copy()
    for col in df.columns:
        df[f"{col}_ret"] = df[col].pct_change()
        df[f"{col}_ma5"] = df[col].rolling(5).mean()
        df[f"{col}_ma20"] = df[col].rolling(20).mean()
        df[f"{col}_vol20"] = df[f"{col}_ret"].rolling(20).std()
    return df
