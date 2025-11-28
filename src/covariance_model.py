import pandas as pd

def compute_covariance(df):
    return df.pct_change().cov()

def compute_correlation(df):
    return df.pct_change().corr()
