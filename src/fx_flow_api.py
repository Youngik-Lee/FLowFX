import requests
import pandas as pd

def fetch_rates_real(currencies, base="USD", days=12):
    url = "https://api.exchangerate.host/timeseries"

    params = {
        "base": base,
        "start_date": pd.Timestamp.today() - pd.Timedelta(days=days),
        "end_date": pd.Timestamp.today(),
    }

    response = requests.get(url, params=params)
    data = response.json()["rates"]

    df = pd.DataFrame(data).T
    df = df[currencies]
    df.index = pd.to_datetime(df.index)
    return df
