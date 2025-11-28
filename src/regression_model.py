from sklearn.linear_model import LinearRegression

def run_linear_regression(df, target="USD"):
    X = df.drop(columns=[target]).pct_change().fillna(0)
    y = df[target].pct_change().shift(-1).fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    return model, model.coef_, model.intercept_
