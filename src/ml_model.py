from sklearn.ensemble import RandomForestRegressor

def train_ml_model(df, target="USD"):
    X = df.drop(columns=[target]).pct_change().fillna(0)
    y = df[target].pct_change().shift(-1).fillna(0)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_next_day(model, X_row):
    return model.predict(X_row.values.reshape(1, -1))[0]
