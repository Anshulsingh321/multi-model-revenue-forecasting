import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FEATURES = [
    'revenue_lag_4',
    'revenue_yoy_lag1',
    'trend_lag_1',
    'quarter',
    'cash_lag_1',
    'total_liabilities_lag_1',
]

HYBRID_RIDGE_WEIGHT = 0.8  # weight for Ridge in hybrid model

def train_models(df):
    X = df[FEATURES]
    y = df['revenue']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Ridge (with feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)

    # SARIMA
    try:
        sarima = SARIMAX(y_train, order=(1,1,2), seasonal_order=(1,1,2,4))
        sarima_fit = sarima.fit(disp=False)
        sarima_pred = sarima_fit.get_forecast(steps=len(y_test)).predicted_mean
    except Exception as e:
        sarima_fit = None
        sarima_pred = np.full(len(y_test), np.nan)

    # Holt-Winters
    try:
        holt = ExponentialSmoothing(
            y_train,
            trend='add',
            seasonal='add',
            damped_trend = False,
            seasonal_periods=4
        ).fit()

        holt_pred = holt.forecast(len(y_test))
    except Exception as e:
        holt = None
        holt_pred = np.full(len(y_test), np.nan)

    # Hybrid
    if holt is not None:
        hybrid_pred = HYBRID_RIDGE_WEIGHT * ridge_pred + (1 - HYBRID_RIDGE_WEIGHT) * holt_pred
    else:
        hybrid_pred = ridge_pred

    results = {
        "Ridge": (y_test, ridge_pred),
        "SARIMA": (y_test, sarima_pred),
        "Holt-Winters": (y_test, holt_pred),
        "Hybrid": (y_test, hybrid_pred)
    }

    return results, ridge, sarima_fit, holt