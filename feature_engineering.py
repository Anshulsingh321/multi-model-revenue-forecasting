def create_features(df):
    df = df.copy()

    df['quarter'] = df['date'].dt.quarter

    df['revenue_lag_4'] = df['revenue'].shift(4)
    df['revenue_yoy'] = df['revenue'] / df['revenue_lag_4'] - 1
    df['revenue_yoy_lag1'] = df['revenue_yoy'].shift(1)

    df['trend_lag_1'] = df['trend'].shift(1)
    df['cash_lag_1'] = df['cash'].shift(1)
    df['total_liabilities_lag_1'] = df['total_liabilities'].shift(1)

    df = df.dropna()

    return df