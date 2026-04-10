import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_models(results):
    metrics = []

    for name, (y_true, y_pred) in results.items():
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Handle division by zero in MAPE
        if (y_true == 0).any():
            mape = np.nan
        else:
            mape = mean_absolute_percentage_error(y_true, y_pred)

        metrics.append([name, rmse, mape])

    df_metrics = pd.DataFrame(metrics, columns=["Model", "RMSE", "MAPE"])

    # Drop models where MAPE could not be computed
    valid_metrics = df_metrics.dropna(subset=["MAPE"])

    if len(valid_metrics) == 0:
        best_model = df_metrics.iloc[0]
    else:
        best_model = valid_metrics.loc[valid_metrics["MAPE"].idxmin()]

    return df_metrics, best_model