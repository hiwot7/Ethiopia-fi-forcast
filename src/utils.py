# src/utils.py
from typing import Dict
import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates evaluation metrics: MAE, RMSE, and MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Prevent division by zero for MAPE
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
            )
            * 100
        )
    else:
        mape = np.nan

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE (%)": round(mape, 4)}