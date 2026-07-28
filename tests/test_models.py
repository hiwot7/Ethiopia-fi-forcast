# tests/test_models.py
import pytest
import pandas as pd
import numpy as np
from src.config import DataConfig, ModelConfig
from src.models import TimeSeriesForecaster
from src.utils import calculate_metrics

def test_calculate_metrics():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 300])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE (%)" in metrics
    assert metrics["MAE"] == 6.6667

def test_forecaster_fit_predict():
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "transaction_volume": np.random.normal(100, 10, 50),
        "lag_1": np.random.normal(100, 10, 50),
        "day_of_week": dates.dayofweek
    })
    
    data_cfg = DataConfig(date_column="date", target_column="transaction_volume")
    model_cfg = ModelConfig(test_size_days=10, n_estimators=5, max_depth=2)
    
    forecaster = TimeSeriesForecaster(data_cfg, model_cfg)
    X_train, X_test, y_train, y_test = forecaster.prepare_train_test_split(df)
    
    assert len(X_test) == 10
    forecaster.train(X_train, y_train)
    metrics = forecaster.evaluate(X_test, y_test)
    assert metrics["RMSE"] >= 0