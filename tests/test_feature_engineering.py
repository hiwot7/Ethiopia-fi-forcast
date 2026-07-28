# tests/test_feature_engineering.py
import numpy as np
import pandas as pd
import pytest
from src.config import DataConfig, FeatureConfig
from src.feature_engineering import create_time_series_features


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    values = np.random.normal(loc=100, scale=10, size=60)
    return pd.DataFrame({"date": dates, "transaction_volume": values})


def test_create_time_series_features(sample_df):
    data_cfg = DataConfig(date_column="date", target_column="transaction_volume")
    feat_cfg = FeatureConfig(lags=[1, 7], rolling_windows=[7])

    df_feats = create_time_series_features(sample_df, data_cfg, feat_cfg)

    # Assert features were generated
    assert "lag_1" in df_feats.columns
    assert "lag_7" in df_feats.columns
    assert "rolling_mean_7" in df_feats.columns
    assert "day_of_week" in df_feats.columns

    # Verify no NaN values remain
    assert df_feats.isna().sum().sum() == 0