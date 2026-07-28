# src/feature_engineering.py
import pandas as pd
from src.config import DataConfig, FeatureConfig


def create_time_series_features(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """Generates temporal, lag, and rolling features without target leakage."""
    df_feat = df.copy()

    # Calendar features
    dt_series = df_feat[data_cfg.date_column].dt
    df_feat["day_of_week"] = dt_series.dayofweek
    df_feat["day_of_month"] = dt_series.day
    df_feat["month"] = dt_series.month
    df_feat["quarter"] = dt_series.quarter
    df_feat["is_weekend"] = dt_series.dayofweek.isin([5, 6]).astype(int)

    # Lag features (shifted to prevent lookahead bias)
    for lag in feat_cfg.lags:
        df_feat[f"lag_{lag}"] = df_feat[data_cfg.target_column].shift(lag)

    # Rolling window statistics
    for window in feat_cfg.rolling_windows:
        df_feat[f"rolling_mean_{window}"] = (
            df_feat[data_cfg.target_column].shift(1).rolling(window=window).mean()
        )
        df_feat[f"rolling_std_{window}"] = (
            df_feat[data_cfg.target_column].shift(1).rolling(window=window).std()
        )

    # Drop NaNs created by lag and rolling operations
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat