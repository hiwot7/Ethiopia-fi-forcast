# src/models.py
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import DataConfig, ModelConfig
from src.utils import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Handles time-aware data splitting, training XGBoost regressors, and computing SHAP values."""

    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.model: xgb.XGBRegressor = None
        self.feature_names: list = []

    def prepare_train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        drop_cols = [self.data_cfg.date_column, self.data_cfg.target_column]
        features = [c for c in df.columns if c not in drop_cols]
        self.feature_names = features

        X = df[features]
        y = df[self.data_cfg.target_column]

        split_idx = len(df) - self.model_cfg.test_size_days
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = xgb.XGBRegressor(
            n_estimators=self.model_cfg.n_estimators,
            max_depth=self.model_cfg.max_depth,
            random_state=self.model_cfg.random_state,
            learning_rate=0.05,
            objective="reg:squarederror",
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        predictions = self.model.predict(X_test)
        return calculate_metrics(y_test.values, predictions)

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, shap.Explainer]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values, explainer