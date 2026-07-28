# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Resolve absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
    processed_data_path: Path = (
        PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
    )
    date_column: str = "date"
    target_column: str = "transaction_volume"


@dataclass
class FeatureConfig:
    lags: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])


@dataclass
class ModelConfig:
    test_size_days: int = 20
    n_estimators: int = 100
    max_depth: int = 4
    random_state: int = 42