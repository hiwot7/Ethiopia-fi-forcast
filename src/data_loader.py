# src/data_loader.py
import logging
import pandas as pd
from src.config import DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_clean_data(config: DataConfig) -> pd.DataFrame:
    """Safely loads data from Excel or CSV files, handling binary encodings and formats."""
    path = str(config.raw_data_path)
    logger.info(f"Loading data from: {path}")

    df = None

    # 1. If the file has an Excel extension or fails CSV reading, parse as Excel
    if path.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(path)
            logger.info("Loaded successfully via pd.read_excel.")
        except Exception as e:
            logger.error(f"Failed to read as Excel: {e}")

    # 2. If df is still None, try CSV parsing with fallback encodings
    if df is None:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except (UnicodeDecodeError, Exception):
            try:
                # Direct attempt at Excel fallback if file was renamed .csv
                df = pd.read_excel(path)
                logger.info("Loaded binary file successfully via pd.read_excel fallback.")
            except Exception:
                # Final fallback for legacy CSV text encodings
                df = pd.read_csv(path, encoding="latin1")

    # 3. Clean and parse dates
    if config.date_column in df.columns:
        df[config.date_column] = pd.to_datetime(df[config.date_column])
        df = df.sort_values(by=config.date_column).reset_index(drop=True)

    return df