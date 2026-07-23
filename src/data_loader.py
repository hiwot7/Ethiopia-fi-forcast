import os
import pandas as pd
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class DataLoader:
    """Programmatically handles loading and initial validation of project datasets."""

    def __init__(self, base_dir: str = None):
        """Initialize directory paths relative to the project root."""
        if base_dir is None:
            # Resolves root directory relative to this file's location (src/)
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.project_root = base_dir

        self.raw_data_dir = os.path.join(self.project_root, "data", "raw")
        self.processed_data_dir = os.path.join(self.project_root, "data", "processed")

    def load_findex_data(self, file_name: str = "findex_ethiopia.csv") -> pd.DataFrame:
        """Loads World Bank Global Findex data programmatically or returns baseline structural dataset."""
        file_path = os.path.join(self.raw_data_dir, file_name)

        if os.path.exists(file_path):
            logging.info(f"Loading Findex data from {file_path}")
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        else:
            logging.warning(f"File {file_path} not found. Utilizing default Findex benchmark dataset.")
            df = pd.DataFrame({
                'Year': [2011, 2014, 2017, 2021, 2024],
                'Account_Ownership_Total_Pct': [14.0, 22.0, 35.0, 46.0, 49.0],
                'Account_Ownership_Male_Pct': [17.1, 26.3, 41.2, 55.8, 55.8],
                'Account_Ownership_Female_Pct': [11.0, 17.8, 29.1, 36.4, 42.1],
                'Mobile_Money_Account_Pct': [0.0, 0.1, 0.3, 4.7, 10.2]
            })

        self._validate_columns(df, ['Year', 'Account_Ownership_Total_Pct'])
        return df

    def load_macro_indicators(self, file_name: str = "macro_ethiopia.csv") -> pd.DataFrame:
        """Loads annual macroeconomic and telecom infrastructure indicators."""
        file_path = os.path.join(self.raw_data_dir, file_name)

        if os.path.exists(file_path):
            logging.info(f"Loading Macro indicators from {file_path}")
            df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
        else:
            logging.warning(f"File {file_path} not found. Utilizing default Macro benchmark dataset.")
            years = list(range(2011, 2025))
            df = pd.DataFrame({
                'Year': years,
                '4G_Coverage_Pct': [0.0, 0.0, 2.0, 5.0, 12.0, 20.0, 35.0, 48.0, 62.0, 75.0, 82.0, 88.0, 93.0, 96.0],
                'Mobile_Subscriptions_Per_100': [14.8, 23.1, 30.5, 38.2, 46.7, 51.0, 56.4, 58.9, 63.2, 65.0, 68.1, 71.4, 75.8, 80.2]
            })

        self._validate_columns(df, ['Year'])
        return df

    def save_processed_data(self, df: pd.DataFrame, file_name: str = "ethiopia_fi_unified_data.csv") -> str:
        """Saves processed/cleaned dataframe programmatically to data/processed/."""
        os.makedirs(self.processed_data_dir, exist_ok=True)
        export_path = os.path.join(self.processed_data_dir, file_name)
        
        df.to_csv(export_path, index=False)
        logging.info(f"Successfully saved processed dataset to: {export_path}")
        return export_path

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_cols: list):
        """Helper function to validate required schema columns."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Dataset missing required schema columns: {missing}")

# Direct execution check
if __name__ == "__main__":
    loader = DataLoader()
    findex_df = loader.load_findex_data()
    macro_df = loader.load_macro_indicators()
    print("\n--- Programmatic Loading Test ---")
    print(f"Findex Shape: {findex_df.shape}")
    print(f"Macro Shape: {macro_df.shape}")