import sys
from pathlib import Path

# Add project root directory to Python path for seamless imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib
matplotlib.use('Agg')  # Prevents GUI locking/freezing on Windows
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

# Import src modules
from src.config import DataConfig, ModelConfig
from src.data_loader import load_and_clean_data
from src.feature_engineering import create_time_series_features
from src.models import TimeSeriesForecaster

# Page Configuration
st.set_page_config(
    page_title="Financial Inclusion Time-Series Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Inclusion Time-Series Forecasting Engine")
st.markdown(
    "Automated end-to-end pipeline for loading time-series data, engineering lag/rolling features, "
    "training XGBoost models, and evaluating SHAP feature drivers."
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

test_days = st.sidebar.slider(
    "Test Set Size (Days)", min_value=7, max_value=60, value=20, step=1
)
n_estimators = st.sidebar.slider(
    "XGBoost Estimators", min_value=10, max_value=300, value=100, step=10
)
max_depth = st.sidebar.slider(
    "Max Depth", min_value=2, max_value=10, value=4, step=1
)

# ---------------------------------------------------------
# Cached Pipeline Execution
# ---------------------------------------------------------
@st.cache_data
def run_pipeline(test_days: int, n_estimators: int, max_depth: int):
    # Setup configs
    data_cfg = DataConfig()
    model_cfg = ModelConfig(
        test_size_days=test_days,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # 1. Load Data
    df_raw = load_and_clean_data(data_cfg)

    # 2. Feature Engineering
    df_features = create_time_series_features(df_raw, data_cfg)

    # 3. Model Training & Evaluation
    forecaster = TimeSeriesForecaster(data_cfg, model_cfg)
    X_train, X_test, y_train, y_test = forecaster.prepare_train_test_split(df_features)
    
    forecaster.train(X_train, y_train)
    metrics = forecaster.evaluate(X_test, y_test)
    preds = forecaster.model.predict(X_test)

    return df_features, X_train, X_test, y_train, y_test, forecaster, metrics, preds, data_cfg

# ---------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------
try:
    with st.spinner("Processing features and running XGBoost forecaster..."):
        df_features, X_train, X_test, y_train, y_test, forecaster, metrics, preds, data_cfg = run_pipeline(
            test_days, n_estimators, max_depth
        )

    # 1. Metric Display Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
    col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
    col3.metric("MAPE", f"{metrics.get('MAPE (%)', 0):.2f}%")

    st.markdown("---")

    # 2. Forecast Visualizer
    st.subheader("📊 Forecast vs Actual Transactions")
    fig, ax = plt.subplots(figsize=(12, 4))
    
    test_dates = df_features[data_cfg.date_column].iloc[-len(y_test):]
    ax.plot(test_dates, y_test.values, label="Actuals", color="#1f77b4", linewidth=2)
    ax.plot(test_dates, preds, label="XGBoost Prediction", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_ylabel("Transaction Volume")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("---")

    # 3. SHAP Interpretability Section
    st.subheader("🔍 SHAP Interpretability")
    
    shap_values, _ = forecaster.compute_shap_values(X_test)
    
    col_shap1, col_shap2 = st.columns(2)
    
    with col_shap1:
        st.markdown("**SHAP Summary Plot**")
        fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
        import shap
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf())
        plt.close('all')

    with col_shap2:
        st.markdown("**Top Feature Drivers Table**")
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Mean Absolute SHAP": mean_shap
        }).sort_values(by="Mean Absolute SHAP", ascending=False).reset_index(drop=True)
        
        st.dataframe(shap_df, use_container_width=True)

except Exception as e:
    st.error(f"Execution Error: {str(e)}")
    st.exception(e)