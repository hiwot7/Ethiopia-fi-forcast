# 🇪🇹 Ethiopia Financial Inclusion Forecasting System

An end-to-end Machine Learning Engineering (MLE) pipeline and interactive dashboard designed to forecast financial inclusion transaction volumes across Ethiopia. Built with modular Python modules, automated feature engineering, and Continuous Integration (CI/CD) pipelines.

---

## 📌 Features

* **Modular System Architecture:** Clear separation of concerns with dedicated config, data loading, feature engineering, and UI modules.
* **Fault-Tolerant Data Ingestion:** Auto-detects Excel (`.xlsx`) vs CSV formats and handles non-standard character encodings (`utf-8`, `latin1`).
* **Automated Feature Pipeline:** Generates time-series lag indicators and rolling statistical aggregates automatically.
* **Interactive Analytics Dashboard:** Streamlit UI allowing interactive dataset inspection, parameter adjustments, and forecast visualization.
* **Automated Testing & CI/CD:** GitHub Actions integration running unit tests via `pytest` on push and pull requests.

---

## 📁 Repository Structure

```text
ethiopia-fi-forecast/
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline configuration
├── app/
│   └── main.py                # Streamlit web application & interface
├── data/
│   ├── raw/                   # Raw transaction datasets
│   └── processed/             # Engineered dataset exports
├── src/
│   ├── config.py              # Centralized dataclass configurations & paths
│   ├── data_loader.py         # Multi-format data loader & encoder
│   └── feature_engineering.py # Time-series lag and rolling statistics module
├── tests/                     # Unit test suite for pytest
├── .gitignore                 # Environment and build exclusion rules
├── requirements.txt           # Pinned project dependencies
└── README.md                  # Project documentation
