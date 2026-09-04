# AutoInsight — Automated Data Analyst
 🚧 **Development Version**

A locally-hostable web app that automatically cleans, explores, and models CSV datasets — no code required.

<p align="center">
  <b>⚠️ <strong>WARNING:</strong> This project is under active development.</b><br>
  <br>
  The data profiling, cleaning, and EDA features <strong>work correctly</strong>.<br>
  However, the <strong>modeling/training part is not functioning properly</strong> and may produce unreliable results.<br>
  <br>
  <strong>Currently working features:</strong> Data upload, profiling, cleaning, EDA visualizations, report generation.<br>
  <strong>Known limitations:</strong> Model training, predictions, and model comparison are disabled/restricted.<br>
  <br>
  Use this tool for data exploration and cleaning only. Modeling features will be restored in a future version.
</p>

## Overview

AutoInsight is a Streamlit-based application that provides an end-to-end data analysis pipeline:

1. **Data Upload** — Accepts CSV files (up to ~50MB)
2. **Automated Cleaning** — Handles missing values, outliers, duplicates, and type coercion
3. **Exploratory Data Analysis** — Generates summary statistics, distribution plots, correlation heatmaps, and target relationship visualizations
4. **Model Training & Evaluation** — Auto-detects task type and trains baseline ML models for comparison
5. **Report Generation** — Downloadable Markdown report summarizing all findings

## Project Structure

```
autoinsight/
├── app.py                  # Streamlit entrypoint / UI orchestration
├── modules/
│   ├── __init__.py
│   ├── data_loader.py       # CSV upload, type inference, basic validation
│   ├── cleaner.py           # missing values, outliers, duplicates, type coercion
│   ├── eda.py                # all plotting/statistics functions
│   ├── model_selector.py     # task-type detection (classification/regression)
│   ├── trainer.py            # model training, evaluation, feature importance
│   └── report.py             # assembles a downloadable summary report
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone this repository or download the `autoinsight/` directory
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open your web browser and navigate to `http://localhost:8501`
5. Upload a CSV file and follow the sidebar controls

## Usage

### Phase 1 — Upload & Profiling
- Upload any CSV file via the sidebar file uploader
- View automatic column type inference (numeric, categorical, datetime, boolean, high-cardinality text/ID)
- See dataset profiling: row/column count, memory usage, missing value percentages, duplicate rows, cardinality

### Phase 2 — Cleaning
- Automatic missing value imputation (median for numeric, mode for categorical, with per-column strategy selection)
- IQR-based outlier detection with capping/removal/ignore options
- Duplicate row removal toggle
- Datetime type coercion on string columns that look like dates
- All cleaning steps logged in a human-readable "cleaning log"

### Phase 3 — EDA
- Summary statistics table with skew and kurtosis
- Histogram + KDE plots for all numeric columns
- Bar charts for categorical columns (top 15 categories)
- Correlation heatmap for numeric columns
- Pairwise scatter plots for top correlated numeric pairs
- Target-vs-feature plots once a target column is selected

### Phase 4 — Modeling
- Auto-detect task type (classification vs regression)
- Auto-exclude unusable columns (IDs, high-missing, constant columns)
- Train and compare baseline models:
  - **Classification**: Logistic Regression, Random Forest, Gradient Boosting
  - **Regression**: Linear Regression, Random Forest, Gradient Boosting
- See accuracy, precision, recall, F1, confusion matrices, ROC curves (binary)
- See R², MAE, RMSE, predicted-vs-actual scatter plots (regression)
- Feature importance horizontal bar charts from tree-based models

### Phase 5 — Report
- Download a Markdown summary report containing:
  - Dataset summary
  - Cleaning log
  - Key EDA findings
  - Chosen target and task type
  - Model comparison table
  - Best model's feature importance

## Notes

- All processing is done locally — no internet connection or API keys required
- Results are cached so re-running the app doesn't retrain unnecessarily
- Designed to produce useful results in under 30 seconds for typical datasets
- Every function has type hints and docstrings
- All plotting functions return matplotlib Figure objects for Streamlit rendering