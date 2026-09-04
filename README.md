# AutoInsight — Automated Data Analyst

A locally-hostable web app that automatically explores and visualizes CSV datasets with attractive charts.

<p align="center">
  <b>📊 Focused on Data Analysis & EDA</b><br>
  <br>
  AutoInsight provides automatic exploratory data analysis with matplotlib, seaborn, and plotly charts.
</p>

## Overview

AutoInsight is a Streamlit-based application that provides an automated data analysis pipeline:

1. **Data Upload** — Accepts CSV files
2. **Dataset Profiling** — Automatic column type inference and statistics
3. **Exploratory Data Analysis** — Generates summary statistics, distribution plots, correlation heatmaps, and target relationship visualizations

## Project Structure

```
autoinsight/
├ app.py                  # Streamlit entrypoint / UI orchestration
├ modules/
│   ├── __init__.py
│   ├── data_loader.py       # CSV upload, type inference, basic validation
│   └ eda.py                # all plotting/statistics functions (matplotlib + plotly)
├ requirements.txt
├ README.md
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
- View automatic column type inference (numeric, categorical, datetime, boolean)
- See dataset profiling: row/column count, memory usage, missing value percentages, duplicate rows, cardinality

### Phase 2 — EDA (Exploratory Data Analysis)
The app provides 6 tabbed analysis sections:

- **Overview** — Dataset statistics (rows, columns, memory, duplicates, missing values, column type counts)
- **Distributions** — Histograms with KDE for all numeric columns (matplotlib + plotly interactive)
- **Categorical Charts** — Bar charts for categorical columns (top categories)
- **Correlations** — Correlation heatmap (matplotlib + plotly) + top correlated pairs listing
- **Scatter Plots** — Pairwise scatter plots for top correlated numeric pairs (matplotlib + plotly interactive)
- **Target Relationships** — Box plots, bar plots, scatter plots between features and target column

All charts display as images with descriptive titles. Both static (matplotlib/seaborn) and interactive (plotly) versions are available.

### Phase 3 — Report
- Download a summary of your analysis

## Notes

- All processing is done locally — no internet connection or API keys required
- Designed to produce useful results quickly for typical datasets
- Every function has type hints and docstrings
- All plotting functions return figures for Streamlit rendering