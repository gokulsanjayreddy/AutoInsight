# AutoInsight — Automated Exploratory Data Analysis

A locally-hostable web application that automatically explores and visualizes CSV datasets. AutoInsight generates strictly separated, independent visualizations across all column combinations and allows users to download the entire chart suite in one click.

## Features

- **Direct Results Interface**: Clean, single-page interface without sidebar panels or top toggle tabs.
- **Strictly Separated Charts**: Every distribution, boxplot, scatter plot, and cross-tabulation is rendered as its own independent figure.
- **Comprehensive Column Combinations**:
  - **Summary Statistics and Missing Values**: Table with mean, dispersion, skewness, kurtosis, and missing values breakdown.
  - **Numeric Distributions**: Individual histograms with KDE curves, mean, and median markers.
  - **Outlier Boxplots**: Individual box and whisker plots highlighting quartiles and flagged outliers.
  - **Categorical Distributions**: Frequency and percentage bar charts for categorical columns.
  - **Correlation Matrix**: Full Pearson correlation heatmap across numeric features.
  - **Numeric vs Numeric Relationships**: Bivariate scatter plots with linear regression trendlines for numeric pairs.
  - **Categorical vs Numeric Relationships**: Grouped boxplots comparing numeric metrics across categories.
  - **Categorical vs Categorical Interactions**: 100% normalized cross-tabulation stacked bar charts for category pairs.
- **Batch and Individual Downloads**: Download all generated charts bundled into an organized ZIP archive in one click, or download individual plots as PNGs.
- **Demo Dataset**: 1-click sample dataset loader for instant testing.

## Project Structure

```
autoinsight/
├ app.py                 # Streamlit entrypoint and UI orchestration
├ modules/
│   ├── __init__.py
│   ├── data_loader.py   # CSV ingestion, column type inference, profiling, demo dataset
│   └ eda.py             # Plotting and statistics functions (matplotlib + plotly)
├ tests/
│   ├── test_eda.py      # Unit tests for plot separation and figure generation
│   └ test_pipeline.py   # End-to-end pipeline and ZIP archive tests
├ .github/workflows/
│   ├── ci.yml           # CI workflow (linting, type checking, unit tests)
│   └ pr-comment.yml     # PR validation workflow
├ requirements.txt
└ README.md
```

## Setup Instructions

1. Clone or download the repository:
   ```bash
   git clone https://github.com/gokulsanjayreddy/AutoInsight.git
   cd AutoInsight
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.

## Usage

1. **Upload Dataset**: Upload any CSV file directly on the main page, or click **Load Demo Dataset** to explore sample employee analytics data.
2. **Review Profile**: Inspect row counts, column counts, memory usage, duplicate rows, and data health metrics.
3. **Explore Categorized Results**: Scroll through categorized sections featuring standalone, dedicated charts.
4. **Download Visualizations**:
   - Click **Download All Plots (ZIP)** to download all generated plots organized by category in a single `.zip` file.
   - Click **Download Plot (PNG)** under any individual card to save a specific visualization.

## Testing and Quality Checks

Run the automated test suite and linters:

```bash
# Run unit and pipeline tests
pytest tests/ --tb=short

# Lint check
ruff check .

# Type check
mypy modules/ app.py --ignore-missing-imports
```

## Notes

- Processing runs entirely locally — no external API keys or remote servers required.
- Uses `matplotlib.use("Agg")` for headless reliability in CI and local deployments.
- Plot generation is cached in session state for instant interactions and downloads.