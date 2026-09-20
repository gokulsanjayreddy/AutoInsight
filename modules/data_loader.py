"""Data loading and profiling for AutoInsight.

Handles CSV ingestion, column type inference, dataset profiling,
and sample dataset generation for demonstration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}") from e


def infer_column_type(series: pd.Series) -> str:
    """Infer semantic column type: numeric, categorical, datetime, boolean, high_cardinality_text."""
    clean_series = series.dropna()
    if clean_series.empty:
        return "high_cardinality_text"

    if pd.api.types.is_bool_dtype(series) or set(clean_series.unique()).issubset({0, 1}):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = clean_series.head(5).astype(str)
        if sample.str.contains(r"[\-/:]").all():
            try:
                pd.to_datetime(sample, errors="raise")
                return "datetime"
            except (ValueError, TypeError):
                pass

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() / len(series) > 0.98 and len(series) > 50:
            return "high_cardinality_text"
        return "numeric"

    if isinstance(series.dtype, pd.CategoricalDtype) or series.nunique() / len(series) < 0.5:
        return "categorical"

    return "high_cardinality_text"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compute high-level dataset profiling metrics."""
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = df.memory_usage(deep=True).sum() / 1_048_576

    missing_percent = (df.isnull().sum() / total_rows * 100).round(2).to_dict() if total_rows > 0 else {}
    duplicate_rows = int(df.duplicated().sum())

    cardinality = {}
    column_types = {}
    for col in df.columns:
        col_type = infer_column_type(df[col])
        column_types[col] = col_type
        if col_type in ("categorical", "high_cardinality_text", "boolean"):
            cardinality[col] = int(df[col].nunique())

    return {
        "rows": total_rows,
        "columns": total_cols,
        "memory_mb": round(memory_mb, 2),
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "cardinality": cardinality,
        "column_types": column_types,
    }


def flag_id_columns(column_types: dict, cardinality: dict, n_rows: int) -> list[str]:
    """Identify columns likely to be surrogate identifiers."""
    id_cols = []
    for col, ctype in column_types.items():
        if ctype == "high_cardinality_text":
            uniq_ratio = cardinality.get(col, 0) / n_rows if n_rows > 0 else 0
            if uniq_ratio > 0.95:
                id_cols.append(col)
    return id_cols


def get_sample_dataset() -> pd.DataFrame:
    """Generate a realistic dataset for testing and demonstration."""
    np.random.seed(42)
    n = 250

    departments = ["Engineering", "Product", "Design", "Marketing", "Sales", "HR"]
    education_levels = ["Bachelor's", "Master's", "PhD", "High School"]
    performance_ratings = ["Needs Improvement", "Meets Expectations", "Exceeds Expectations", "Outstanding"]
    remote_statuses = ["Remote", "Hybrid", "On-site"]

    age = np.random.randint(22, 60, size=n)
    years_experience = np.clip(age - 21 + np.random.randint(-2, 3, size=n), 0, 38)
    base_salary = 45000 + (years_experience * 3800) + np.random.normal(0, 8000, size=n)
    satisfaction_score = np.clip(np.random.normal(7.2, 1.8, size=n), 1.0, 10.0).round(1)
    projects_completed = np.random.poisson(lam=5 + years_experience * 0.3, size=n)
    bonus_percentage = np.clip((satisfaction_score * 2.5) + np.random.normal(0, 3, size=n), 0, 35).round(1)

    df = pd.DataFrame({
        "Age": age,
        "YearsExperience": years_experience,
        "Salary": np.round(base_salary, -2),
        "SatisfactionScore": satisfaction_score,
        "ProjectsCompleted": projects_completed,
        "BonusPercentage": bonus_percentage,
        "Department": np.random.choice(departments, size=n, p=[0.30, 0.18, 0.12, 0.15, 0.17, 0.08]),
        "Education": np.random.choice(education_levels, size=n, p=[0.55, 0.30, 0.08, 0.07]),
        "PerformanceRating": np.random.choice(performance_ratings, size=n, p=[0.10, 0.50, 0.30, 0.10]),
        "WorkMode": np.random.choice(remote_statuses, size=n, p=[0.35, 0.45, 0.20]),
        "Overtime": np.random.choice(["Yes", "No"], size=n, p=[0.28, 0.72]),
    })

    missing_indices_sat = np.random.choice(n, size=12, replace=False)
    df.loc[missing_indices_sat, "SatisfactionScore"] = np.nan

    missing_indices_bonus = np.random.choice(n, size=8, replace=False)
    df.loc[missing_indices_bonus, "BonusPercentage"] = np.nan

    return df