"""Data loading and profiling for AutoInsight.

Handles CSV ingestion, column type inference, dataset profiling,
and sample dataset generation for demonstration.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import pandas as pd


def _extract_raw_bytes(uploaded_file: Any, max_size_mb: float) -> bytes:
    """Read and validate raw bytes from uploaded file or path."""
    if (
        hasattr(uploaded_file, "size")
        and uploaded_file.size is not None
        and uploaded_file.size > max_size_mb * 1024 * 1024
    ):
        raise ValueError(f"File size exceeds the {max_size_mb} MB limit.")

    if hasattr(uploaded_file, "read"):
        raw_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    elif isinstance(uploaded_file, (bytes, bytearray)):
        raw_bytes = bytes(uploaded_file)
    else:
        with open(str(uploaded_file), "rb") as f:
            raw_bytes = f.read()

    if len(raw_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(f"File size exceeds the {max_size_mb} MB limit.")
    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError("Dataset is empty: file contains no data.")

    return bytes(raw_bytes)


def _decode_bytes(raw_bytes: bytes) -> str:
    """Decode bytes using utf-8-sig or latin-1 fallback."""
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw_bytes.decode("latin-1")
        except Exception as e:
            raise ValueError(f"Failed to decode CSV file: {e}") from e


def _sniff_delimiter(text: str) -> str:
    """Sniff CSV delimiter from sample text."""
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:  # noqa: BLE001
        return ","


def _parse_datetime_columns(df: pd.DataFrame) -> None:
    """Coerce datetime strings if at least 90% parse successfully."""
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            clean = df[col].dropna()
            if len(clean) >= 5:
                sample_str = clean.head(20).astype(str)
                if sample_str.str.contains(r"[\-/:]").any():
                    parsed = pd.to_datetime(clean, errors="coerce")
                    if parsed.notna().sum() / len(clean) >= 0.90:
                        df[col] = pd.to_datetime(df[col], errors="coerce")


def load_csv(
    uploaded_file: Any,
    max_size_mb: float = 200.0,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame."""
    raw_bytes = _extract_raw_bytes(uploaded_file, max_size_mb)
    text = _decode_bytes(raw_bytes)
    delimiter = _sniff_delimiter(text)

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=max_rows,
        )
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}") from e

    if df.empty or len(df.columns) == 0 or len(df) == 0:
        raise ValueError("Dataset is empty: file contains no data rows or columns.")

    _parse_datetime_columns(df)
    return df


def _infer_string_type(clean_series: pd.Series) -> str:
    """Infer column type for object/string series."""
    str_vals = set(clean_series.astype(str).str.strip().str.lower().unique())
    if str_vals.issubset({"true", "false", "t", "f", "yes", "no", "y", "n"}):
        return "boolean"

    sample = clean_series.head(10).astype(str)
    if sample.str.contains(r"[\-/:]").all():
        try:
            parsed = pd.to_datetime(sample, errors="raise")
            if len(parsed) == len(sample):
                return "datetime"
        except (ValueError, TypeError):
            pass

    n_unique = clean_series.nunique()
    n_total = len(clean_series)
    if n_unique <= 10 or (n_unique / n_total <= 0.5):
        return "categorical"
    return "high_cardinality_text"


def _infer_numeric_type(clean_series: pd.Series, col_name: str) -> str:
    """Infer column type for numeric series."""
    uniq_vals = set(clean_series.unique())
    is_bool_named = col_name.startswith(("is_", "has_", "flag_")) or col_name in (
        "active",
        "enabled",
        "selected",
    )
    if uniq_vals.issubset({0, 1}) and len(uniq_vals) == 2 and is_bool_named:
        return "boolean"

    return "numeric"


def infer_column_type(series: pd.Series) -> str:
    """Infer semantic column type: numeric, categorical, datetime, boolean, text."""
    clean_series = series.dropna()
    if clean_series.empty:
        return "high_cardinality_text"

    col_name = str(series.name).lower() if series.name is not None else ""

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return _infer_string_type(clean_series)
    if pd.api.types.is_numeric_dtype(series):
        return _infer_numeric_type(clean_series, col_name)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    return "high_cardinality_text"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compute high-level dataset profiling metrics."""
    if df.columns.has_duplicates:
        cols: list[str] = []
        counts: dict[str, int] = {}
        for c in df.columns:
            cs = c
            if cs in counts:
                counts[cs] += 1
                cols.append(f"{cs}_{counts[cs]}")
            else:
                counts[cs] = 0
                cols.append(cs)
        df = df.copy(deep=False)
        df.columns = pd.Index(cols)
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = df.memory_usage(deep=True).sum() / 1_048_576

    missing_percent = (
        (df.isnull().sum() / total_rows * 100).round(2).to_dict()
        if total_rows > 0
        else {}
    )
    duplicate_rows = int(df.duplicated().sum())

    cardinality = {}
    column_types = {}
    for col in df.columns:
        col_type = infer_column_type(df[col])
        column_types[col] = col_type
        if col_type in ("categorical", "high_cardinality_text", "boolean", "numeric"):
            cardinality[col] = df[col].nunique()

    id_columns = flag_id_columns(column_types, cardinality, total_rows)

    return {
        "rows": total_rows,
        "columns": total_cols,
        "memory_mb": round(memory_mb, 2),
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "cardinality": cardinality,
        "column_types": column_types,
        "id_columns": id_columns,
    }


def flag_id_columns(column_types: dict, cardinality: dict, n_rows: int) -> list[str]:
    """Identify columns likely to be surrogate identifiers."""
    id_cols = []
    for col, ctype in column_types.items():
        col_lower = str(col).lower()
        uniq_ratio = cardinality.get(col, 0) / n_rows if n_rows > 0 else 0
        if (ctype == "high_cardinality_text" and uniq_ratio > 0.95) or (
            (col_lower.endswith("_id") or col_lower in ("id", "identifier"))
            and uniq_ratio > 0.90
        ):
            id_cols.append(col)
    return id_cols


def get_sample_dataset() -> pd.DataFrame:
    """Generate a realistic dataset for testing and demonstration."""
    rng = np.random.default_rng(42)
    n = 250

    departments = ["Engineering", "Product", "Design", "Marketing", "Sales", "HR"]
    education_levels = ["Bachelor's", "Master's", "PhD", "High School"]
    performance_ratings = [
        "Needs Improvement",
        "Meets Expectations",
        "Exceeds Expectations",
        "Outstanding",
    ]
    remote_statuses = ["Remote", "Hybrid", "On-site"]

    age = rng.integers(22, 60, size=n)
    years_experience = np.clip(age - 21 + rng.integers(-2, 3, size=n), 0, 38)
    base_salary = 45000 + (years_experience * 3800) + rng.normal(0, 8000, size=n)
    satisfaction_score = np.clip(rng.normal(7.2, 1.8, size=n), 1.0, 10.0).round(1)
    projects_completed = rng.poisson(lam=5 + years_experience * 0.3, size=n)
    bonus_pct = np.clip((satisfaction_score * 2.5) + rng.normal(0, 3, size=n), 0, 35).round(1)

    df: pd.DataFrame = pd.DataFrame({
        "Age": age,
        "YearsExperience": years_experience,
        "Salary": np.round(base_salary, -2),
        "SatisfactionScore": satisfaction_score,
        "ProjectsCompleted": projects_completed,
        "BonusPercentage": bonus_pct,
        "Department": rng.choice(departments, size=n, p=[0.30, 0.18, 0.12, 0.15, 0.17, 0.08]),
        "Education": rng.choice(education_levels, size=n, p=[0.55, 0.30, 0.08, 0.07]),
        "PerformanceRating": rng.choice(performance_ratings, size=n, p=[0.10, 0.50, 0.30, 0.10]),
        "WorkMode": rng.choice(remote_statuses, size=n, p=[0.35, 0.45, 0.20]),
        "Overtime": rng.choice(["Yes", "No"], size=n, p=[0.28, 0.72]),
    })

    df.loc[rng.choice(n, size=12, replace=False), "SatisfactionScore"] = np.nan
    df.loc[rng.choice(n, size=8, replace=False), "BonusPercentage"] = np.nan

    return df