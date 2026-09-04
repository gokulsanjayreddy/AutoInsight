"""Data loading and profiling for AutoInsight.

Handles CSV upload, column type inference, and basic dataset profiling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The uploaded CSV file from Streamlit's file uploader.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file cannot be parsed as CSV.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}") from e


def infer_column_type(series: pd.Series) -> str:
    """Infer the type of a pandas Series.

    Categories: numeric, categorical, datetime, boolean, high_cardinality_text.

    Parameters
    ----------
    series : pd.Series
        A single column from a DataFrame.

    Returns
    -------
    str
        Inferred type label.
    """
    # Boolean check - must be before numeric check
    if set(series.dropna().unique()).issubset({True, False, 0, 1}):
        return "boolean"

    # Datetime check
    try:
        pd.to_datetime(series, errors="raise")
        return "datetime"
    except (ValueError, TypeError):
        pass

    # Numeric check
    if pd.api.types.is_numeric_dtype(series):
        # High-cardinality numeric treated as ID-like
        if series.nunique() / len(series) > 0.95:
            return "high_cardinality_text"
        return "numeric"

    # Categorical / text
    if pd.api.types.is_categorical_dtype(series) or series.nunique() / len(series) < 0.5:
        return "categorical"

    # Default to high-cardinality text/ID
    return "high_cardinality_text"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compute profiling statistics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.

    Returns
    -------
    dict
        Profiling info including:
        - rows, columns
        - memory_usage_mb
        - missing_percent (per column)
        - duplicate row count
        - cardinality per categorical column
        - column types
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_mb = df.memory_usage(deep=True).sum() / 1_048_576

    missing_percent = (df.isnull().sum() / total_rows * 100).round(2).to_dict()

    duplicate_rows = int(df.duplicated().sum())

    cardinality = {}
    column_types = {}
    for col in df.columns:
        col_type = infer_column_type(df[col])
        column_types[col] = col_type
        if col_type in ("categorical", "high_cardinality_text"):
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


def flag_id_columns(column_types: dict, cardinality: dict, n_rows: int) -> list:
    """Flag columns likely to be IDs (near-unique values).

    Parameters
    ----------
    column_types : dict
        Mapping of column name -> inferred type.
    cardinality : dict
        Mapping of column name -> unique value count.
    n_rows : int
        Total number of rows in the dataset.

    Returns
    -------
    list[str]
        List of column names flagged as potential IDs.
    """
    id_cols = []
    for col, ctype in column_types.items():
        if ctype == "high_cardinality_text":
            uniq_ratio = cardinality.get(col, 0) / n_rows if n_rows > 0 else 0
            if uniq_ratio > 0.95:
                id_cols.append(col)
    return id_cols