"""Data cleaning module for AutoInsight.

Handles missing values, outliers, duplicates, and type coercion.
Every cleaning step is logged in a human-readable cleaning log list.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict,
) -> tuple[pd.DataFrame, list]:
    """Impute missing values per column according to the given strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may be modified in-place or copied).
    strategy : dict
        Mapping of column name -> imputation strategy.
        Numeric columns: "mean" or "median".
        Categorical columns: "mode" or "unknown".

    Returns
    -------
    tuple[pd.DataFrame, list]
        Cleaned DataFrame and a list of human-readable log messages.
    """
    df = df.copy()
    logs = []

    for col, strat in strategy.items():
        if strat is None or df[col].isnull().sum() == 0:
            continue

        null_count = int(df[col].isnull().sum())

        if pd.api.types.is_numeric_dtype(df[col]):
            if strat == "mean":
                fill_val = df[col].mean()
                df[col].fillna(fill_val, inplace=True)
                logs.append(
                    f"Filled {null_count} missing values in '{col}' with mean ({fill_val:.1f})"
                )
            elif strat == "median":
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                logs.append(
                    f"Filled {null_count} missing values in '{col}' with median ({fill_val:.1f})"
                )
        else:
            # Categorical
            if strat == "mode":
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(fill_val, inplace=True)
                logs.append(
                    f"Filled {null_count} missing values in '{col}' with mode ({fill_val})"
                )
            elif strat == "unknown":
                df[col].fillna("Unknown", inplace=True)
                logs.append(
                    f"Filled {null_count} missing values in '{col}' with 'Unknown'"
                )

    return df, logs


def detect_and_flag_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: list,
) -> tuple[pd.DataFrame, dict, list]:
    """Flag outliers using the IQR method on numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list[str]
        List of numeric column names.

    Returns
    -------
    tuple[pd.DataFrame, dict, list]
        - Cleaned DataFrame (outliers capped by default)
        - dict mapping column -> count of flagged outliers
        - list of human-readable log messages
    """
    df = df.copy()
    outlier_counts = {}
    logs = []

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())

        if count > 0:
            outlier_counts[col] = count
            # Cap outliers to the bounds instead of removing
            df[col] = df[col].clip(lower, upper)
            logs.append(
                f"Flagged {count} outlier(s) in '{col}' (IQR method), capped to [{{lower:.1f}}, {{upper:.1f}}]"
            )

    return df, outlier_counts, logs


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Remove duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, list]
        DataFrame with duplicates removed and a log message.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_removed = n_before - len(df_clean)

    if n_removed > 0:
        logs.append(f"Removed {n_removed} duplicate row(s)")
    else:
        logs.append("No duplicate rows found — keeping all")

    return df_clean, ["Removed duplicate rows"] if n_removed > 0 else ["No duplicate rows found"]


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Attempt datetime parsing on string columns that look like dates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, list]
        DataFrame with coerced types and a list of log messages.
    """
    df = df.copy()
    logs = []

    for col in df.columns:
        if df[col].dtype == object:
            # Try datetime parsing on a sample
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors="raise")
                    # If successful, convert the whole column
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    logs.append(f"Coerced column '{col}' to datetime")
                    continue
                except (ValueError, TypeError):
                    pass

    return df, logs