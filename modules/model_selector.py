"""Model selection and task type detection for AutoInsight.

Auto-detects classification vs regression tasks, excludes unusable columns,
and prepares features for model training.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_task_type(target_series: pd.Series, n_unique_threshold: int = 20) -> str:
    """Auto-detect the task type based on the target column.

    Parameters
    ----------
    target_series : pd.Series
        The target column values.
    n_unique_threshold : int, default 20
        If a numeric target has fewer than this many unique values,
        it's treated as classification.

    Returns
    -------
    str
        Either "classification" or "regression".
    """
    n_unique = target_series.nunique(dropna=True)

    # If target is numeric with low cardinality -> classification
    if pd.api.types.is_numeric_dtype(target_series) and n_unique < n_unique_threshold:
        return "classification"

    # If target is categorical -> classification
    if not pd.api.types.is_numeric_dtype(target_series) or target_series.dtype.name == "category":
        return "classification"

    # Otherwise -> regression
    return "regression"


def detect_unusable_columns(
    df: pd.DataFrame,
    target_col: str,
    missing_threshold: float = 0.90,
) -> tuple[list, list, list]:
    """Identify columns that are unusable for modeling.

    Unusable columns include:
    - IDs / near-unique columns
    - Columns with >90% missing values
    - Constant columns (only one unique value)
    - The target column itself

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    target_col : str
        Name of the target column.
    missing_threshold : float, default 0.90
        Ratio of missing values above which a column is flagged.

    Returns
    -------
    tuple[list, list, list]
        - List of ID column names
        - List of high-missing-column names
        - List of constant column names
    """
    # Infer column types
    from modules.data_loader import infer_column_type, profile_dataset

    profiling = profile_dataset(df)
    column_types = profiling["column_types"]
    cardinality = profiling["cardinality"]

    n_rows = len(df)

    id_cols = []
    high_missing_cols = []
    constant_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        # ID columns: near-unique high-cardinality text
        if column_types.get(col) == "high_cardinality_text":
            uniq_ratio = cardinality.get(col, 0) / n_rows if n_rows > 0 else 0
            if uniq_ratio > 0.95:
                id_cols.append(col)

        # High-missing columns
        missing_pct = df[col].isnull().mean()
        if missing_pct > missing_threshold:
            high_missing_cols.append(col)

        # Constant columns
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)

    return id_cols, high_missing_cols, constant_cols


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    encoding_strategy: str = "auto",
) -> tuple:
    """Prepare feature matrix and target vector for modeling.

    Handles:
    - Removal of unusable columns (IDs, high-missing, constant)
    - Encoding of categorical features (one-hot for low cardinality,
      label/target encoding for high cardinality)
    - Scaling of numeric features (StandardScaler)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    target_col : str
        Name of the target column.
    task_type : str
        Either "classification" or "regression".
    encoding_strategy : str, default "auto"
        "onehot", "label", or "auto".

    Returns
    -------
    tuple
        (X, y, feature_names) where X is the feature matrix,
        y is the target vector, and feature_names is a list.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    df = df.copy()
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Detect unusable columns
    id_cols, high_missing_cols, constant_cols = detect_unusable_columns(
        X, target_col=target_col
    )

    # Remove unusable columns
    unusable = set(id_cols + high_missing_cols + constant_cols)
    # Also remove if only 1 row left
    unusable.discard(target_col)

    # Remove from X
    cols_to_drop = [c for c in X.columns if c in unusable]
    X = X.drop(columns=cols_to_drop, errors="ignore")

    # Separate numeric and categorical columns remaining
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Ensure target is encoded for classification
    if task_type == "classification" and pd.api.types.is_numeric_dtype(y) and y.dtype == np.float64:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Determine encoding strategy per categorical column
    encoding_map = {}
    for col in cat_cols:
        n_unique = X[col].nunique(dropna=True)
        if task_type == "classification" and n_unique <= 10:
            encoding_map[col] = "onehot"
        elif encoding_strategy == "auto":
            # Use onehot if low cardinality, else label
            encoding_map[col] = "onehot" if n_unique <= 5 else "label"
        else:
            encoding_map[col] = encoding_strategy

    # Build preprocessing pipeline
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", StandardScaler(), numeric_cols)
        )

    if cat_cols and encoding_map:
        if encoding_map.get(list(encoding_map.keys())[0], "") == "onehot":
            from sklearn.preprocessing import OneHotEncoder

            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )
        else:
            from sklearn.preprocessing import OrdinalEncoder

            # Ordinal encode each categorical column (for high cardinality)
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            transformers.append(
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
            )

    if not transformers:
        # No columns to transform, just return as-is
        feature_names = X.columns.tolist()
        return X, y, feature_names

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    return X_transformed, y, feature_names