"""Model training and evaluation module for AutoInsight.

Trains baseline ML models and returns performance metrics,
feature importance, and learning curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def train_classification_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate classification models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector (binary or multiclass).
    test_size : float, default 0.2
        Proportion of test split.
    random_state : int, default 42

    Returns
    -------
    dict
        Dictionary mapping model name to dict of:
        - model: fitted model
        - accuracy, precision, recall, f1
        - confusion_matrix (np.ndarray)
        - roc_auc (float or None)
        - roc_fpr, roc_tpr (arrays for plotting)
        - y_test, y_pred (arrays)
    """
    # Handle NaN values in target - drop rows with NaN y
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return {"error": "Not enough valid target values for training (minimum 10 required after NaN removal)."}

    # Try stratified split, fall back to non-stratified if classes are too imbalanced
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError:
        # Stratification failed - fall back to non-stratified split
        # This happens when some classes have too few members
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Multi-class AUC via One-vs-Rest approximation
        roc_auc = None
        fpr, tpr = None, None
        if y_proba is not None and len(np.unique(y)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
            except Exception:
                pass

        # Precision/Recall need average parameter for multiclass
        avg = "binary" if len(np.unique(y)) == 2 else "weighted"

        precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_auc": roc_auc,
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


def train_regression_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate regression models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector (continuous).
    test_size : float, default 0.2
        Proportion of test split.
    random_state : int, default 42

    Returns
    -------
    dict
        Dictionary mapping model name to dict of:
        - model: fitted model
        - r2, mae, rmse
        - y_test, y_pred (arrays)
    """
    # Handle NaN values in target - drop rows with NaN y
    mask = ~pd.isna(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return {"error": "Not enough valid target values for training (minimum 10 required after NaN removal)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "model": model,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


def plot_confusion_matrix(cm: np.ndarray, labels: list) -> plt.Figure:
    """Plot a confusion matrix as a matplotlib Figure.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    labels : list
        Class labels.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Confusion Matrix", pad=10)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return fig


def plot_roc_curve(fpr, tpr, roc_auc: float = None) -> plt.Figure:
    """Plot ROC curve as a matplotlib Figure.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    roc_auc : float, optional
        AUC score.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})" if roc_auc else "ROC curve")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plot predicted vs actual values scatter plot for regression.

    Parameters
    ----------
    y_test : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="w", linewidth=0.5)
    # Plot ideal line
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.set_xlim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
    ax.set_ylim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance from tree-based models.

    Parameters
    ----------
    model : fitted model
        Must have feature_importances_ or coef_ attributes.
    feature_names : list[str]
        List of feature names.
    top_n : int, default 20
        Number of top features to display.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    import numpy as np

    # Get importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        importances = np.zeros(len(feature_names))

    # Pair with feature names and sort
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:top_n]

    names = [p[0] for p in top_pairs]
    values = [p[1] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y_pos = range(len(names))
    ax.barh(list(reversed(y_pos)), values, align="center")
    ax.set_yticks(list(reversed(y_pos)))
    ax.set_yticklabels([n if len(n) < 30 else n[:27] + "..." for n in reversed(names)])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig