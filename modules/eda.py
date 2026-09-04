"""EDA (Exploratory Data Analysis) module for AutoInsight.

Provides plotting and statistics functions for dataset overview,
distributions, correlations, and target relationships.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True})


def plot_summary_statistics(df: pd.DataFrame) -> plt.Figure:
    """Generate a summary statistics table with skew and kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the summary table.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
        return fig

    desc = num_df.describe().T
    desc["skewness"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=desc.round(3).values,
        colLabels=desc.columns,
        rowLabels=desc.index,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title("Summary Statistics (with Skew & Kurtosis)", pad=20)
    return fig


def plot_distributions(df: pd.DataFrame, max_cols: int = 30) -> list[plt.Figure]:
    """Generate histogram + KDE plots for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    max_cols : int, default 30
        Maximum number of columns to plot (to avoid UI overload).

    Returns
    -------
    list[plt.Figure]
        List of matplotlib figures, one per column (or grouped).
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return []

    columns = num_df.columns.tolist()
    n_cols = min(len(columns), max_cols)
    figures = []

    # If too many numeric cols, create subplots in batches
    for start in range(0, n_cols, max_cols):
        batch = columns[start : start + max_cols]
        n = len(batch)
        fig, axes = plt.subplots(
            nrows=min(n, 5), ncols=min(n, 5), figsize=(5 * min(n, 5), 4 * min(n, 5))
        )
        axes = axes.flatten() if n > 1 else [axes]

        for i, col in enumerate(batch):
            if i >= len(axes):
                break
            ax = axes[i]
            data = num_df[col].dropna()
            if len(data) > 0:
                sns.histplot(data, kde=True, ax=ax, bins="auto")
                ax.set_title(f"{col}", fontsize=10)
            else:
                ax.set_title(f"{col} (empty)", fontsize=10)

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"Distribution Plots (showing {n} of {len(columns)} numeric columns)",
            fontsize=14,
            y=1.02,
        )
        figures.append(fig)

    return figures


def plot_categorical_bars(df: pd.DataFrame, max_categories: int = 15) -> list[plt.Figure]:
    """Generate bar charts for categorical columns (top N categories).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    max_categories : int, default 15
        Maximum categories to display per column.

    Returns
    -------
    list[plt.Figure]
        List of matplotlib figures.
    """
    cat_df = df.select_dtypes(
        include=["object", "category"]
    ).copy()
    if cat_df.empty:
        return []

    figures = []
    for col in cat_df.columns:
        value_counts = df[col].value_counts(dropna=False).head(max_categories)
        # Convert index to strings for seaborn compatibility
        value_counts_idx = [str(idx) for idx in value_counts.index]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=value_counts.values,
            y=value_counts_idx,
            ax=ax,
            order=value_counts_idx,
        )
        ax.set_title(f"{col} (top {min(max_categories, df[col].nunique())} categories)", fontsize=10)
        ax.xaxis.set_label_position("bottom")
        figures.append(fig)

    return figures


def plot_correlation_heatmap(df: pd.DataFrame, max_correlated: int = 15) -> plt.Figure:
    """Generate a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    max_correlated : int, default 15
        Maximum number of top-correlated pairs to highlight (not a limit on matrix size).

    Returns
    -------
    plt.Figure
        Matplotlib figure with the heatmap.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
        return fig

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", pad=20)
    return fig


def plot_pairwise_scatter(
    df: pd.DataFrame, max_plots: int = 6, sample_size: int = 500
) -> list[plt.Figure]:
    """Generate pairwise scatter plots for the top-N most correlated numeric pairs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    max_plots : int, default 6
        Maximum number of scatter plots to generate.
    sample_size : int, default 500
        Sample size for large datasets to avoid performance issues.

    Returns
    -------
    list[plt.Figure]
        List of matplotlib figure objects.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        return []

    # Sample data if too large
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    corr_matrix = sample_df.corr().abs()
    # Get upper triangle mask
    upper = corr_matrix.where(np.tri(len(corr_matrix), k=1, dtype=bool))

    # Get top correlated pairs
    stacked = upper.unstack()
    sorted_pairs = stacked.dropna().sort_values(ascending=False)
    top_pairs = sorted_pairs.head(max_plots)

    figures = []
    if not top_pairs.empty:
        n = len(top_pairs)
        fig, axes = plt.subplots(
            nrows=(n + 1) // 2, ncols=2, figsize=(6 * 2, 4 * ((n + 1) // 2))
        )
        axes = axes.flatten()

        for i, ( (col1, col2), corr_val ) in enumerate(top_pairs.items()):
            if i >= len(axes):
                break
            ax = axes[i]
            sns.scatterplot(
                data=sample_df,
                x=col1,
                y=col2,
                ax=ax,
                alpha=0.6,
            )
            ax.set_title(f"{col1} vs {col2}\n(r = {corr_val:.2f})", fontsize=10)
            ax.set_xlabel(col1, fontsize=8)
            ax.set_ylabel(col2, fontsize=8)

        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"Top Correlated Numeric Pairs (max {max_plots} plots)",
            fontsize=14,
            y=1.02,
        )
        figures.append(fig)

    return figures


def plot_target_relationships(
    df: pd.DataFrame,
    target_col: str,
) -> list[plt.Figure]:
    """Generate target-vs-feature plots.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset with a target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    list[plt.Figure]
        List of matplotlib figures.
    """
    figures = []
    num_df = df.select_dtypes(include=[np.number]).copy()
    cat_df = df.select_dtypes(
        include=["object", "category"]
    ).copy()

    if target_col not in df.columns:
        return figures

    # Numeric features vs target
    for col in num_df.columns:
        if col == target_col:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x=target_col, y=col, ax=ax)
        ax.set_title(f"{col} vs {target_col} (numeric target)", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        figures.append(fig)

    # Categorical features vs target (grouped bar)
    for col in cat_df.columns:
        if col == target_col:
            continue
        if df[col].nunique() > 20:
            continue  # skip high-cardinality categorical
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            sns.barplot(
                data=df,
                x=target_col,
                y=col,
                hue=col,
                ax=ax,
                dodge=True,
                errorbar="ci",
            )
            ax.set_title(f"{col} vs {target_col} (categorical target)", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            figures.append(fig)
        except Exception:
            pass

    # Numeric target vs numeric feature scatter
    if pd.api.types.is_numeric_dtype(df[target_col]):
        for col in num_df.columns:
            if col == target_col:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                sns.scatterplot(data=df, x=col, y=target_col, ax=ax, alpha=0.6)
                ax.set_title(f"{col} vs {target_col} (scatter)", fontsize=10)
                figures.append(fig)
            except Exception:
                pass

    return figures