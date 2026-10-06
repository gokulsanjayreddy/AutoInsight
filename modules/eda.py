"""EDA (Exploratory Data Analysis) module for AutoInsight.

Generates independent figures for univariate, bivariate, and multivariate
data analysis across all column interactions.
"""

from __future__ import annotations

import io
import itertools
import zipfile
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.max_open_warning": 0,
})

PRIMARY_COLOR = "#2563eb"


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Convert a matplotlib figure to PNG byte data."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def create_all_plots_zip(named_figures: Sequence[tuple[str, str, bytes | plt.Figure]]) -> bytes:
    """Create an in-memory zip archive containing all generated plots."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()
        for folder, name_base, content in named_figures:
            clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name_base)
            zip_path = f"{folder}/{clean_name}.png"
            counter = 1
            while zip_path in used_names:
                zip_path = f"{folder}/{clean_name}_{counter}.png"
                counter += 1
            used_names.add(zip_path)

            if isinstance(content, bytes):
                png_data = content
            elif isinstance(content, bytearray):
                png_data = bytes(content)
            else:
                png_data = fig_to_png_bytes(content)
                plt.close(content)
            zf.writestr(zip_path, png_data)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def get_categorical_columns(
    df: pd.DataFrame,
    max_cardinality: int = 25,
    id_cols: list[str] | None = None,
) -> list[str]:
    """Return categorical and low-cardinality discrete columns suitable for grouping."""
    excluded = set(id_cols or [])
    cat_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Check if column is datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            n_unique = df[col].nunique()
            if 1 < n_unique <= max_cardinality:
                cat_cols.append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols


def plot_summary_statistics(df: pd.DataFrame) -> plt.Figure:
    """Generate a summary statistics table figure with skewness and kurtosis."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No numeric columns found",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
        )
        ax.axis("off")
        return fig

    desc = num_df.describe().T
    desc["skewness"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()

    fig_height = max(3.5, 0.45 * len(desc) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("tight")
    ax.axis("off")
    cell_text = [[str(x) for x in row] for row in desc.round(2).to_numpy()]
    col_labels = list(desc.columns)
    row_labels = [str(r) for r in desc.index]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.4)

    for col_idx in range(len(desc.columns)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white", weight="bold")
    for row_idx in range(1, len(desc) + 1):
        cell = table[(row_idx, -1)]
        cell.set_facecolor("#f8fafc")
        cell.set_text_props(weight="semibold", color="#334155")

    ax.set_title(
        "Numeric Summary Statistics (with Skew & Kurtosis)",
        pad=15,
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
    )
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure | None:
    """Generate a bar chart of missing values percentage if any exist."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None

    pct = (missing / len(df) * 100).round(2)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(missing) + 1.0)))
    bars = ax.barh(missing.index, pct.to_numpy(), color="#ef4444", alpha=0.85, edgecolor="#b91c1c")
    ax.set_xlabel("Missing Values (%)", fontsize=10, fontweight="semibold")
    ax.set_title(
        "Missing Values by Column (%)",
        fontsize=11,
        fontweight="bold",
        color="#1e293b",
        pad=12,
    )

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f" {width:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#1e293b",
            fontweight="semibold",
        )

    ax.set_xlim(0, max(pct.values) * 1.15)
    return fig


def plot_distributions(df: pd.DataFrame, max_cols: int = 50) -> list[tuple[str, plt.Figure]]:
    """Generate independent histogram + KDE figures for each numeric column.

    Returns a list of (col_name, Figure) tuples. All-NaN columns are skipped.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return []

    columns = num_df.columns.tolist()[:max_cols]
    results: list[tuple[str, plt.Figure]] = []

    for col in columns:
        data = num_df[col].dropna()
        if len(data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data,
            kde=True,
            ax=ax,
            color=PRIMARY_COLOR,
            edgecolor="white",
            line_kws={"linewidth": 2, "color": "#1e3a8a"},
            alpha=0.6,
        )

        mean_val = float(data.mean())
        median_val = float(data.median())
        ax.axvline(
            mean_val,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="#16a34a",
            linestyle=":",
            linewidth=1.5,
            label=f"Median: {median_val:.2f}",
        )

        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count / Density", fontsize=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8)
        results.append((col, fig))

    return results


def plot_numeric_boxplots(df: pd.DataFrame, max_cols: int = 50) -> list[tuple[str, plt.Figure]]:
    """Generate independent boxplots for each numeric column for outlier detection."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return []

    columns = num_df.columns.tolist()[:max_cols]
    results: list[tuple[str, plt.Figure]] = []

    for col in columns:
        data = num_df[col].dropna()
        if len(data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.8))
        flier_kws = {
            "marker": "o",
            "markerfacecolor": "#ef4444",
            "markeredgecolor": "none",
            "alpha": 0.6,
        }
        sns.boxplot(
            x=data,
            ax=ax,
            color="#93c5fd",
            fliersize=4,
            flierprops=flier_kws,
            boxprops={"edgecolor": "#1d4ed8", "linewidth": 1.2},
            whiskerprops={"color": "#1d4ed8", "linewidth": 1.2},
            capprops={"color": "#1d4ed8", "linewidth": 1.2},
            medianprops={"color": "#b91c1c", "linewidth": 2},
        )

        q25 = float(data.quantile(0.25))
        q75 = float(data.quantile(0.75))
        iqr = q75 - q25
        outliers_count = ((data < (q25 - 1.5 * iqr)) | (data > (q75 + 1.5 * iqr))).sum()

        ax.set_title(
            f"Outlier Boxplot: {col} ({outliers_count} outliers)",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_xlabel(col, fontsize=10)
        results.append((col, fig))

    return results


def plot_categorical_bars(
    df: pd.DataFrame,
    max_categories: int = 15,
    id_cols: list[str] | None = None,
) -> list[plt.Figure]:
    """Generate independent bar charts for each categorical column.

    Percentages are computed against non-null count and exclude NaN from bars.
    """
    cat_cols = get_categorical_columns(df, max_cardinality=max_categories * 2, id_cols=id_cols)
    if not cat_cols:
        return []

    figures: list[plt.Figure] = []
    palette = sns.color_palette("mako", n_colors=max_categories)

    for col in cat_cols:
        clean = df[col].dropna()
        total_non_null = len(clean)
        if total_non_null == 0:
            continue

        value_counts = clean.astype(str).value_counts().head(max_categories)
        if value_counts.empty:
            continue

        missing_count = df[col].isna().sum()
        fig_height = max(3.5, 0.35 * len(value_counts) + 1.0)
        fig, ax = plt.subplots(figsize=(6.5, fig_height))

        y_labels = [str(idx) for idx in value_counts.index]
        bars = ax.barh(
            y_labels,
            value_counts.to_numpy(),
            color=palette[: len(value_counts)],
            edgecolor="#334155",
            linewidth=0.5,
        )

        for bar in bars:
            width = bar.get_width()
            pct = (width / total_non_null * 100) if total_non_null > 0 else 0.0
            ax.annotate(
                f" {width:,} ({pct:.1f}%)",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#0f172a",
                fontweight="semibold",
            )

        title_suffix = f", {missing_count} missing" if missing_count > 0 else ""
        ax.set_title(
            f"Frequency: {col} (Top {len(value_counts)}{title_suffix})",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_xlabel("Count", fontsize=10)
        ax.set_xlim(0, max(value_counts.values) * 1.25)
        figures.append(fig)

    return figures


def plot_correlation_heatmap(df: pd.DataFrame, max_correlated: int = 25) -> plt.Figure:
    """Generate a standalone correlation heatmap for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5,
            0.5,
            "At least 2 numeric columns required for correlation",
            ha="center",
            va="center",
            color="#64748b",
        )
        ax.axis("off")
        return fig

    cols = num_df.columns
    if len(cols) > max_correlated:
        variances = num_df.var().sort_values(ascending=False)
        cols = variances.head(max_correlated).index

    corr = num_df[cols].corr()

    fig_size = max(6.5, min(14, 0.6 * len(cols) + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.8,
        linecolor="#f8fafc",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title(
        "Numeric Feature Correlation Matrix",
        pad=15,
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
    )
    return fig


def plot_pairwise_scatter(
    df: pd.DataFrame, max_plots: int = 50, sample_size: int = 1500
) -> list[plt.Figure]:
    """Generate independent scatter plots comparing pairs of numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        return []

    num_cols = num_df.columns.tolist()
    sample_df = (
        df[num_cols].sample(n=sample_size, random_state=42)
        if len(df) > sample_size
        else num_df
    )

    corr_matrix = sample_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    stacked = upper.unstack()
    sorted_pairs = stacked.dropna().sort_values(ascending=False)  # type: ignore[call-overload]
    selected_pairs = sorted_pairs.head(max_plots)

    figures: list[plt.Figure] = []

    for col1, col2 in selected_pairs.index:
        sub_data = sample_df[[col1, col2]].dropna()
        if len(sub_data) < 3:
            continue

        raw_r = float(sub_data[col1].corr(sub_data[col2]))

        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.regplot(
            data=sub_data,
            x=col1,
            y=col2,
            ax=ax,
            color=PRIMARY_COLOR,
            scatter_kws={"alpha": 0.45, "s": 25, "edgecolor": "none"},
            line_kws={"color": "#dc2626", "linewidth": 1.8},
        )

        ax.set_title(
            f"{col1} vs {col2} (r = {raw_r:+.2f})",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_xlabel(col1, fontsize=10)
        ax.set_ylabel(col2, fontsize=10)
        figures.append(fig)

    return figures


_pair_score_cache: dict[str, float] = {}


def compute_eta_squared(df: pd.DataFrame, cat_col: str, num_col: str) -> float:
    """Compute eta-squared (between-group variance / total variance). Cached by shape."""
    cache_key = f"eta2_{cat_col}_{num_col}_{len(df)}_{df.shape[1]}"
    if cache_key in _pair_score_cache:
        return _pair_score_cache[cache_key]

    sub = df[[cat_col, num_col]].dropna()
    if len(sub) < 5 or sub[cat_col].nunique() < 2:
        _pair_score_cache[cache_key] = -1.0
        return -1.0

    y = sub[num_col].to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    total_ss = float(np.sum((y - y_mean) ** 2))
    if total_ss <= 0:
        _pair_score_cache[cache_key] = 0.0
        return 0.0

    grouped = sub.groupby(cat_col)[num_col]
    group_means = grouped.mean().to_numpy(dtype=float)
    group_counts = grouped.count().to_numpy(dtype=float)
    between_ss = float(np.sum(group_counts * ((group_means - y_mean) ** 2)))
    score = float(np.clip(between_ss / total_ss, 0.0, 1.0))
    _pair_score_cache[cache_key] = score
    return score


def compute_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Compute Cramer's V association score for two categorical variables."""
    cache_key = f"cramers_{col1}_{col2}_{len(df)}_{df.shape[1]}"
    if cache_key in _pair_score_cache:
        return _pair_score_cache[cache_key]

    sub = df[[col1, col2]].dropna()
    if len(sub) < 5 or sub[col1].nunique() < 2 or sub[col2].nunique() < 2:
        _pair_score_cache[cache_key] = -1.0
        return -1.0

    ct = pd.crosstab(sub[col1], sub[col2])
    r, k = ct.shape
    if min(r, k) < 2:
        _pair_score_cache[cache_key] = -1.0
        return -1.0

    observed = ct.to_numpy(dtype=float)
    n = float(observed.sum())
    if n <= 0:
        _pair_score_cache[cache_key] = 0.0
        return 0.0

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / n
    terms = ((observed - expected) ** 2) / np.where(expected > 0, expected, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum(np.nan_to_num(terms)))

    denom = n * (min(r, k) - 1)
    score = (
        0.0
        if denom <= 0 or chi2 <= 0
        else float(np.clip(np.sqrt(chi2 / denom), 0.0, 1.0))
    )

    _pair_score_cache[cache_key] = score
    return score


def plot_cat_num_relationships(
    df: pd.DataFrame,
    max_pairs: int = 60,
    sample_size: int = 2500,
    id_cols: list[str] | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Generate independent figures comparing numeric metrics across categoricals."""
    excluded = set(id_cols or [])
    cat_cols = get_categorical_columns(df, max_cardinality=15, id_cols=id_cols)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]

    if not cat_cols or not num_cols:
        return []

    plot_data = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    # Rank candidate pairs by eta-squared
    scored_candidates = []
    for cat_col in cat_cols:
        for num_col in num_cols:
            score = compute_eta_squared(plot_data, cat_col, num_col)
            if score >= 0:
                scored_candidates.append((score, cat_col, num_col))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:max_pairs]

    results: list[tuple[str, plt.Figure]] = []
    for score, cat_col, num_col in top_candidates:
        sub = plot_data[[cat_col, num_col]].dropna()
        if len(sub) < 5 or sub[cat_col].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        top_cats = sub[cat_col].value_counts().head(8).index
        filtered = sub[sub[cat_col].isin(top_cats)]

        sns.boxplot(
            data=filtered,
            x=cat_col,
            y=num_col,
            hue=cat_col,
            legend=False,
            ax=ax,
            palette="Blues",
            boxprops={"edgecolor": "#1e3a8a", "linewidth": 1},
            medianprops={"color": "#dc2626", "linewidth": 1.5},
        )

        ax.set_title(
            f"{num_col} by {cat_col} (η² = {score:.2f})",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_xlabel(cat_col, fontsize=10)
        ax.set_ylabel(num_col, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        label = f"{num_col}_by_{cat_col}"
        results.append((label, fig))

    return results


def plot_cat_cat_relationships(
    df: pd.DataFrame,
    max_pairs: int = 30,
    id_cols: list[str] | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Generate independent figures showing interactions between categorical columns."""
    cat_cols = get_categorical_columns(df, max_cardinality=10, id_cols=id_cols)

    if len(cat_cols) < 2:
        return []

    # Rank candidate pairs by Cramer's V
    scored_candidates = []
    for col1, col2 in itertools.combinations(cat_cols, 2):
        score = compute_cramers_v(df, col1, col2)
        if score >= 0:
            scored_candidates.append((score, col1, col2))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:max_pairs]

    results: list[tuple[str, plt.Figure]] = []
    for score, col1, col2 in top_candidates:
        sub = df[[col1, col2]].dropna()
        if len(sub) < 5 or sub[col1].nunique() < 2 or sub[col2].nunique() < 2:
            continue

        ct = pd.crosstab(sub[col1].astype(str), sub[col2].astype(str), normalize="index") * 100
        if ct.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ct.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            colormap="viridis",
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(
            f"{col1} vs {col2} (Cramér's V = {score:.2f})",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_xlabel(col1, fontsize=10)
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.legend(title=col2, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        label = f"{col1}_vs_{col2}"
        results.append((label, fig))
    return results