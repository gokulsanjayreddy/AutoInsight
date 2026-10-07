"""Automated EDA analysis pipeline for AutoInsight.

This module provides the pure execution engine for dataset profiling,
figure generation, PNG rendering, and ZIP archiving. It contains no
Streamlit dependencies or UI code.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

from modules.data_loader import profile_dataset
from modules.eda import (
    create_all_plots_zip,
    fig_to_png_bytes,
    plot_cat_cat_relationships,
    plot_cat_num_relationships,
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_distributions,
    plot_missing_values,
    plot_numeric_boxplots,
    plot_pairwise_scatter,
    plot_summary_statistics,
)
from modules.zip_utils import sanitize_filename_stem


class PlotCollector:
    """Collects and encodes plots up to a specified maximum count."""

    def __init__(self, max_total_plots: int = 100) -> None:
        self.max_total_plots = max_total_plots
        self.total_count = 0
        self.all_named_figures: list[tuple[str, str, bytes]] = []
        self.card_data: dict[str, list[tuple[str, str, bytes]]] = {
            "summary": [],
            "missing": [],
            "dists": [],
            "boxplots": [],
            "cats": [],
            "corr": [],
            "scatters": [],
            "cat_num": [],
            "cat_cat": [],
        }

    def has_capacity(self) -> bool:
        """Return True if collector can accept more plots."""
        return self.total_count < self.max_total_plots

    def add(
        self,
        section_key: str,
        folder: str,
        file_basename: str,
        title: str,
        download_filename: str,
        fig: plt.Figure,
    ) -> bool:
        """Encode figure to PNG, add to collection, and close figure immediately."""
        if self.total_count >= self.max_total_plots:
            plt.close(fig)
            return False

        png = fig_to_png_bytes(fig)
        plt.close(fig)

        safe_dl_name = sanitize_filename_stem(download_filename, max_len=60, fallback="plot")
        self.all_named_figures.append((folder, file_basename, png))
        self.card_data[section_key].append((title, safe_dl_name, png))
        self.total_count += 1
        return True


def _collect_summary_and_missing(collector: PlotCollector, df: pd.DataFrame) -> None:
    """Generate summary statistics and missing value plots."""
    fig_summary = plot_summary_statistics(df)
    collector.add(
        "summary",
        "00_summary",
        "summary_statistics_table",
        "Summary Statistics Table",
        "summary_statistics",
        fig_summary,
    )

    fig_missing = plot_missing_values(df)
    if fig_missing is not None:
        collector.add(
            "missing",
            "00_summary",
            "missing_values_breakdown",
            "Missing Values Breakdown (%)",
            "missing_values",
            fig_missing,
        )


def _collect_numeric_distributions(collector: PlotCollector, df: pd.DataFrame) -> None:
    """Generate histograms and boxplots for numeric features."""
    if not collector.has_capacity():
        return

    dist_pairs = plot_distributions(df)
    for i, (col, fig) in enumerate(dist_pairs):
        if not collector.add(
            "dists",
            "01_univariate_numeric_distributions",
            f"dist_{col}",
            f"Distribution: {col}",
            f"distribution_{col}",
            fig,
        ):
            for _, rem in dist_pairs[i + 1 :]:
                plt.close(rem)
            break

    if not collector.has_capacity():
        return

    box_pairs = plot_numeric_boxplots(df)
    for i, (col, fig) in enumerate(box_pairs):
        if not collector.add(
            "boxplots",
            "02_outlier_boxplots",
            f"boxplot_{col}",
            f"Boxplot: {col}",
            f"boxplot_{col}",
            fig,
        ):
            for _, rem in box_pairs[i + 1 :]:
                plt.close(rem)
            break


def _collect_categorical_and_corr(
    collector: PlotCollector, df: pd.DataFrame, id_cols: list[str]
) -> None:
    """Generate categorical bar charts and correlation matrix."""
    if collector.has_capacity():
        cat_figs = plot_categorical_bars(df, id_cols=id_cols)
        for i, fig in enumerate(cat_figs):
            title = fig.axes[0].get_title() if fig.axes else f"Categorical Distribution {i+1}"
            if not collector.add(
                "cats",
                "03_categorical_distributions",
                f"cat_distribution_{i+1}",
                title,
                f"cat_dist_{i+1}",
                fig,
            ):
                for rem in cat_figs[i + 1 :]:
                    plt.close(rem)
                break

    if collector.has_capacity():
        fig_corr = plot_correlation_heatmap(df)
        collector.add(
            "corr",
            "04_correlation_heatmap",
            "correlation_matrix",
            "Correlation Matrix",
            "correlation_heatmap",
            fig_corr,
        )


def _collect_scatters(collector: PlotCollector, df: pd.DataFrame) -> None:
    """Generate bivariate scatter plots for correlated numeric pairs."""
    if not collector.has_capacity():
        return
    scatters = plot_pairwise_scatter(df)
    for i, fig in enumerate(scatters):
        title = fig.axes[0].get_title() if fig.axes else f"Scatter Plot {i+1}"
        if not collector.add(
            "scatters",
            "05_numeric_relationships",
            f"scatter_pair_{i+1}",
            title,
            f"scatter_pair_{i+1}",
            fig,
        ):
            for rem in scatters[i + 1 :]:
                plt.close(rem)
            break


def _collect_cat_num(
    collector: PlotCollector, df: pd.DataFrame, id_cols: list[str]
) -> None:
    """Generate categorical vs numeric boxplots ranked by eta-squared."""
    if not collector.has_capacity():
        return
    cat_num = plot_cat_num_relationships(df, id_cols=id_cols)
    for i, (label, fig) in enumerate(cat_num):
        title = fig.axes[0].get_title() if fig.axes else f"Grouped Comparison: {label}"
        if not collector.add(
            "cat_num",
            "06_cat_vs_numeric",
            f"cat_num_{label}",
            title,
            f"cat_num_{label}",
            fig,
        ):
            for _, rem in cat_num[i + 1 :]:
                plt.close(rem)
            break


def _collect_cat_cat(
    collector: PlotCollector, df: pd.DataFrame, id_cols: list[str]
) -> None:
    """Generate categorical cross-tabulation bar charts ranked by Cramer's V."""
    if not collector.has_capacity():
        return
    cat_cat = plot_cat_cat_relationships(df, id_cols=id_cols)
    for i, (label, fig) in enumerate(cat_cat):
        title = fig.axes[0].get_title() if fig.axes else f"Cross-Tabulation: {label}"
        if not collector.add(
            "cat_cat",
            "07_cat_vs_cat",
            f"cat_cat_{label}",
            title,
            f"cat_cat_{label}",
            fig,
        ):
            for _, rem in cat_cat[i + 1 :]:
                plt.close(rem)
            break


def _collect_relationship_plots(
    collector: PlotCollector, df: pd.DataFrame, id_cols: list[str]
) -> None:
    """Generate bivariate scatters, cat-num boxplots, and cat-cat bars."""
    _collect_scatters(collector, df)
    _collect_cat_num(collector, df, id_cols)
    _collect_cat_cat(collector, df, id_cols)


def generate_full_analysis(
    df: pd.DataFrame,
    max_total_plots: int = 100,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Generate all figures and prepare in-memory ZIP package for download."""
    collector = PlotCollector(max_total_plots=max_total_plots)

    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(fraction, message)

    report(0.05, "Computing dataset profile and summary statistics...")
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
    profiling = profile_dataset(df)
    id_cols = profiling.get("id_columns", [])

    _collect_summary_and_missing(collector, df)

    report(0.20, "Generating numeric distribution histograms and boxplots...")
    _collect_numeric_distributions(collector, df)

    report(0.50, "Generating categorical frequency charts and correlation...")
    _collect_categorical_and_corr(collector, df, id_cols)

    report(0.75, "Analyzing bivariate relationships and interactions...")
    _collect_relationship_plots(collector, df, id_cols)

    plt.close("all")

    report(0.97, "Packaging figures into ZIP archive...")
    zip_bytes = create_all_plots_zip(collector.all_named_figures)

    report(1.0, "Analysis complete.")
    return {
        "card_data": collector.card_data,
        "zip_bytes": zip_bytes,
        "total_count": collector.total_count,
        "profiling": profiling,
    }
