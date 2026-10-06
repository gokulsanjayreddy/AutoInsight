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


def generate_full_analysis(
    df: pd.DataFrame,
    max_total_plots: int = 100,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Generate all figures and prepare in-memory ZIP package for download.

    Pure function returning plain data (sections, PNG bytes, zip bytes) with
    no Streamlit imports. Renders each figure to PNG exactly once and closes
    all figures immediately to avoid memory leaks.
    """
    total_plots_count = 0
    all_named_figures: list[tuple[str, str, bytes]] = []
    card_data: dict[str, list[tuple[str, str, bytes]]] = {
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

    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(fraction, message)

    report(0.05, "Computing dataset profile and summary statistics...")

    profiling = profile_dataset(df)
    id_cols = profiling.get("id_columns", [])

    def add_plot(
        section_key: str,
        folder: str,
        file_basename: str,
        title: str,
        download_filename: str,
        fig: plt.Figure,
    ) -> bool:
        nonlocal total_plots_count
        if total_plots_count >= max_total_plots:
            plt.close(fig)
            return False

        png = fig_to_png_bytes(fig)
        plt.close(fig)

        all_named_figures.append((folder, file_basename, png))
        card_data[section_key].append((title, download_filename, png))
        total_plots_count += 1
        return True

    # 00. Summary Statistics & Missing Values
    fig_summary = plot_summary_statistics(df)
    add_plot(
        "summary",
        "00_summary",
        "summary_statistics_table",
        "Summary Statistics Table",
        "summary_statistics",
        fig_summary,
    )

    fig_missing = plot_missing_values(df)
    if fig_missing is not None:
        add_plot(
            "missing",
            "00_summary",
            "missing_values_breakdown",
            "Missing Values Breakdown (%)",
            "missing_values",
            fig_missing,
        )

    # 01. Univariate Numeric Distributions
    if total_plots_count < max_total_plots:
        report(0.20, "Generating numeric distribution histograms...")
        dist_pairs = plot_distributions(df)
        for i, (col_name, fig) in enumerate(dist_pairs):
            if not add_plot(
                "dists",
                "01_univariate_numeric_distributions",
                f"dist_{col_name}",
                f"Distribution: {col_name}",
                f"distribution_{col_name}",
                fig,
            ):
                for _, remaining_fig in dist_pairs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # 02. Outlier Boxplots
    if total_plots_count < max_total_plots:
        report(0.35, "Generating outlier detection boxplots...")
        boxplot_pairs = plot_numeric_boxplots(df)
        for i, (col_name, fig) in enumerate(boxplot_pairs):
            if not add_plot(
                "boxplots",
                "02_outlier_boxplots",
                f"boxplot_{col_name}",
                f"Boxplot: {col_name}",
                f"boxplot_{col_name}",
                fig,
            ):
                for _, remaining_fig in boxplot_pairs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # 03. Categorical Distributions
    if total_plots_count < max_total_plots:
        report(0.50, "Generating categorical frequency charts...")
        cat_figs = plot_categorical_bars(df, id_cols=id_cols)
        for i, fig in enumerate(cat_figs):
            ax_title = fig.axes[0].get_title() if fig.axes else f"Categorical Distribution {i+1}"
            if not add_plot(
                "cats",
                "03_categorical_distributions",
                f"cat_distribution_{i+1}",
                ax_title,
                f"cat_dist_{i+1}",
                fig,
            ):
                for remaining_fig in cat_figs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # 04. Correlation Heatmap
    if total_plots_count < max_total_plots:
        report(0.65, "Calculating feature correlation matrix...")
        fig_corr = plot_correlation_heatmap(df)
        add_plot(
            "corr",
            "04_correlation_heatmap",
            "correlation_matrix",
            "Correlation Matrix",
            "correlation_heatmap",
            fig_corr,
        )

    # 05. Numeric vs Numeric Relationships
    if total_plots_count < max_total_plots:
        report(0.75, "Generating numeric bivariate scatter plots...")
        scatter_figs = plot_pairwise_scatter(df)
        for i, fig in enumerate(scatter_figs):
            ax_title = fig.axes[0].get_title() if fig.axes else f"Scatter Plot {i+1}"
            if not add_plot(
                "scatters",
                "05_numeric_relationships",
                f"scatter_pair_{i+1}",
                ax_title,
                f"scatter_pair_{i+1}",
                fig,
            ):
                for remaining_fig in scatter_figs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # 06. Categorical vs Numeric Relationships
    if total_plots_count < max_total_plots:
        report(0.85, "Analyzing categorical vs numeric interactions...")
        cat_num_pairs = plot_cat_num_relationships(df, id_cols=id_cols)
        for i, (label, fig) in enumerate(cat_num_pairs):
            ax_title = fig.axes[0].get_title() if fig.axes else f"Grouped Comparison: {label}"
            if not add_plot(
                "cat_num",
                "06_cat_vs_numeric",
                f"cat_num_{label}",
                ax_title,
                f"cat_num_{label}",
                fig,
            ):
                for _, remaining_fig in cat_num_pairs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # 07. Categorical vs Categorical Interactions
    if total_plots_count < max_total_plots:
        report(0.92, "Analyzing categorical cross-tabulations...")
        cat_cat_pairs = plot_cat_cat_relationships(df, id_cols=id_cols)
        for i, (label, fig) in enumerate(cat_cat_pairs):
            ax_title = fig.axes[0].get_title() if fig.axes else f"Cross-Tabulation: {label}"
            if not add_plot(
                "cat_cat",
                "07_cat_vs_cat",
                f"cat_cat_{label}",
                ax_title,
                f"cat_cat_{label}",
                fig,
            ):
                for _, remaining_fig in cat_cat_pairs[i + 1 :]:
                    plt.close(remaining_fig)
                break

    # Ensure all open figures are completely closed
    plt.close("all")

    # ZIP Creation using pre-rendered PNG bytes
    report(0.97, "Packaging figures into ZIP archive...")
    zip_bytes = create_all_plots_zip(all_named_figures)

    report(1.0, "Analysis complete.")

    return {
        "card_data": card_data,
        "zip_bytes": zip_bytes,
        "total_count": total_plots_count,
        "profiling": profiling,
    }
