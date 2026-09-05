"""AutoInsight — Automated Data Analysis Streamlit App.

A locally-hostable web app that automatically explores and visualizes
CSV datasets with attractive charts using matplotlib, seaborn, and plotly.
"""

from __future__ import annotations

import os
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np

# Import local modules
from modules.data_loader import load_csv, infer_column_type, profile_dataset
from modules.eda import (
    plot_summary_statistics,
    plot_distributions,
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_pairwise_scatter,
    plot_target_relationships,
    plotly_distribution_histogram,
    plotly_correlation_heatmap,
    plotly_pairwise_scatter,
    plotly_target_relationships,
)


# Page configuration
st.set_page_config(
    page_title="AutoInsight Automated Data Analyst",
    page_icon="",
    layout="wide",
)


# Initialize session state for app flow
if "app_flow" not in st.session_state:
    st.session_state.app_flow = "welcome"  # welcome | loaded
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None  # Store raw DataFrame
if "df_profiling" not in st.session_state:
    st.session_state.df_profiling = None  # Store profiling dict


# ---- Cached utilities ----

@st.cache_data(show_spinner="Loading and profiling dataset...")
def cached_load_and_profile(uploaded_file):
    """Load CSV and compute profiling dict. Cached per uploaded file."""
    df = load_csv(uploaded_file)
    profiling = profile_dataset(df)
    return df, profiling


# ---- Helper functions ----




def display_matplotlib_fig(fig, title: str):
    """Display a matplotlib figure with title in Streamlit."""
    st.markdown(f"### {title}")
    st.pyplot(fig, use_container_width=False)


def display_plotly_fig(fig, title: str):
    """Display a plotly figure with title in Streamlit."""
    st.markdown(f"### {title}")
    st.plotly_chart(fig, use_container_width=False)


# Main app entry point
def main():
    """Main entry point for the AutoInsight Streamlit app."""
    # If no data profiling, show welcome message and file uploader
    if st.session_state.df_profiling is None:
        st.title("AutoInsight Automated Data Analyst")
        st.markdown("### Automated Data Analyst")
        st.markdown("---")
        st.markdown(
            "Welcome! Upload any CSV file and let AutoInsight automatically "
            "explore and visualize your data with attractive charts."
        )

        st.markdown("**Get started by uploading a CSV file:**")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file (up to 50MB) to analyze automatically",
            label_visibility="visible",
        )

        if uploaded_file is not None:
            df, profiling = cached_load_and_profile(uploaded_file)
            st.session_state.df_raw = df
            st.session_state.df_profiling = profiling
            st.session_state.app_flow = "loaded"
            st.rerun()
        return

    # Data is loaded - proceed with normal app flow
    df_raw = st.session_state.df_raw
    profiling = st.session_state.df_profiling

    # Show overview and all charts automatically
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{profiling.get('rows', 0):,}")
    with col2:
        st.metric("Columns", profiling.get('columns', 0))
    with col3:
        st.metric("Memory (MB)", f"{profiling.get('memory_mb', 0):.2f}")
    with col4:
        st.metric("Duplicates", profiling.get('duplicate_rows', 0))

    st.markdown("**Column Types:**")
    col_types = profiling.get("column_types", {})
    if col_types:
        type_counts = {}
        for t in col_types.values():
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, count in type_counts.items():
            st.write(f"- {t}: {count}")

    # Show available charts
    st.markdown("### Available Charts")
    st.markdown("The following charts are generated from your data:")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("1. **Summary Statistics** - Table with mean, std, min, max, skew, kurtosis")
        st.markdown("2. **Distribution Plots** - Histograms with KDE for numeric columns")
        st.markdown("3. **Categorical Bar Charts** - Top categories for categorical columns")
    with chart_cols[1]:
        st.markdown("4. **Correlation Heatmap** - Pearson correlation matrix heatmap")
        st.markdown("5. **Pairwise Scatter Plots** - Top correlated numeric pairs")

    # Generate and display charts in a grid layout (2 plots per row)
    st.markdown("---")

    # Row 1: Summary Statistics + Distribution Plots
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Summary Statistics", expanded=True):
            fig = plot_summary_statistics(df_raw)
            display_matplotlib_fig(fig, "Summary Statistics Table")
    with col2:
        with st.expander("Distribution Plots"):
            dist_figs = plot_distributions(df_raw)
            if dist_figs:
                for i, fig in enumerate(dist_figs):
                    display_matplotlib_fig(fig, f"Distribution Plot {i+1}")
            else:
                st.info("No numeric columns found for distribution plots.")

    # Row 2: Categorical Bar Charts + Correlation Heatmap
    col3, col4 = st.columns(2)
    with col3:
        with st.expander("Categorical Bar Charts"):
            bar_figs = plot_categorical_bars(df_raw)
            if bar_figs:
                # Display charts in pairs per row
                for i in range(0, len(bar_figs), 2):
                    pair = bar_figs[i:i+2]
                    row_cols = st.columns(len(pair))
                    for j, fig in enumerate(pair):
                        with row_cols[j]:
                            display_matplotlib_fig(fig, f"Categorical Bar Chart {i+j+1}")
            else:
                st.info("No categorical columns found for bar charts.")
    with col4:
        with st.expander("Correlation Heatmap"):
            fig = plot_correlation_heatmap(df_raw)
            display_matplotlib_fig(fig, "Correlation Heatmap")

    # Row 3: Pairwise Scatter Plots
    col5, col6 = st.columns(2)
    with col5:
        with st.expander("Pairwise Scatter Plots"):
            pair_figs = plot_pairwise_scatter(df_raw)
            if pair_figs:
                for i, fig in enumerate(pair_figs):
                    display_matplotlib_fig(fig, f"Scatter Plot {i+1}")
            else:
                st.info("Not enough numeric columns for pairwise scatter plots.")


if __name__ == "__main__":
    main()