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
    page_title="AutoInsight — Automated Data Analyst",
    page_icon="📊",
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

def render_sidebar(profiling: dict) -> dict:
    """Render the sidebar and return user choices.

    Returns a dict with:
    - target_col: selected target column name
    - numeric_cols: list of numeric column names
    - categorical_cols: list of categorical column names
    """
    st.sidebar.title("AutoInsight Controls")

    # Dataset info overview
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"Rows: {profiling.get('rows', 0):,}")
    st.sidebar.write(f"Columns: {profiling.get('columns', 0)}")
    st.sidebar.write(f"Memory: {profiling.get('memory_mb', 0)} MB")

    # Separate columns by type for sensible defaults
    all_cols = profiling.get("column_types", {})
    if not all_cols:
        st.sidebar.warning("No columns available yet — waiting for CSV upload.")
        return {"target_col": None, "numeric_cols": [], "categorical_cols": []}

    numeric_cols = [c for c, t in all_cols.items() if t == "numeric"]
    categorical_cols = [c for c, t in all_cols.items() if t in ("categorical", "high_cardinality_text")]
    datetime_cols = [c for c, t in all_cols.items() if t == "datetime"]
    boolean_cols = [c for c, t in all_cols.items() if t == "boolean"]

    target_col = st.sidebar.selectbox(
        "Select target column (for relationship analysis)",
        options=["(none)"] + numeric_cols + categorical_cols + datetime_cols + boolean_cols,
        index=0,
    )

    if target_col == "(none)":
        return {"target_col": None, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}

    return {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def display_matplotlib_fig(fig, title: str):
    """Display a matplotlib figure with title in Streamlit."""
    st.markdown(f"### {title}")
    st.pyplot(fig, use_container_width=True)


def display_plotly_fig(fig, title: str):
    """Display a plotly figure with title in Streamlit."""
    st.markdown(f"### {title}")
    st.plotly_chart(fig, use_container_width=True)


# Main app entry point
def main():
    """Main entry point for the AutoInsight Streamlit app."""
    # If no data profiling, show welcome message and file uploader
    if st.session_state.df_profiling is None:
        st.title("📊 AutoInsight")
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

    # Render sidebar
    state = render_sidebar(profiling)
    target_col = state.get("target_col")
    numeric_cols = state.get("numeric_cols", [])
    categorical_cols = state.get("categorical_cols", [])

    # ---- Header area ----
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Upload New Dataset"):
        st.session_state.app_flow = "welcome"
        st.session_state.df_raw = None
        st.session_state.df_profiling = None
        st.rerun()

    # ---- Main content area based on whether target is selected ----
    if target_col is None or target_col == "(none)":
        # Show overview without target selection
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

        # Show available charts even without target
        st.markdown("### Available Charts")
        st.markdown("The following charts can be generated from your data:")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("1. **Summary Statistics** - Table with mean, std, min, max, skew, kurtosis")
            st.markdown("2. **Distribution Plots** - Histograms with KDE for numeric columns")
            st.markdown("3. **Categorical Bar Charts** - Top categories for categorical columns")
        with chart_cols[1]:
            st.markdown("4. **Correlation Heatmap** - Pearson correlation matrix heatmap")
            st.markdown("5. **Pairwise Scatter Plots** - Top correlated numeric pairs")
            st.markdown("6. **Target Relationships** - Available after selecting a target column")

        # Generate and display charts without target
        st.markdown("---")
        st.header("Generated Charts")

        # Chart 1: Summary Statistics
        with st.expander("Summary Statistics", expanded=True):
            fig = plot_summary_statistics(df_raw)
            display_matplotlib_fig(fig, "Summary Statistics Table")

        # Chart 2: Distribution plots
        with st.expander("Distribution Plots"):
            dist_figs = plot_distributions(df_raw)
            if dist_figs:
                for i, fig in enumerate(dist_figs):
                    display_matplotlib_fig(fig, f"Distribution Plot {i+1}")
            else:
                st.info("No numeric columns found for distribution plots.")

        # Chart 3: Categorical bar charts
        with st.expander("Categorical Bar Charts"):
            bar_figs = plot_categorical_bars(df_raw)
            if bar_figs:
                for i, fig in enumerate(bar_figs):
                    display_matplotlib_fig(fig, f"Categorical Bar Chart {i+1}")
            else:
                st.info("No categorical columns found for bar charts.")

        # Chart 4: Correlation heatmap
        with st.expander("Correlation Heatmap"):
            fig = plot_correlation_heatmap(df_raw)
            display_matplotlib_fig(fig, "Correlation Heatmap")

        # Chart 5: Pairwise scatter plots
        with st.expander("Pairwise Scatter Plots"):
            pair_figs = plot_pairwise_scatter(df_raw)
            if pair_figs:
                for i, fig in enumerate(pair_figs):
                    display_matplotlib_fig(fig, f"Scatter Plot {i+1}")
            else:
                st.info("Not enough numeric columns for pairwise scatter plots.")

    else:
        # Target column selected - show full EDA interface
        st.sidebar.success(f"Target column: {target_col}")

        tab_names = [
            "Overview",
            "Distributions",
            "Categorical Charts",
            "Correlations",
            "Scatter Plots",
            "Target Relationships",
        ]
        tabs = st.tabs(tab_names)

        with tabs[0]:  # Overview
            st.header("Dataset Overview")
            st.write(f"**Rows:** {profiling.get('rows', 0):,}")
            st.write(f"**Columns:** {profiling.get('columns', 0)}")
            st.write(f"**Memory Usage:** {profiling.get('memory_mb', 0)} MB")
            st.write(f"**Duplicate Rows:** {profiling.get('duplicate_rows', 0)}")
            st.write(f"**Missing Values %:** {profiling.get('missing_percent', {})}")

        with tabs[1]:  # Distributions
            st.header("Distribution Plots (Numeric)")
            st.markdown("Histograms with KDE for numeric columns")
            dist_figs = plot_distributions(df_raw)
            if dist_figs:
                for i, fig in enumerate(dist_figs):
                    display_matplotlib_fig(fig, f"Distribution Plot {i+1}")
            else:
                st.info("No numeric columns found.")

            # Also show plotly versions
            st.markdown("**Plotly Interactive Versions:**")
            numeric_cols_available = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
            for col in numeric_cols_available[:5]:  # Limit to first 5 for performance
                fig = plotly_distribution_histogram(df_raw, col)
                display_plotly_fig(fig, f"Distribution: {col}")

        with tabs[2]:  # Categorical Charts
            st.header("Categorical Bar Charts")
            st.markdown("Top categories for categorical columns")
            bar_figs = plot_categorical_bars(df_raw)
            if bar_figs:
                for i, fig in enumerate(bar_figs):
                    display_matplotlib_fig(fig, f"Categorical Bar Chart {i+1}")
            else:
                st.info("No categorical columns found for bar charts.")

            # Plotly versions
            st.markdown("**Plotly Interactive Versions:**")
            for col in categorical_cols[:5]:
                value_counts = df_raw[col].value_counts(dropna=False).head(10)
                fig = px.bar(
                    x=value_counts.values,
                    y=[str(idx) for idx in value_counts.index],
                    orientation="h",
                    title=f"Categorical: {col}",
                )
                fig.update_layout(template="plotly_white")
                display_plotly_fig(fig, f"Categorical: {col}")

        with tabs[3]:  # Correlations
            st.header("Correlation Analysis")
            st.markdown("**Matplotlib Correlation Heatmap:**")
            fig = plot_correlation_heatmap(df_raw)
            display_matplotlib_fig(fig, "Correlation Heatmap")

            st.markdown("**Plotly Correlation Heatmap:**")
            fig_plotly = plotly_correlation_heatmap(df_raw)
            display_plotly_fig(fig_plotly, "Correlation Heatmap (Plotly)")

            # Top correlated pairs info
            st.markdown("**Top Correlated Pairs:**")
            num_df = df_raw.select_dtypes(include=[np.number])
            if not num_df.empty and len(num_df.columns) >= 2:
                corr_matrix = num_df.corr().abs()
                upper = corr_matrix.where(np.tri(len(corr_matrix), k=1, dtype=bool))
                stacked = upper.unstack()
                sorted_pairs = stacked.dropna().sort_values(ascending=False)
                top_pairs = sorted_pairs.head(10)
                for (c1, c2), val in top_pairs.items():
                    st.write(f"- `{c1}` vs `{c2}`: r = {val:.3f}")

        with tabs[4]:  # Scatter Plots
            st.header("Pairwise Scatter Plots")
            st.markdown("**Matplotlib Top Correlated Pairs:**")
            pair_figs = plot_pairwise_scatter(df_raw)
            if pair_figs:
                for i, fig in enumerate(pair_figs):
                    display_matplotlib_fig(fig, f"Scatter Plot {i+1}")
            else:
                st.info("Not enough numeric columns for pairwise scatter plots.")

            st.markdown("**Plotly Interactive Scatter Plots:**")
            pair_figs_plotly = plotly_pairwise_scatter(df_raw)
            if pair_figs_plotly:
                for i, fig in enumerate(pair_figs_plotly):
                    display_plotly_fig(fig, f"Interactive Scatter Plot {i+1}")
            else:
                st.info("Not enough numeric columns for plotly scatter plots.")

        with tabs[5]:  # Target Relationships
            st.header("Target vs Feature Relationships")
            st.markdown(f"**Target Column:** {target_col}")

            target_figs = plot_target_relationships(df_raw, target_col)
            if target_figs:
                for i, fig in enumerate(target_figs):
                    display_matplotlib_fig(fig, f"Target Relationship {i+1}")
            else:
                st.info("No target relationship plots could be generated.")

            # Plotly versions
            st.markdown("**Plotly Interactive Target Relationships:**")
            plotly_figs = plotly_target_relationships(df_raw, target_col)
            if plotly_figs:
                for i, fig in enumerate(plotly_figs):
                    display_plotly_fig(fig, f"Interactive Target Relationship {i+1}")
            else:
                st.info("No plotly target relationship plots available.")


if __name__ == "__main__":
    main()