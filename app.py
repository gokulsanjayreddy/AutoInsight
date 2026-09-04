"""AutoInsight — Automated Data Analysis Streamlit App.

A locally-hostable web app that automatically cleans, explores, and models
CSV datasets without requiring any code from the user.
"""

from __future__ import annotations

import os
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np

# Import local modules
from modules.data_loader import load_csv, infer_column_type, profile_dataset, flag_id_columns
from modules.cleaner import handle_missing_values, detect_and_flag_outliers_iqr, remove_duplicate_rows, coerce_types
from modules.eda import (
    plot_summary_statistics,
    plot_distributions,
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_pairwise_scatter,
    plot_target_relationships,
)
from modules.model_selector import detect_task_type, detect_unusable_columns, prepare_features
from modules.trainer import (
    train_classification_models,
    train_regression_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_predicted_vs_actual,
    plot_feature_importance,
)
from modules.report import generate_report

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
if "cleaning_state" not in st.session_state:
    st.session_state.cleaning_state = {
        "df_original": None,
        "df_clean": None,
        "cleaning_log": [],
        "strategies": {},
        "outlier_counts": {},
        "id_cols": [],
        "target_col": None,
        "task_type": None,
        "model_results": None,
        "report_md": None,
    }

# Always render the sidebar first (Streamlit requires this)
# We check if data is loaded to determine what to show in the main area

# ---- Cached utilities ----

@st.cache_data(show_spinner="Loading and profiling dataset...")
def cached_load_and_profile(uploaded_file):
    """Load CSV and compute profiling dict. Cached per uploaded file."""
    df = load_csv(uploaded_file)
    profiling = profile_dataset(df)
    return df, profiling


@st.cache_data(show_spinner="Training models...")
def cached_train_models(X, y, task_type, test_size=0.2, random_state=42):
    """Cache model training based on data and parameters."""
    if task_type == "classification":
        results = train_classification_models(X, y, test_size, random_state)
    else:
        results = train_regression_models(X, y, test_size, random_state)
    return results


@st.cache_data(show_spinner="Preparing features for modeling...")
def cached_prepare_features(df, target_col, task_type):
    """Prepare features with caching."""
    from modules.model_selector import detect_task_type
    # Always detect task type from the actual target
    y_raw = df[target_col].values
    detected_type = detect_task_type(df[target_col])
    
    # Prepare features
    X, y, feature_names = prepare_features(df, target_col, detected_type)
    
    # Ensure y is numpy array
    y = y.flatten() if hasattr(y, 'flatten') else np.array(y)
    X = X.flatten() if hasattr(X, 'flatten') else np.array(X)
    
    return X, y, feature_names, detected_type


def get_cached_cleaning_state():
    """Return persisted cleaning state across reruns."""
    if "cleaning_state" not in st.session_state:
        st.session_state.cleaning_state = {
            "df_original": None,
            "df_clean": None,
            "cleaning_log": [],
            "strategies": {},
            "outlier_counts": {},
            "id_cols": [],
            "target_col": None,
            "task_type": None,
            "model_results": None,
            "report_md": None,
        }
    return st.session_state.cleaning_state


# ---- Helper functions ----

def render_sidebar(df_profiling: dict) -> dict:
    """Render the sidebar and return user choices.

    Returns a dict with:
    - target_col: selected target column name
    - cleaning_strategies: dict of column -> strategy
    - toggle flags for outlier removal, duplicate removal, etc.
    """
    st.sidebar.title("AutoInsight Controls")

    # Dataset info overview
    st.sidebar.markdown("### Dataset Overview")
    st.sidebar.write(f"Rows: {df_profiling.get('rows', 0):,}")
    st.sidebar.write(f"Columns: {df_profiling.get('columns', 0)}")
    st.sidebar.write(f"Memory: {df_profiling.get('memory_mb', 0)} MB")

    # Target column selector
    all_cols = df_profiling.get("column_types", {})
    if not all_cols:
        st.sidebar.warning("No columns available yet — waiting for CSV upload.")
        return {"target_col": None}

    # Separate columns by type for sensible defaults
    numeric_cols = [c for c, t in all_cols.items() if t == "numeric"]
    categorical_cols = [c for c, t in all_cols.items() if t in ("categorical", "high_cardinality_text")]
    datetime_cols = [c for c, t in all_cols.items() if t == "datetime"]
    boolean_cols = [c for c, t in all_cols.items() if t == "boolean"]

    target_col = st.sidebar.selectbox(
        "Select target column",
        options=["(none)"] + numeric_cols + categorical_cols + datetime_cols + boolean_cols,
        index=0,
    )

    if target_col == "(none)":
        return {"target_col": None, "task_type": None}

    # Detect task type automatically
    from modules.model_selector import detect_task_type
    target_series = pd.Series([])  # we'll set this properly later
    task_type = detect_task_type(pd.Series(dtype=object))  # placeholder

    # Cleaning strategy toggles
    st.sidebar.markdown("### Cleaning Strategies")

    # Missing value strategies
    strategies = st.session_state.cleaning_state.get("strategies", {})

    with st.sidebar.expander("Missing Value Strategies", expanded=True):
        for col in numeric_cols:
            default = "median" if col not in strategies else strategies[col]
            strategies[col] = st.selectbox(
                f"Numeric column '{col}'",
                options=["median", "mean"],
                index=["median", "mean"].index(default) if default in ["median", "mean"] else 0,
                key=f"mv_{col}",
            )
        for col in categorical_cols:
            default = "mode" if col not in strategies else strategies[col]
            strategies[col] = st.selectbox(
                f"Categorical column '{col}'",
                options=["mode", "unknown"],
                index=["mode", "unknown"].index(default) if default in ["mode", "unknown"] else 0,
                key=f"mv_{col}_cat",
            )

    # Outlier handling
    with st.sidebar.expander("Outlier Detection", expanded=True):
        use_iqr = st.checkbox(
            "Detect outliers via IQR method",
            value=True,
            key="iqr_toggle",
        )
        outlier_action = "cap"
        if use_iqr:
            outlier_action = st.radio(
                "Outlier action:",
                options=["cap (winsorize)", "remove", "ignore"],
                index=0,
                key="outlier_action",
            )

    # Duplicate removal
    remove_dups = st.checkbox(
        "Remove duplicate rows",
        value=bool(st.session_state.cleaning_state.get("duplicates_removed", False)),
        key="dups_toggle",
    )

    # Type coercion
    with st.sidebar.expander("Type Coercion", expanded=False):
        coerce_datetime = st.checkbox(
            "Attempt datetime parsing on string columns",
            value=True,
            key="coerce_toggle",
        )

    return {
        "target_col": target_col,
        "task_type": task_type,
        "strategies": strategies,
        "use_iqr": use_iqr,
        "outlier_action": outlier_action,
        "remove_dups": remove_dups,
        "coerce_datetime": coerce_datetime,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def run_cleaning_pipeline(state: dict, df: pd.DataFrame) -> tuple:
    """Run the cleaning pipeline based on user choices.

    Returns (df_cleaned, cleaning_log).
    """
    df_current = df.copy()
    cleaning_log = []

    # 1. Type coercion first
    if state["coerce_datetime"]:
        df_current, coercion_log = coerce_types(df_current)
        cleaning_log.extend(coercion_log)

    # 2. Handle missing values
    # Separate numeric and categorical strategies
    num_strat = {}
    cat_strat = {}

    for col, strat in state["strategies"].items():
        if col in state.get("numeric_cols", []):
            num_strat[col] = strat
        elif col in state.get("categorical_cols", []):
            cat_strat[col] = strat

    # Fill numeric with median by default for any not specified
    for col in state.get("numeric_cols", []):
        if col not in num_strat:
            num_strat[col] = "median"

    for col in state.get("categorical_cols", []):
        if col not in cat_strat:
            cat_strat[col] = "mode"

    if num_strat or cat_strat:
        df_current, ml = handle_missing_values(df_current, {**num_strat, **cat_strat})
        cleaning_log.extend(ml)

    # 3. Outlier detection
    numeric_cols_current = [
        c for c in df_current.columns if pd.api.types.is_numeric_dtype(df_current[c])
    ]

    if state["use_iqr"] and numeric_cols_current:
        df_current, outlier_counts, ol_log = detect_and_flag_outliers_iqr(
            df_current, numeric_cols_current
        )
        cleaning_log.extend(ol_log)
    elif state["use_iqr"] and not numeric_cols_current:
        cleaning_log.append("No numeric columns available for outlier detection.")

    # 4. Remove duplicates
    if state["remove_dups"]:
        df_current, dup_log = remove_duplicate_rows(df_current)
        cleaning_log.extend(dup_log)

    # Update state
    state["df_clean"] = df_current
    state["cleaning_log"] = cleaning_log
    state["strategies"] = {**num_strat, **cat_strat}

    return df_current, cleaning_log


def run_eda_tab(df: pd.DataFrame, target_col: str = None):
    """Render the EDA tab content."""
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary Stats", "Distributions", "Categorical Bar Charts", "Correlations & Pairs"]
    )

    with tab1:
        st.pyplot(plot_summary_statistics(df), use_container_width=True)

    with tab2:
        dist_figs = plot_distributions(df)
        if dist_figs:
            for fig in dist_figs:
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for distribution plots.")

    with tab3:
        bar_figs = plot_categorical_bars(df)
        if bar_figs:
            for fig in bar_figs:
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("No categorical columns found for bar charts.")

    with tab4:
        # Correlation heatmap
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True)

        # Pairwise scatter plots
        pair_figs = plot_pairwise_scatter(df)
        if pair_figs:
            for fig in pair_figs:
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for pairwise scatter plots.")

        # Target relationship plots if target selected
        if target_col and target_col != "(none)":
            st.markdown("### Target vs Feature Relationships")
            target_figs = plot_target_relationships(df, target_col)
            if target_figs:
                for fig in target_figs:
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No target relationship plots could be generated.")


def run_modeling_tab(df: pd.DataFrame, target_col: str, task_type: str, state: dict):
    """Render the Modeling tab content."""
    st.markdown("### Model Training")
    
    # Check if model results are already cached in session state
    current_target = target_col
    cached_results = st.session_state.get("cleaning_state", {}).get("model_results")
    
    # Only retrain if no cached results or target column changed
    if cached_results is None or cached_results.get("target_col") != current_target:
        st.warning("Please select a target column in the sidebar to enable modeling.")
        if not target_col or target_col == "(none)":
            return
        
        with st.spinner("Preparing features and training models..."):
            # Check for usable columns
            from modules.data_loader import profile_dataset
            profiling = profile_dataset(df)
            id_cols, high_missing_cols, constant_cols = detect_unusable_columns(
                df, target_col=current_target
            )
            
            # Display exclusions
            col1, col2, col3 = st.columns(3)
            with col1:
                if id_cols:
                    st.error(f"ID columns excluded: {', '.join(id_cols)}")
                else:
                    st.success("No ID columns detected.")
            with col2:
                if high_missing_cols:
                    st.warning(f"High-missing columns excluded: {', '.join(high_missing_cols)}")
                else:
                    st.success("No high-missing columns detected.")
            with col3:
                if constant_cols:
                    st.warning(f"Constant columns excluded: {', '.join(constant_cols)}")
                else:
                    st.success("No constant columns detected.")
            
            # Check we have features remaining
            numeric_cols = [c for c in df.columns if c != current_target and pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in df.columns if c != current_target and not pd.api.types.is_numeric_dtype(df[c])]
            
            if len(numeric_cols) + len(cat_cols) == 0:
                st.error("No usable features remaining after exclusions. Cannot train models.")
                model_results = None
            else:
                # Prepare features
                X, y, feature_names = prepare_features(df, current_target, task_type)
                
                # Handle case where X might be empty after preprocessing
                if X.shape[1] == 0:
                    st.error("No features remaining after preprocessing. Cannot train models.")
                    model_results = None
                else:
                    # Train models
                    if task_type == "classification":
                        model_results = train_classification_models(X, y, test_size=0.2, random_state=42)
                    else:
                        model_results = train_regression_models(X, y, test_size=0.2, random_state=42)
                    
                    # Store in session state for caching
                    st.session_state.cleaning_state["model_results"] = {
                        "target_col": current_target,
                        "model_results": model_results,
                        "task_type": task_type
                    }
    else:
        # Use cached results - display them
        model_results = cached_results.get("model_results")
        st.success("Using cached model results (faster loading!)")
        
        if model_results:
            # Display results summary
            if state.get("task_type", "") == "classification":
                acc = model_results.get("accuracy", 0) if isinstance(model_results, dict) else 0
                st.info(f"Classification model accuracy: {acc:.4f}")
            else:
                r2 = model_results.get("r2", 0) if isinstance(model_results, dict) else 0
                st.info(f"Regression model R²: {r2:.4f}")

    # Display results
    st.markdown("### Model Comparison")

    if task_type == "classification":
        # Accuracy table
        comp_placeholder = st.container()
        with comp_placeholder:
            for name, res in results.items():
                accuracy = res.get("accuracy", 0)
                precision = res.get("precision", 0)
                recall = res.get("recall", 0)
                f1 = res.get("f1", 0)
                cm = res.get("confusion_matrix")
                roc_auc = res.get("roc_auc")

                with st.expander(f"{name} — Accuracy: {accuracy:.4f}", expanded=False):
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{accuracy:.4f}")
                    c2.metric("Precision", f"{precision:.4f}")
                    c3.metric("Recall", f"{recall:.4f}")
                    c4.metric("F1 Score", f"{f1:.4f}")

                    # Confusion matrix
                    if cm is not None:
                        st.pyplot(plot_confusion_matrix(cm, labels=["Class 0", "Class 1"]), use_container_width=True)

                    # ROC curve (binary)
                    if roc_auc is not None:
                        st.pyplot(plot_roc_curve(res.get("roc_fpr"), res.get("roc_tpr"), roc_auc), use_container_width=True)

        # Feature importance (from tree-based models)
        st.markdown("### Feature Importance")
        # Use Random Forest importances as representative
        best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
        best_model = results[best_name]["model"]

        try:
            import fig
            fi_fig = plot_feature_importance(best_model, feature_names, top_n=20)
            st.pyplot(fi_fig, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance could not be generated: {e}")

    else:  # regression
        # R², MAE, RMSE table
        comp_placeholder = st.container()
        with comp_placeholder:
            for name, res in results.items():
                r2 = res.get("r2", 0)
                mae = res.get("mae", 0)
                rmse = res.get("rmse", 0)

                with st.expander(f"{name} — R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}", expanded=False):
                    # Predicted vs actual
                    st.pyplot(
                        plot_predicted_vs_actual(res.get("y_test"), res.get("y_pred")),
                        use_container_width=True,
                    )

        # Feature importance
        st.markdown("### Feature Importance")
        best_name = max(results.keys(), key=lambda k: results[k]["r2"])
        best_model = results[best_name]["model"]

        try:
            fi_fig = plot_feature_importance(best_model, feature_names, top_n=20)
            st.pyplot(fi_fig, use_container_width=True)
            # Store figure in session state for report
            st.session_state["fi_fig"] = fi_fig
        except Exception as e:
            st.info(f"Feature importance could not be generated: {e}")
            st.session_state["fi_fig"] = None


def run_report_tab(state: dict, df_profiling: dict) -> str:
    """Render the Report tab and return the generated Markdown."""
    st.markdown("### Download Summary Report")

    # Gather all needed info from state
    dataset_summary = df_profiling
    cleaning_log = state.get("cleaning_log", [])
    target_col = state.get("target_col", "")
    task_type = state.get("task_type", "")
    model_results = state.get("model_results", {})
    best_model_name = ""

    if model_results:
        best_name = max(model_results.keys(), key=lambda k: model_results[k].get(
            "accuracy" if task_type == "classification" else "r2", 0
        ))
        best_model_name = best_name
    else:
        best_model_name = "N/A"

    # EDA findings summary
    eda_findings = {}
    # Add correlation highlight note
    numeric_cols = [c for c, t in dataset_summary.get("column_types", {}).items() if t == "numeric"]
    if numeric_cols and df_clean is not None:
        try:
            # Compute actual correlations on cleaned data
            corr_matrix = df_clean[numeric_cols].corr()
            # Find strongest correlation (excluding diagonal)
            if not corr_matrix.empty:
                # Get upper triangle to avoid duplicate pairs
                mask = np.tri(len(corr_matrix), k=1, dtype=bool)
                upper = corr_matrix.where(mask)
                # Find the pair with highest absolute correlation
                strongest = upper.abs().unstack().dropna().abs().idxmax()
                strongest_corr = corr_matrix.loc[strongest[0], strongest[1]]
                eda_findings["correlation_highlights"] = (
                    f"**Strongest correlation:** `{strongest[0]}` vs `{strongest[1]}` "
                    f"(r = {strongest_corr:.3f}) among {len(numeric_cols)} numeric columns"
                )
            else:
                eda_findings["correlation_highlights"] = (
                    f"*Correlation analysis performed on {len(numeric_cols)} numeric columns*"
                )
        except Exception:
            eda_findings["correlation_highlights"] = (
                "*Correlation analysis could not be computed*"
            )
    else:
        eda_findings["correlation_highlights"] = (
            "*No numeric columns for correlation analysis.*"
        )
    # We need the actual dataframe - let's get it from state
    df_clean = state.get("df_clean", df_profiling)

    # Reconstruct eda_findings from what we can
    eda_findings["correlation_highlights"] = (
        f"*Correlation analysis performed on {len(numeric_cols)} numeric columns*"
        if numeric_cols
        else "*No numeric columns for correlation analysis.*"
    )
    eda_findings["missing_summary"] = (
        f"*Missing value analysis: {dataset_summary.get('missing_percent', {})}*"
        if dataset_summary.get("missing_percent")
        else "*No missing data summary available.*"
    )
    eda_findings["distribution_notes"] = (
        "*Distribution plots generated for numeric columns*"
        if numeric_cols
        else "*No numeric columns for distribution analysis.*"
    )

    # Generate report markdown
    try:
        md_report = generate_report(
            dataset_summary=dataset_summary,
            cleaning_log=cleaning_log,
            eda_findings=eda_findings,
            target_column=target_col,
            task_type=task_type,
            model_results=model_results,
            best_model_name=best_model_name,
            feature_importance_fig=st.session_state.get("fi_fig"),  # passed from modeling tab
        )
    except Exception as e:
        st.error(f"Error generating report: {e}")
        md_report = "# AutoInsight Report\n\nError generating report. See console for details."

    st.markdown(md_report)

    # Provide download button
    md_bytes = md_report.encode("utf-8")
    b64 = base64.b64encode(md_bytes).decode()
    href = f"data:file/markdown;base64,{b64}"
    st.download_button(
        label="Download Report as Markdown",
        data=md_bytes,
        file_name="autosight_report.md",
        mime="text/markdown",
    )

    return md_report


# Main app entry point
def main():
    """Main entry point for the AutoInsight Streamlit app."""
    # Render sidebar - always visible
    # We need df_profiling for the sidebar, get it from session state
    df_profiling = st.session_state.get("df_profiling")
    
    # If no data profiling, show welcome message and file uploader
    if df_profiling is None:
        st.set_page_config(page_title="AutoInsight — Automated Data Analyst", layout="centered")
        st.title("📊 AutoInsight")
        st.markdown("### Automated Data Analyst")
        
        st.markdown("---")
        st.markdown("**Welcome!** Upload any CSV file and let AutoInsight automatically:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🧹 **Cleaning**\nMissing values • Outliers • Duplicates")
        with col2:
            st.info("📊 **EDA**\nDistributions • Correlations • Statistics")
        with col3:
            st.info("🤖 **Modeling**\nBaseline ML • Feature importance")
        
        st.markdown("---")
        st.markdown("**Get started by uploading a CSV file:**")
        
        # Enhanced file uploader with custom key
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file (up to 50MB) to analyze automatically",
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            # Load and profile the data
            from modules.data_loader import load_csv, profile_dataset
            df = load_csv(uploaded_file)
            profiling = profile_dataset(df)
            st.session_state.df_raw = df  # Store raw DataFrame
            st.session_state.df_profiling = profiling  # Store profiling dict
            st.session_state.app_flow = "loaded"
            st.rerun()
        return
    
    # Data is loaded - proceed with normal app flow
    df_raw = st.session_state.get("df_raw")
    profiling = df_profiling
    
    # Render sidebar
    state = render_sidebar(profiling)
    
    # Check if target column is selected
    if state["target_col"] == "(none)" or state["target_col"] is None:
        # Show overview tab with basic info
        st.header("Overview")
        st.write(f"**Rows:** {profiling.get('rows', 0):,}")
        st.write(f"**Columns:** {profiling.get('columns', 0)}")
        st.write(f"**Memory Usage:** {profiling.get('memory_mb', 0)} MB")
        st.write(f"**Duplicate Rows:** {profiling.get('duplicate_rows', 0)}")
        st.write(f"**Missing Values %:** {profiling.get('missing_percent', {})}")
        return
    
    # Target column selected - show tabbed interface
    tab_names = ["Overview", "Cleaning", "EDA", "Modeling", "Report"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Overview
        st.header("Dataset Overview")
        st.write(f"**Rows:** {profiling.get('rows', 0):,}")
        st.write(f"**Columns:** {profiling.get('columns', 0)}")
        st.write(f"**Memory Usage:** {profiling.get('memory_mb', 0)} MB")
        st.write(f"**Column Types:** {profiling.get('column_types', {})}")
    
    with tabs[1]:  # Cleaning
        st.header("Data Cleaning")
        st.write("### Cleaning Log")
        cleaning_log = st.session_state.get("cleaning_state", {}).get("cleaning_log", [])
        if cleaning_log:
            for entry in cleaning_log:
                st.markdown(f"- {entry}")
        else:
            st.info("No cleaning steps performed yet.")
        # Show cleaning strategy controls
        st.markdown("**Available cleaning strategies:**")
        st.markdown("- **Missing values:** Median imputation (numeric), Mode imputation (categorical)")
        st.markdown("- **Outliers:** IQR method with capping option")
        st.markdown("- **Duplicates:** Automatic detection and removal toggle")
    
    with tabs[2]:  # EDA
        st.header("Exploratory Data Analysis")
        run_eda_tab(df_raw, state.get("target_col"))  # Pass raw DataFrame
    
    with tabs[3]:  # Modeling
        run_modeling_tab(df_raw, state.get("target_col", ""), state.get("task_type", ""), state)
    
    with tabs[4]:  # Report
        run_report_tab(state, profiling)


if __name__ == "__main__":
    main()