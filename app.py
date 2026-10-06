"""AutoInsight Streamlit application entry point.

Direct, single-page automated exploratory data analysis interface.
Displays categorized, strictly separated standalone figures with 1-click batch ZIP download.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.data_loader import (
    get_sample_dataset,
    load_csv,
    profile_dataset,
)
from modules.pipeline import generate_full_analysis

st.set_page_config(
    page_title="AutoInsight - Automated Data Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def cached_load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Cached CSV parsing keyed on raw file bytes."""
    import io
    return load_csv(io.BytesIO(file_bytes))


@st.cache_data
def cached_profile_dataset(df: pd.DataFrame) -> dict:
    """Cached dataset profiling keyed on DataFrame content."""
    return profile_dataset(df)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px;
    }

    .hero-container {
        background: #0f172a;
        padding: 2rem 2.25rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.75rem;
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    .category-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-color, #0f172a);
        margin-top: 2.25rem;
        margin-bottom: 0.2rem;
    }
    .category-subtitle {
        font-size: 0.9rem;
        color: var(--text-color, #64748b);
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color, #ffffff);
        border: 1px solid var(--border-color, #e2e8f0);
        padding: 0.9rem 1.15rem;
        border-radius: 8px;
    }

    button[kind="primary"] {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_plot_card(
    title: str,
    download_filename: str,
    png_bytes: bytes,
    key: str,
):
    """Render a standalone plot inside a bordered container with a PNG download button."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.image(png_bytes, use_container_width=True)
        st.download_button(
            label="Download Plot (PNG)",
            data=png_bytes,
            file_name=f"{download_filename}.png",
            mime="image/png",
            key=key,
            use_container_width=True,
        )


def main():
    """Main application layout and execution."""
    import html
    import re

    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">AutoInsight</h1>
            <p class="hero-subtitle">Automated Exploratory Data Analysis and Diagnostic Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "df" not in st.session_state:
        st.session_state.df = None
    if "source_name" not in st.session_state:
        st.session_state.source_name = None
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = None
    if "cached_hash" not in st.session_state:
        st.session_state.cached_hash = None
    if "uploaded_file_id" not in st.session_state:
        st.session_state.uploaded_file_id = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    upload_col, sample_col = st.columns([3, 1], gap="medium")

    with upload_col:
        uploader_widget_key = f"file_uploader_{st.session_state.uploader_key}"
        uploaded_file = st.file_uploader(
            "Upload a CSV dataset:",
            type=["csv"],
            key=uploader_widget_key,
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}_{hash(file_bytes[:4096])}"
            if st.session_state.uploaded_file_id != current_file_id:
                try:
                    st.session_state.df = cached_load_csv(file_bytes)
                    st.session_state.source_name = uploaded_file.name
                    st.session_state.uploaded_file_id = current_file_id
                    st.session_state.analysis_cache = None
                    st.session_state.cached_hash = None
                except ValueError as e:
                    st.session_state.df = None
                    st.session_state.source_name = None
                    st.session_state.uploaded_file_id = None
                    st.session_state.analysis_cache = None
                    st.session_state.cached_hash = None
                    st.error(f"Error loading CSV file: {e}")

    with sample_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Load Demo Dataset", use_container_width=True):
            st.session_state.df = get_sample_dataset()
            st.session_state.source_name = "sample_employee_analytics.csv"
            st.session_state.uploaded_file_id = "demo"
            st.session_state.uploader_key += 1
            st.session_state.analysis_cache = None
            st.session_state.cached_hash = None
            st.rerun()

        if st.session_state.df is not None and st.button("Reset Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.source_name = None
            st.session_state.analysis_cache = None
            st.session_state.cached_hash = None
            st.session_state.uploaded_file_id = None
            st.session_state.uploader_key += 1
            st.rerun()

    if st.session_state.df is None:
        st.info("Upload a CSV file above or click Load Demo Dataset to begin analysis.")
        return

    df: pd.DataFrame = st.session_state.df
    source_name = st.session_state.source_name or "dataset"
    escaped_source_name = html.escape(source_name)
    clean_dl_name = re.sub(r"(?i)\.csv$", "", source_name)
    clean_dl_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", clean_dl_name)

    profiling = cached_profile_dataset(df)
    total_rows = profiling.get("rows", len(df))
    total_cols = profiling.get("columns", len(df.columns))
    memory_mb = profiling.get("memory_mb", 0.0)
    dup_rows = profiling.get("duplicate_rows", 0)
    id_cols = profiling.get("id_columns", [])
    col_types = profiling.get("column_types", {})

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c not in id_cols]

    st.markdown("<div class='category-title'>Overview and Data Health</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='category-subtitle'>Dataset: <strong>{escaped_source_name}</strong></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Rows", f"{total_rows:,}")
    with m2:
        st.metric("Columns", total_cols)
    with m3:
        st.metric("Numeric Columns", len(num_cols))
    with m4:
        st.metric("Categorical Columns", len(cat_cols))
    with m5:
        st.metric("Memory", f"{memory_mb:.2f} MB")
    with m6:
        st.metric("Duplicate Rows", dup_rows)

    current_hash = f"{pd.util.hash_pandas_object(df, index=True).sum()}_{list(df.columns)}_{[str(t) for t in df.dtypes]}_{source_name}"
    if st.session_state.analysis_cache is None or st.session_state.cached_hash != current_hash:
        progress_bar = st.progress(0, text="Analyzing dataset and generating charts...")

        def update_progress(pct: float, msg: str) -> None:
            progress_bar.progress(int(pct * 100), text=msg)

        st.session_state.analysis_cache = generate_full_analysis(
            df,
            progress_callback=update_progress,
        )
        st.session_state.cached_hash = current_hash
        progress_bar.empty()

    analysis = st.session_state.analysis_cache
    card_data = analysis["card_data"]
    zip_bytes = analysis["zip_bytes"]
    total_plots = analysis["total_count"]

    st.markdown("---")
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.download_button(
            label=f"Download All Plots ({total_plots} Files, ZIP)",
            data=zip_bytes,
            file_name=f"autoinsight_eda_{clean_dl_name}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
            key="btn_dl_all_top",
        )
    with col_info:
        st.markdown(
            f"<div style='padding-top: 0.5rem; color: var(--text-color, #475569);'>Generated <strong>{total_plots}</strong> separated standalone charts across all column combinations.</div>",
            unsafe_allow_html=True,
        )

    # 1. Summary Statistics & Data Health
    st.markdown("<div class='category-title'>Summary Statistics and Missing Values</div>", unsafe_allow_html=True)
    st.markdown("<div class='category-subtitle'>Detailed statistical summary and missing value breakdown.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2] if card_data.get("missing") else [1, 0.01])
    with c1:
        for title, fname, png in card_data["summary"]:
            render_plot_card(title, fname, png, "sec1_summary")
    if card_data.get("missing"):
        with c2:
            for title, fname, png in card_data["missing"]:
                render_plot_card(title, fname, png, "sec1_missing")

    # 2. Univariate Numeric Distributions
    dists = card_data.get("dists", [])
    if dists:
        st.markdown("<div class='category-title'>Numeric Distributions</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Histograms with kernel density estimation, mean, and median.</div>", unsafe_allow_html=True)

        for i in range(0, len(dists), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = dists[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec2_dist_{i}")

            if i + 1 < len(dists):
                title_b, fname_b, png_b = dists[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec2_dist_{i+1}")

    # 3. Outlier Boxplots
    boxplots = card_data.get("boxplots", [])
    if boxplots:
        st.markdown("<div class='category-title'>Outlier Boxplots</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Box and whisker plots highlighting median, interquartile range, and outliers.</div>", unsafe_allow_html=True)

        for i in range(0, len(boxplots), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = boxplots[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec3_box_{i}")

            if i + 1 < len(boxplots):
                title_b, fname_b, png_b = boxplots[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec3_box_{i+1}")

    # 4. Categorical Distributions
    cats = card_data.get("cats", [])
    if cats:
        st.markdown("<div class='category-title'>Categorical Distributions</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Frequency distributions and proportions for categorical variables.</div>", unsafe_allow_html=True)

        for i in range(0, len(cats), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = cats[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec4_cat_{i}")

            if i + 1 < len(cats):
                title_b, fname_b, png_b = cats[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec4_cat_{i+1}")

    # 5. Correlation Matrix
    corrs = card_data.get("corr", [])
    if corrs:
        st.markdown("<div class='category-title'>Correlation Matrix</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Pearson correlation matrix across all numeric features.</div>", unsafe_allow_html=True)
        for title, fname, png in corrs:
            render_plot_card(title, fname, png, "sec5_corr")

    # 6. Numeric vs Numeric Relationships
    scatters = card_data.get("scatters", [])
    if scatters:
        st.markdown("<div class='category-title'>Numeric vs Numeric Relationships</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Bivariate scatter plots with linear trendlines for numeric pairs.</div>", unsafe_allow_html=True)

        for i in range(0, len(scatters), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = scatters[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec6_scat_{i}")

            if i + 1 < len(scatters):
                title_b, fname_b, png_b = scatters[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec6_scat_{i+1}")

    # 7. Categorical vs Numeric Relationships
    cat_nums = card_data.get("cat_num", [])
    if cat_nums:
        st.markdown("<div class='category-title'>Categorical vs Numeric Relationships</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Comparative distributions of numeric metrics grouped by category.</div>", unsafe_allow_html=True)

        for i in range(0, len(cat_nums), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = cat_nums[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec7_cn_{i}")

            if i + 1 < len(cat_nums):
                title_b, fname_b, png_b = cat_nums[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec7_cn_{i+1}")

    # 8. Categorical vs Categorical Interactions
    cat_cats = card_data.get("cat_cat", [])
    if cat_cats:
        st.markdown("<div class='category-title'>Categorical vs Categorical Interactions</div>", unsafe_allow_html=True)
        st.markdown("<div class='category-subtitle'>Cross-tabulations demonstrating relationships between categorical variables.</div>", unsafe_allow_html=True)

        for i in range(0, len(cat_cats), 2):
            col_a, col_b = st.columns(2)
            title_a, fname_a, png_a = cat_cats[i]
            with col_a:
                render_plot_card(title_a, fname_a, png_a, f"sec8_cc_{i}")

            if i + 1 < len(cat_cats):
                title_b, fname_b, png_b = cat_cats[i + 1]
                with col_b:
                    render_plot_card(title_b, fname_b, png_b, f"sec8_cc_{i+1}")

    st.markdown("---")
    bot_col, _ = st.columns([1, 2])
    with bot_col:
        st.download_button(
            label=f"Download All Plots ({total_plots} Files, ZIP)",
            data=zip_bytes,
            file_name=f"autoinsight_eda_{clean_dl_name}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
            key="btn_dl_all_bottom",
        )


if __name__ == "__main__":
    main()