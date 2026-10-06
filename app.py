"""AutoInsight Streamlit application entry point.

Direct, single-page automated exploratory data analysis interface.
Displays categorized, strictly separated standalone figures with 1-click batch ZIP download.
"""

from __future__ import annotations

import html
import io
import re
from typing import Any, cast

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
) -> None:
    """Render a standalone plot inside a bordered container with a PNG download button."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.image(png_bytes, width="stretch")
        st.download_button(
            label="Download Plot (PNG)",
            data=png_bytes,
            file_name=f"{download_filename}.png",
            mime="image/png",
            key=key,
            width="stretch",
        )


def render_hero() -> None:
    """Render hero header banner."""
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">AutoInsight</h1>
            <p class="hero-subtitle">Automated Exploratory Data Analysis and Diagnostic Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize necessary Streamlit session state variables."""
    defaults: dict[str, Any] = {
        "df": None,
        "source_name": None,
        "analysis_cache": None,
        "cached_hash": None,
        "uploaded_file_id": None,
        "uploader_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_file_controls() -> None:
    """Render file uploader and Demo/Reset control buttons."""
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
        if st.button("Load Demo Dataset", width="stretch"):
            st.session_state.df = get_sample_dataset()
            st.session_state.source_name = "sample_employee_analytics.csv"
            st.session_state.uploaded_file_id = "demo"
            st.session_state.uploader_key += 1
            st.session_state.analysis_cache = None
            st.session_state.cached_hash = None
            st.rerun()

        if st.session_state.df is not None and st.button("Reset Dataset", width="stretch"):
            st.session_state.df = None
            st.session_state.source_name = None
            st.session_state.analysis_cache = None
            st.session_state.cached_hash = None
            st.session_state.uploaded_file_id = None
            st.session_state.uploader_key += 1
            st.rerun()


def render_overview_metrics(profiling: dict[str, Any], source_name: str) -> None:
    """Render metric overview cards for dataset health."""
    escaped_source_name = html.escape(source_name)
    total_rows = profiling.get("rows", 0)
    total_cols = profiling.get("columns", 0)
    memory_mb = profiling.get("memory_mb", 0.0)
    dup_rows = profiling.get("duplicate_rows", 0)
    id_cols = profiling.get("id_columns", [])
    col_types = profiling.get("column_types", {})

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [
        c for c, t in col_types.items()
        if t in ("categorical", "boolean") and c not in id_cols
    ]

    st.markdown(
        "<div class='category-title'>Overview and Data Health</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='category-subtitle'>Dataset: <strong>{escaped_source_name}</strong></div>",
        unsafe_allow_html=True,
    )

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


def render_summary_section(card_data: dict[str, list[tuple[str, str, bytes]]]) -> None:
    """Render summary statistics and missing value cards."""
    st.markdown(
        "<div class='category-title'>Summary Statistics and Missing Values</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='category-subtitle'>"
        "Detailed statistical summary and missing value breakdown.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2] if card_data.get("missing") else [1, 0.01])
    with c1:
        for title, fname, png in card_data.get("summary", []):
            render_plot_card(title, fname, png, "sec1_summary")
    if card_data.get("missing"):
        with c2:
            for title, fname, png in card_data["missing"]:
                render_plot_card(title, fname, png, "sec1_missing")


def render_section(
    key: str,
    title: str,
    subtitle: str,
    cards: list[tuple[str, str, bytes]],
    key_prefix: str,
) -> None:
    """Render a titled section containing plot cards in a responsive grid."""
    if not cards:
        return

    st.markdown(f"<div class='category-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='category-subtitle'>{subtitle}</div>", unsafe_allow_html=True)

    if key == "corr":
        for title_c, fname_c, png_c in cards:
            render_plot_card(title_c, fname_c, png_c, key_prefix)
        return

    for i in range(0, len(cards), 2):
        col_a, col_b = st.columns(2)
        title_a, fname_a, png_a = cards[i]
        with col_a:
            render_plot_card(title_a, fname_a, png_a, f"{key_prefix}_{i}")

        if i + 1 < len(cards):
            title_b, fname_b, png_b = cards[i + 1]
            with col_b:
                render_plot_card(title_b, fname_b, png_b, f"{key_prefix}_{i+1}")


def render_download_banner(
    zip_bytes: bytes,
    clean_dl_name: str,
    total_plots: int,
    btn_key: str,
) -> None:
    """Render batch ZIP download button with summary caption."""
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.download_button(
            label=f"Download All Plots ({total_plots} Files, ZIP)",
            data=zip_bytes,
            file_name=f"autoinsight_eda_{clean_dl_name}.zip",
            mime="application/zip",
            width="stretch",
            type="primary",
            key=btn_key,
        )
    with col_info:
        st.markdown(
            f"<div style='padding-top: 0.5rem; color: var(--text-color, #475569);'>"
            f"Generated <strong>{total_plots}</strong> separated standalone charts "
            "across all column combinations.</div>",
            unsafe_allow_html=True,
        )


SECTION_CONFIG = [
    (
        "dists",
        "Numeric Distributions",
        "Histograms with kernel density estimation, mean, and median.",
        "sec2_dist",
    ),
    (
        "boxplots",
        "Outlier Boxplots",
        "Box and whisker plots highlighting median, interquartile range, and outliers.",
        "sec3_box",
    ),
    (
        "cats",
        "Categorical Distributions",
        "Frequency distributions and proportions for categorical variables.",
        "sec4_cat",
    ),
    (
        "corr",
        "Correlation Matrix",
        "Pearson correlation matrix across all numeric features.",
        "sec5_corr",
    ),
    (
        "scatters",
        "Numeric vs Numeric Relationships",
        "Bivariate scatter plots with linear trendlines for numeric pairs.",
        "sec6_scat",
    ),
    (
        "cat_num",
        "Categorical vs Numeric Relationships",
        "Comparative distributions of numeric metrics grouped by category.",
        "sec7_cn",
    ),
    (
        "cat_cat",
        "Categorical vs Categorical Interactions",
        "Cross-tabulations demonstrating relationships between categorical variables.",
        "sec8_cc",
    ),
]


def ensure_analysis(df: pd.DataFrame, source_name: str) -> dict[str, Any]:
    """Execute EDA pipeline if cache is stale or missing."""
    h_data = pd.util.hash_pandas_object(df, index=True).sum()
    h_cols = list(df.columns)
    h_dtypes = [str(t) for t in df.dtypes]
    current_hash = f"{h_data}_{h_cols}_{h_dtypes}_{source_name}"

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

    return cast(dict[str, Any], st.session_state.analysis_cache)


def main() -> None:
    """Main application layout and execution."""
    render_hero()
    init_session_state()
    render_file_controls()

    if st.session_state.df is None:
        st.info("Upload a CSV file above or click Load Demo Dataset to begin analysis.")
        return

    df: pd.DataFrame = st.session_state.df
    source_name = st.session_state.source_name or "dataset"
    clean_dl_name = re.sub(r"(?i)\.csv$", "", source_name)
    clean_dl_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", clean_dl_name)

    profiling = cached_profile_dataset(df)
    render_overview_metrics(profiling, source_name)

    analysis = ensure_analysis(df, source_name)
    card_data = analysis["card_data"]
    zip_bytes = analysis["zip_bytes"]
    total_plots = analysis["total_count"]

    st.markdown("---")
    render_download_banner(zip_bytes, clean_dl_name, total_plots, "btn_dl_all_top")
    render_summary_section(card_data)

    for key, title, subtitle, prefix in SECTION_CONFIG:
        render_section(key, title, subtitle, card_data.get(key, []), prefix)

    st.markdown("---")
    render_download_banner(zip_bytes, clean_dl_name, total_plots, "btn_dl_all_bottom")


if __name__ == "__main__":
    main()