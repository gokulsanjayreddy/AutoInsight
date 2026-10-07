"""EDA (Exploratory Data Analysis) module for AutoInsight.

Generates independent figures for univariate, bivariate, and multivariate
data analysis across all column interactions.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import re
import time
import unicodedata
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


# Windows reserved device names (DOS device names).
# Cannot be used as stem (before extension) or directory name in Win32.
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    "CONIN$",
    "CONOUT$",
    "COM0",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT0",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
})

# Windows forbidden characters: < > : " / \ | ? * and ASCII control characters (0-31, 127)
WINDOWS_INVALID_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')

# Redundant underscores pattern
MULTI_UNDERSCORE_RE = re.compile(r"_+")


def normalize_for_collision(path: str) -> str:
    """Return a Unicode-normalized casefolded string for collision detection.

    Ensures that both case variations (e.g. 'Salary.png' vs 'salary.png') and
    Unicode normalization forms (NFC vs NFD) map to identical collision keys.
    """
    return unicodedata.normalize("NFC", path).casefold()


def sanitize_filename_stem(
    name: str,
    max_len: int = 55,
    fallback: str = "item",
) -> str:
    """Sanitize a filename stem to be Windows-safe, readable, and bounded in length.

    Transformations:
    1. Normalize Unicode to NFC form.
    2. Replace Windows-invalid characters, control characters, spaces, and separators with '_'.
    3. Collapse consecutive underscores and strip leading/trailing dots, spaces, and underscores.
    4. Guard against empty names by falling back to `fallback`.
    5. Disambiguate Windows reserved device names (e.g. 'CON' -> 'CON_').
    6. Truncate long names sensibly to `max_len`, preserving a readable prefix and
       appending a deterministic short hash suffix (e.g., '_..._A1B2').
    """
    if not name:
        name = fallback

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", name)

    # 2. Replace invalid / separator / control characters with underscore
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in normalized)

    # 3. Collapse multiple underscores
    cleaned = MULTI_UNDERSCORE_RE.sub("_", cleaned)

    # Strip leading and trailing dots, spaces, underscores
    cleaned = cleaned.strip(". _")

    # 4. Fallback if empty or all characters were stripped
    if not cleaned:
        cleaned = fallback

    # 5. Check Windows reserved device names (case-insensitive check)
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"

    # 6. Length truncation
    if len(cleaned) > max_len:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:4].upper()
        hash_suffix = f"_..._{digest}"
        keep_len = max(1, max_len - len(hash_suffix))
        prefix = cleaned[:keep_len].rstrip("._")
        if not prefix:
            prefix = cleaned[:keep_len]
        cleaned = f"{prefix}{hash_suffix}"
        if cleaned.upper() in WINDOWS_RESERVED_NAMES:
            cleaned = f"{cleaned}_"

    return cleaned


def sanitize_folder_component(
    folder: str,
    max_len: int = 40,
    fallback: str = "plots",
) -> str:
    """Sanitize a directory path to be Windows-safe, preserving subfolder structure if any."""
    folder = folder.replace("\\", "/").strip("/ ")
    parts = [p for p in folder.split("/") if p and p != "." and p != ".."]
    if not parts:
        return fallback

    sanitized_parts = []
    for part in parts:
        clean_part = sanitize_filename_stem(part, max_len=max_len, fallback=fallback)
        sanitized_parts.append(clean_part)

    return "/".join(sanitized_parts)


def generate_safe_zip_entry_path(
    folder: str,
    name_base: str,
    used_names: set[str],
    ext: str = ".png",
    max_stem_len: int = 55,
) -> str:
    """Generate a Windows-safe relative ZIP path with normalized collision tracking.

    Parameters
    ----------
    folder : str
        Directory category (e.g. '01_univariate_numeric_distributions').
    name_base : str
        Base filename before extension (e.g. column name or metric name).
    used_names : set[str]
        Set of normalized collision keys (from `normalize_for_collision`) already reserved.
        Updated in-place with the selected candidate's normalized key.
    ext : str, optional
        File extension, default '.png'.
    max_stem_len : int, optional
        Maximum length of the filename stem.

    Returns
    -------
    str
        The final Windows-safe relative ZIP path (e.g. '01_dists/Salary.png' or
        '01_dists/salary_2.png').
    """
    clean_folder = sanitize_folder_component(folder)

    # Strip extension if already present in name_base
    if ext and name_base.lower().endswith(ext.lower()):
        name_base = name_base[:-len(ext)]

    clean_stem = sanitize_filename_stem(name_base, max_len=max_stem_len)
    zip_path = f"{clean_folder}/{clean_stem}{ext}"

    candidate_key = normalize_for_collision(zip_path)
    if candidate_key not in used_names:
        used_names.add(candidate_key)
        return zip_path

    # Collision detected: generate deterministic suffix _2, _3, ...
    counter = 2
    suffix_sep = "" if clean_stem.endswith("_") else "_"
    while True:
        zip_path = f"{clean_folder}/{clean_stem}{suffix_sep}{counter}{ext}"
        candidate_key = normalize_for_collision(zip_path)
        if candidate_key not in used_names:
            used_names.add(candidate_key)
            return zip_path
        counter += 1


def build_safe_zip_path(
    folder: str,
    name_base: str,
    ext: str = ".png",
    used_paths_casefolded: set[str] | None = None,
    max_stem_len: int = 55,
) -> str:
    """Build a Windows-safe relative ZIP path with case-insensitive collision handling.

    Compatibility wrapper around generate_safe_zip_entry_path.
    """
    if used_paths_casefolded is None:
        used_paths_casefolded = set()
    return generate_safe_zip_entry_path(
        folder=folder,
        name_base=name_base,
        used_names=used_paths_casefolded,
        ext=ext,
        max_stem_len=max_stem_len,
    )


def _check_trailing_spaces_dots(path: str, part: str) -> None:
    """Validate trailing spaces and periods on component and stem."""
    if part.endswith(" "):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has trailing space (invalid on Windows)."
        )
    if part.endswith("."):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has trailing period (invalid on Windows)."
        )
    if "." in part:
        part_stem = part.rsplit(".", 1)[0]
        if part_stem.endswith(" "):
            raise ValueError(
                f"ZIP entry '{path}' component stem '{part_stem}' has trailing space before "
                "extension (invalid on Windows)."
            )
        if part_stem.endswith("."):
            raise ValueError(
                f"ZIP entry '{path}' component stem '{part_stem}' has trailing period before "
                "extension (invalid on Windows)."
            )


def _validate_path_component(path: str, part: str) -> None:
    """Validate a single path component for Windows filesystem safety."""
    if not part or part == ".":
        raise ValueError(f"ZIP entry '{path}' contains empty or '.' component.")
    if part == "..":
        raise ValueError(f"ZIP entry '{path}' contains directory traversal '..' component.")

    _check_trailing_spaces_dots(path, part)

    if part.startswith(" "):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has leading space (invalid on Windows)."
        )

    invalid_match = WINDOWS_INVALID_CHARS_RE.search(part)
    if invalid_match:
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' contains Windows-invalid character: "
            f"{invalid_match.group()!r}"
        )

    stem = part.split(".")[0].upper()
    if stem in WINDOWS_RESERVED_NAMES or part.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' uses Windows reserved device name '{stem}'."
        )


def validate_zip_archive(
    zip_bytes: bytes,
    max_path_len: int = 180,
    allow_empty: bool = False,
) -> None:
    """Validate in-memory ZIP archive for integrity and complete Windows compatibility.

    Raises
    ------
    ValueError
        If any validation rule is violated, with a clear explanation of the offending entry.
    """
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise ValueError("Invalid ZIP archive: data is not a valid zip archive.")

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        corrupted_entry = zf.testzip()
        if corrupted_entry is not None:
            raise ValueError(f"Corrupted ZIP entry detected: '{corrupted_entry}' failed CRC check.")

        infolist = zf.infolist()
        if not infolist and not allow_empty:
            raise ValueError("Invalid ZIP archive: archive contains 0 entries.")

        seen_exact: set[str] = set()
        seen_normalized: set[str] = set()

        for info in infolist:
            path = info.filename

            # 1. Non-empty check
            if not path or not path.strip():
                raise ValueError(f"Invalid ZIP entry: empty or whitespace-only filename '{path}'.")

            # 2. Exact duplicate check
            if path in seen_exact:
                raise ValueError(f"Duplicate ZIP entry detected: '{path}' appears multiple times.")
            seen_exact.add(path)

            # 3. Case-insensitive and Unicode-normalized collision check
            collision_key = normalize_for_collision(path)
            if collision_key in seen_normalized:
                raise ValueError(
                    f"Windows case-insensitive collision detected: '{path}' collides with an "
                    "existing entry."
                )
            seen_normalized.add(collision_key)

            # 4. Path separator check: must use '/', never '\'
            if "\\" in path:
                raise ValueError(f"ZIP entry contains backslash path separator: '{path}'")

            # Must not have leading slash
            if path.startswith("/"):
                raise ValueError(f"ZIP entry contains forbidden leading slash: '{path}'")

            # 5. Path length check
            if len(path) > max_path_len:
                raise ValueError(
                    f"ZIP entry path length ({len(path)}) exceeds Windows limit "
                    f"({max_path_len}): '{path}'"
                )

            # 6. Check each component
            normalized_path = path[:-1] if path.endswith("/") else path
            parts = normalized_path.split("/")
            for part in parts:
                _validate_path_component(path, part)


def create_all_plots_zip(named_figures: Sequence[tuple[str, str, bytes | plt.Figure]]) -> bytes:
    """Create an in-memory zip archive containing all generated plots with Windows-safe paths."""
    zip_buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, name_base, content in named_figures:
            zip_path = generate_safe_zip_entry_path(
                folder=folder,
                name_base=name_base,
                used_names=used_names,
                ext=".png",
            )

            if isinstance(content, bytes):
                png_data = content
            elif isinstance(content, bytearray):
                png_data = bytes(content)
            else:
                png_data = fig_to_png_bytes(content)
                plt.close(content)

            zinfo = zipfile.ZipInfo(zip_path)
            zinfo.date_time = time.localtime(time.time())[:6]
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.flag_bits |= 0x800  # Explicitly enforce UTF-8 filename encoding flag (bit 11)
            zinfo.external_attr = 0o644 << 16
            zf.writestr(zinfo, png_data)

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()
    validate_zip_archive(zip_bytes, allow_empty=(len(named_figures) == 0))
    return zip_bytes


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