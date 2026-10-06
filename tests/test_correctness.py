"""Tests for Section 1 correctness bugs.

These tests assert correct behavior for bugs 1.1 - 1.8.
"""

import io
import re

import numpy as np
import pandas as pd
import pytest

from modules.data_loader import (
    flag_id_columns,
    get_sample_dataset,
    infer_column_type,
    load_csv,
)
from modules.eda import (
    plot_cat_cat_relationships,
    plot_cat_num_relationships,
    plot_categorical_bars,
    plot_distributions,
)


def test_1_3_plot_distributions_returns_named_tuples_and_does_not_shift():
    """Bug 1.3: plot_distributions must return list[tuple[str, Figure]] with real column names."""
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan],
        "val_a": [1.0, 2.0, 3.0],
        "val_b": [10.0, 20.0, 30.0],
    })
    results = plot_distributions(df)
    # Must return a list of (col_name, Figure) tuples
    assert isinstance(results, list)
    assert len(results) == 2
    col_names = [col for col, fig in results]
    assert col_names == ["val_a", "val_b"]
    assert "all_nan" not in col_names


def test_1_4_categorical_bars_percentages_sum_to_100_with_nans():
    """Bug 1.4: Percentages in categorical bar charts must sum to 100%."""
    df = pd.DataFrame({
        "status": ["active", "active", "active", "pending", "pending", np.nan, np.nan]
    })
    figs = plot_categorical_bars(df)
    assert len(figs) == 1
    ax = figs[0].axes[0]
    # Check annotations text for percentages
    percentages = []
    for text in ax.texts:
        match = re.search(r"\((\d+\.?\d*)%\)", text.get_text())
        if match:
            percentages.append(float(match.group(1)))
    assert len(percentages) > 0
    total_pct = sum(percentages)
    assert pytest.approx(total_pct, abs=0.5) == 100.0


def test_1_5_pair_ranking_cat_num_and_cat_cat():
    """Bug 1.5: Pair selection ranks by eta-squared and Cramer's V, putting scores in title."""
    rng = np.random.default_rng(42)
    n = 100
    # Perfect predictor vs noisy predictor
    cat_strong = np.array(["A"] * 50 + ["B"] * 50)
    cat_weak = rng.choice(["X", "Y"], size=n)
    num_metric = np.where(cat_strong == "A", 10.0, 100.0) + rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        "CatWeak": cat_weak,
        "CatStrong": cat_strong,
        "Metric": num_metric,
    })

    # Cat vs Num: CatStrong should be ranked first
    cat_nums = plot_cat_num_relationships(df, max_pairs=1)
    assert len(cat_nums) == 1
    label, fig = cat_nums[0]
    assert "CatStrong" in label
    title = fig.axes[0].get_title()
    assert ("η²" in title or "eta²" in title)

    # Cat vs Cat: Cramer's V
    cat_a = np.array(["Alpha"] * 50 + ["Beta"] * 50)
    cat_b = np.array(["One"] * 50 + ["Two"] * 50)  # perfectly correlated with cat_a
    cat_c = rng.choice(["P", "Q"], size=n)          # uncorrelated

    df_cat = pd.DataFrame({
        "CatC": cat_c,
        "CatA": cat_a,
        "CatB": cat_b,
    })
    cat_cats = plot_cat_cat_relationships(df_cat, max_pairs=1)
    assert len(cat_cats) == 1
    label_cc, fig_cc = cat_cats[0]
    assert ("CatA" in label_cc and "CatB" in label_cc)
    title_cc = fig_cc.axes[0].get_title()
    assert ("V =" in title_cc or "Cramér" in title_cc or "Cramer" in title_cc)


def test_1_6_load_csv_delimiters_encodings_and_validation():
    """Bug 1.6: load_csv handles semicolons, tabs, latin-1, utf-8-sig, empty files, size caps."""
    # Semicolon delimited
    csv_semi = b"col1;col2;col3\n1;2;3\n4;5;6\n"
    df_semi = load_csv(io.BytesIO(csv_semi))
    assert list(df_semi.columns) == ["col1", "col2", "col3"]
    assert len(df_semi) == 2

    # Tab delimited
    csv_tab = b"colA\tcolB\n10\t20\n30\t40\n"
    df_tab = load_csv(io.BytesIO(csv_tab))
    assert list(df_tab.columns) == ["colA", "colB"]
    assert len(df_tab) == 2

    # Latin-1 encoding with special char
    csv_latin1 = "name,city\nRen\xe9,Montr\xe9al\n".encode("latin-1")
    df_latin1 = load_csv(io.BytesIO(csv_latin1))
    assert len(df_latin1) == 1
    assert df_latin1.iloc[0]["city"] == "Montréal"

    # UTF-8 BOM
    csv_bom = "\ufeffid,val\n1,100\n".encode("utf-8")
    df_bom = load_csv(io.BytesIO(csv_bom))
    assert list(df_bom.columns) == ["id", "val"]

    # Empty file
    with pytest.raises(ValueError, match="empty"):
        load_csv(io.BytesIO(b""))

    # Header-only / 0 rows
    with pytest.raises(ValueError, match="empty|no data rows"):
        load_csv(io.BytesIO(b"col1,col2\n"))


def test_1_7_infer_column_type_and_id_columns():
    """Bug 1.7: Consistent column typing, 0/1 counts, small sample yes/no, id flags."""
    # 0/1 counts should be numeric when not bool dtype or bool named
    count_series = pd.Series([0, 1, 0, 1, 0, 1], name="count_events")
    assert infer_column_type(count_series) == "numeric"

    # Real boolean dtype or string yes/no
    bool_series = pd.Series([True, False, True], name="is_active")
    assert infer_column_type(bool_series) == "boolean"

    # Small sample yes/no (4 rows)
    small_yes_no = pd.Series(["yes", "no", "yes", "no"], name="consented")
    assert infer_column_type(small_yes_no) in ("boolean", "categorical")
    assert infer_column_type(small_yes_no) != "high_cardinality_text"

    # ID columns detection
    types = {
        "user_id": "high_cardinality_text",
        "category": "categorical",
    }
    cards = {"user_id": 100, "category": 3}
    ids = flag_id_columns(types, cards, n_rows=100)
    assert ids == ["user_id"]


def test_1_8_sample_dataset_does_not_mutate_global_rng():
    """Bug 1.8: get_sample_dataset should use default_rng and not mutate global np.random."""
    np.random.seed(12345)
    r1 = np.random.rand()
    np.random.seed(12345)
    _ = get_sample_dataset()
    r2 = np.random.rand()
    # If get_sample_dataset calls np.random.seed(42), r1 != r2
    assert r1 == r2
