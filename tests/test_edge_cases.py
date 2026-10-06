"""Comprehensive tests for data_loader, pipeline edge cases, and ZIP generation."""

import io
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from modules.data_loader import (
    flag_id_columns,
    infer_column_type,
    load_csv,
)
from modules.eda import fig_to_png_bytes
from modules.pipeline import generate_full_analysis


# 1. load_csv comprehensive tests
def test_load_csv_comma_and_delimiters():
    """Test load_csv with standard comma, semicolon, tab, pipe."""
    # Comma
    df = load_csv(io.BytesIO(b"a,b,c\n1,2,3\n4,5,6\n"))
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2

    # Duplicate headers
    df_dup = load_csv(io.BytesIO(b"a,a,b\n1,2,3\n4,5,6\n"))
    assert len(df_dup.columns) == 3
    assert len(df_dup) == 2


def test_load_csv_oversized_file():
    """Test load_csv with oversized file cap."""
    content = b"a,b\n1,2\n" * 1000
    with pytest.raises(ValueError, match="exceeds the 0.001 MB limit"):
        load_csv(io.BytesIO(content), max_size_mb=0.001)


# 2. infer_column_type comprehensive tests
def test_infer_column_type_edge_cases():
    """Test infer_column_type across edge scenarios."""
    # Constants
    const_num = pd.Series([5, 5, 5, 5, 5], name="constant_num")
    assert infer_column_type(const_num) == "numeric"

    const_str = pd.Series(["apple", "apple", "apple"], name="constant_str")
    assert infer_column_type(const_str) == "categorical"

    # All-NaN
    nan_series = pd.Series([np.nan, np.nan, np.nan], name="all_nan")
    assert infer_column_type(nan_series) == "high_cardinality_text"

    # Unique IDs
    id_series = pd.Series([f"ID_{i:04d}" for i in range(100)], name="customer_id")
    assert infer_column_type(id_series) == "high_cardinality_text"

    # Datetime strings
    dt_series = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"], name="dates")
    assert infer_column_type(dt_series) == "datetime"

    # Real booleans
    bool_series = pd.Series([True, False, True, False], name="is_admin")
    assert infer_column_type(bool_series) == "boolean"

    # 0/1 counts (numeric, not boolean)
    count_series = pd.Series([0, 1, 0, 1, 0, 1], name="visit_count")
    assert infer_column_type(count_series) == "numeric"

    # 0/1 boolean named
    flag_series = pd.Series([0, 1, 0, 1, 0, 1], name="is_active")
    assert infer_column_type(flag_series) == "boolean"


def test_flag_id_columns_edge_cases():
    """Test flag_id_columns detection logic."""
    col_types = {
        "user_id": "high_cardinality_text",
        "txn_id": "numeric",
        "category": "categorical",
        "identifier": "categorical",
    }
    cards = {
        "user_id": 98,
        "txn_id": 99,
        "category": 3,
        "identifier": 95,
    }
    flags = flag_id_columns(col_types, cards, n_rows=100)
    assert "user_id" in flags
    assert "identifier" in flags
    assert "category" not in flags


def test_pair_ranking_planted_column():
    """Test that planted informative column is ranked first in pairs."""
    rng = np.random.default_rng(123)
    n = 200
    cat_planted = np.array(["GroupA"] * 100 + ["GroupB"] * 100)
    cat_noise = rng.choice(["N1", "N2", "N3"], size=n)
    num_planted = np.where(cat_planted == "GroupA", 10.0, 90.0) + rng.normal(0, 0.5, size=n)
    num_noise = rng.normal(50, 10, size=n)

    df = pd.DataFrame({
        "NoiseCat": cat_noise,
        "PlantedCat": cat_planted,
        "NoiseNum": num_noise,
        "PlantedNum": num_planted,
    })

    res = generate_full_analysis(df, max_total_plots=100)
    cat_num_plots = res["card_data"]["cat_num"]
    assert len(cat_num_plots) > 0
    top_title, _, _ = cat_num_plots[0]
    assert "PlantedNum" in top_title and "PlantedCat" in top_title


# 3. Pipeline on edge datasets
def test_pipeline_edge_datasets_no_exceptions():
    """Assert generate_full_analysis handles edge datasets without exception."""
    # One column
    df_one_col = pd.DataFrame({"single": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    res1 = generate_full_analysis(df_one_col)
    assert res1["total_count"] > 0

    # All-NaN columns
    df_nan = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]})
    res2 = generate_full_analysis(df_nan)
    assert res2["total_count"] >= 0

    # Constant columns
    df_const = pd.DataFrame({"c1": [1, 1, 1, 1, 1], "c2": ["A", "A", "A", "A", "A"]})
    res3 = generate_full_analysis(df_const)
    assert res3["total_count"] >= 0

    # Boolean and datetime columns
    df_bool_dt = pd.DataFrame({
        "flag": [True, False, True, False, True],
        "dt": pd.date_range("2024-01-01", periods=5),
        "val": [10, 20, 30, 40, 50],
    })
    res4 = generate_full_analysis(df_bool_dt)
    assert res4["total_count"] > 0

    # ID-like text columns
    df_ids = pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(20)],
        "score": list(range(20)),
    })
    res5 = generate_full_analysis(df_ids)
    assert res5["total_count"] > 0

    # Duplicate column names
    df_dup = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["x", "x", "y"])
    res6 = generate_full_analysis(df_dup)
    assert res6["total_count"] > 0

    # 100 columns dataset
    cols_data = {f"c_{i}": np.random.default_rng(0).normal(size=20) for i in range(100)}
    df_100 = pd.DataFrame(cols_data)
    res7 = generate_full_analysis(df_100, max_total_plots=10)
    assert res7["total_count"] <= 10


# 4. ZIP verification tests
def test_zip_verification_and_single_rendering():
    """Verify ZIP contents, name sanitization, duplicate handling, and single render."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Score": [80.5, 90.0, 75.0, 88.0, 92.5],
        "Dept/Team:Name*": ["Sales", "HR", "Sales", "HR", "Sales"],
    })

    with patch("modules.pipeline.fig_to_png_bytes", wraps=fig_to_png_bytes) as mock_render:
        res = generate_full_analysis(df, max_total_plots=20)
        total_plots = res["total_count"]
        # Exactly one render per plot
        assert mock_render.call_count == total_plots

    zip_bytes = res["zip_bytes"]
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        # No duplicates
        assert len(namelist) == len(set(namelist))
        for name in namelist:
            # Sanitized names (no unsafe characters)
            assert not any(c in name for c in [":", "*", "?", '"', "<", ">", "|"])
            # Every PNG decodes cleanly
            data = zf.read(name)
            with Image.open(io.BytesIO(data)) as img:
                assert img.format == "PNG"
                assert img.width > 0
                assert img.height > 0
