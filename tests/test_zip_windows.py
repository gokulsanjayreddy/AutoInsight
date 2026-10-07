"""Unit and integration tests for Windows-safe ZIP generation, sanitization, and extraction."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from modules.data_loader import get_sample_dataset
from modules.eda import create_all_plots_zip
from modules.pipeline import generate_full_analysis
from modules.zip_utils import (
    WINDOWS_RESERVED_NAMES,
    build_safe_zip_path,
    sanitize_filename_stem,
    sanitize_folder_component,
    validate_zip_archive,
)


def _create_dummy_png_bytes() -> bytes:
    """Generate minimal valid PNG bytes."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test A: Two columns Salary and salary (case-insensitive collision)
# ---------------------------------------------------------------------------
def test_a_case_insensitive_collision_salary():
    """Salary and salary must produce two distinct ZIP entries with no Windows collision."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("01_distributions", "dist_Salary", raw_bytes),
        ("01_distributions", "dist_salary", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 2
        assert "01_distributions/dist_Salary.png" in namelist
        assert "01_distributions/dist_salary_2.png" in namelist
        # Verify no casefold collision
        casefolded = [n.casefold() for n in namelist]
        assert len(casefolded) == len(set(casefolded))


# ---------------------------------------------------------------------------
# Test B: Columns whose names become identical after sanitization
# ---------------------------------------------------------------------------
def test_b_identical_names_after_sanitization():
    """Customer/Name and Customer:Name must produce deterministic, distinct filenames."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("03_categorical", "Customer/Name", raw_bytes),
        ("03_categorical", "Customer:Name", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 2
        assert "03_categorical/Customer_Name.png" in namelist
        assert "03_categorical/Customer_Name_2.png" in namelist


# ---------------------------------------------------------------------------
# Test C: Reserved Windows device names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reserved", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"])
def test_c_reserved_windows_names(reserved: str):
    """Reserved Windows device names must be safely escaped without collisions."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("00_summary", reserved, raw_bytes),
        ("00_summary", reserved.lower(), raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 2
        assert f"00_summary/{reserved}_.png" in namelist
        assert f"00_summary/{reserved.lower()}_2.png" in namelist

        for name in namelist:
            stem = Path(name).stem.upper()
            assert stem not in WINDOWS_RESERVED_NAMES


# ---------------------------------------------------------------------------
# Test D: Trailing spaces and dots
# ---------------------------------------------------------------------------
def test_d_trailing_spaces_and_dots():
    """Filenames with trailing spaces or dots must be sanitized cleanly."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("01_distributions", "Score.", raw_bytes),
        ("01_distributions", "Score ", raw_bytes),
        ("01_distributions", "Score. . ", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 3
        assert "01_distributions/Score.png" in namelist
        assert "01_distributions/Score_2.png" in namelist
        assert "01_distributions/Score_3.png" in namelist
        for name in namelist:
            assert not name.endswith(". ")
            assert not name.endswith(" ")
            stem = Path(name).stem
            assert not stem.endswith(".")
            assert not stem.endswith(" ")


# ---------------------------------------------------------------------------
# Test E: Excessively long column names and path lengths
# ---------------------------------------------------------------------------
def test_e_very_long_column_names():
    """Excessively long names must be capped with readable prefix, hash suffix, and extension."""
    raw_bytes = _create_dummy_png_bytes()
    long_name_1 = "VERY_LONG_COLUMN_NAME_THAT_GOES_ON_AND_ON_AND_ON_FOR_SURVEY_QUESTION_ALPHA"
    long_name_2 = "VERY_LONG_COLUMN_NAME_THAT_GOES_ON_AND_ON_AND_ON_FOR_SURVEY_QUESTION_BETA"

    items = [
        ("01_univariate_numeric_distributions", long_name_1, raw_bytes),
        ("01_univariate_numeric_distributions", long_name_2, raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 2
        for name in namelist:
            assert name.endswith(".png")
            assert len(name) <= 120
            assert "_..._" in name


# ---------------------------------------------------------------------------
# Test F: Unicode column names and ZIP UTF-8 encoding
# ---------------------------------------------------------------------------
def test_f_unicode_column_names():
    """Unicode column names must produce valid ZIP entries with bit 11 UTF-8 set."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("01_distributions", "dist_München_Temperatur", raw_bytes),
        ("02_categories", "cat_東京_支社", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infolist = zf.infolist()
        assert len(infolist) == 2
        for info in infolist:
            # Bit 11 is the general purpose UTF-8 flag
            assert info.flag_bits & 0x800 != 0


# ---------------------------------------------------------------------------
# Test G: Two names differing only by Unicode normalization
# ---------------------------------------------------------------------------
def test_g_unicode_normalization_collision():
    """NFC and NFD representations of the same word must not collide on Windows."""
    raw_bytes = _create_dummy_png_bytes()
    nfc_name = "dist_caf\u00e9"  # é as precomposed
    nfd_name = "dist_cafe\u0301"  # e + combining acute

    items = [
        ("01_distributions", nfc_name, raw_bytes),
        ("01_distributions", nfd_name, raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 2
        casefolded = [n.casefold() for n in namelist]
        assert len(casefolded) == len(set(casefolded))


# ---------------------------------------------------------------------------
# Test H: Full pipeline generated ZIP validation (Demo Dataset)
# ---------------------------------------------------------------------------
def test_h_complete_demo_dataset_pipeline_zip():
    """Verify entire pipeline generation for demo dataset meets all Windows requirements."""
    df = get_sample_dataset()
    res = generate_full_analysis(df, max_total_plots=75)
    zip_bytes = res["zip_bytes"]
    total_count = res["total_count"]

    assert zipfile.is_zipfile(io.BytesIO(zip_bytes))
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert zf.testzip() is None
        namelist = zf.namelist()
        assert len(namelist) == total_count
        assert len(namelist) == len(set(namelist))

        # Check case-insensitive duplicates
        casefolded = [n.casefold() for n in namelist]
        assert len(casefolded) == len(set(casefolded))

        # Check maximum path length and Windows characters
        for name in namelist:
            assert len(name) <= 120
            assert "\\" not in name
            assert not any(c in name for c in '<>:"|?*')
            parts = name.split("/")
            for p in parts:
                stem = p.split(".")[0].upper()
                assert stem not in WINDOWS_RESERVED_NAMES
                assert not p.endswith(" ")
                assert not p.endswith(".")
                assert not p.startswith(" ")


# ---------------------------------------------------------------------------
# Test I: Awkward Column Names Pipeline Regression Test
# ---------------------------------------------------------------------------
def test_i_awkward_dataset_pipeline_zip_regression():
    """Verify pipeline handles awkward columns with zero Windows collisions."""
    rng = np.random.default_rng(123)
    n = 60
    awkward_df = pd.DataFrame({
        "Salary": rng.normal(50000, 10000, n),
        "salary": rng.normal(60000, 15000, n),
        "Customer/Name": rng.choice(["Alice", "Bob", "Charlie"], n),
        "Customer:Name": rng.choice(["X", "Y", "Z"], n),
        "CON": rng.normal(10, 2, n),
        "PRN": rng.choice(["Low", "Med", "High"], n),
        "AUX": rng.normal(100, 20, n),
        "NUL": rng.normal(5, 1, n),
        "Score. ": rng.normal(75, 5, n),
        "Score.": rng.normal(80, 5, n),
        "caf\u00e9": rng.choice(["Espresso", "Latte"], n),
        "cafe\u0301": rng.choice(["Hot", "Cold"], n),
        "A" * 80: rng.normal(0, 1, n),
        "B" * 80: rng.choice(["P", "Q"], n),
    })

    res = generate_full_analysis(awkward_df, max_total_plots=50)
    zip_bytes = res["zip_bytes"]
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert zf.testzip() is None
        namelist = zf.namelist()
        assert len(namelist) == res["total_count"]

        # Case-insensitive collisions must be exactly 0
        casefolded = [n.casefold() for n in namelist]
        assert len(casefolded) == len(set(casefolded))

        # Test actual extraction to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            extracted = list(Path(tmpdir).rglob("*.png"))
            assert len(extracted) == len(namelist)


# ---------------------------------------------------------------------------
# Test J: Validator Error Handling
# ---------------------------------------------------------------------------
def test_validator_detects_corrupted_archive():
    """Validator must raise ValueError on corrupted data."""
    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        validate_zip_archive(b"not_a_zip_file")


def test_validator_detects_case_insensitive_collision():
    """Validator must raise ValueError if an archive contains case-insensitive duplicates."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("01_dists/Salary.png", b"test")
        zf.writestr("01_dists/salary.png", b"test")
    buf.seek(0)
    with pytest.raises(ValueError, match="case-insensitive collision"):
        validate_zip_archive(buf.getvalue())


def test_validator_detects_reserved_device_names():
    """Validator must raise ValueError if an archive contains reserved device names."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("01_dists/CON.png", b"test")
    buf.seek(0)
    with pytest.raises(ValueError, match="reserved device name"):
        validate_zip_archive(buf.getvalue())


def test_validator_detects_trailing_period_or_space():
    """Validator must raise ValueError if an entry has a trailing period or space."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("01_dists/score..png", b"test")
    buf.seek(0)
    # stem is 'score.', which ends with '.'
    with pytest.raises(ValueError, match="trailing period"):
        validate_zip_archive(buf.getvalue())


def test_validator_detects_excessive_path_length():
    """Validator must raise ValueError if an entry exceeds max_path_len."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("folder/" + "a" * 200 + ".png", b"test")
    buf.seek(0)
    with pytest.raises(ValueError, match="exceeds Windows limit"):
        validate_zip_archive(buf.getvalue(), max_path_len=100)


def test_sanitize_helpers_edge_cases():
    """Test standalone helper functions on unusual edge cases."""
    # Empty and all-spaces/dots
    assert sanitize_filename_stem("") == "item"
    assert sanitize_filename_stem("   ...   ") == "item"
    assert sanitize_filename_stem("???:::***") == "item"

    # Multiple slashes and backslashes in folder
    assert sanitize_folder_component(r"folder\subfolder//") == "folder/subfolder"
    assert sanitize_folder_component("") == "plots"

    # Collision path building with existing set
    used: set[str] = set()
    p1 = build_safe_zip_path("01_dists", "Salary", used_paths_casefolded=used)
    p2 = build_safe_zip_path("01_dists", "salary", used_paths_casefolded=used)
    p3 = build_safe_zip_path("01_dists", "SALARY", used_paths_casefolded=used)
    assert p1 == "01_dists/Salary.png"
    assert p2 == "01_dists/salary_2.png"
    assert p3 == "01_dists/SALARY_3.png"
