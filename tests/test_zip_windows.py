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
from modules.eda import (
    WINDOWS_RESERVED_NAMES,
    build_safe_zip_path,
    create_all_plots_zip,
    generate_safe_zip_entry_path,
    normalize_for_collision,
    sanitize_filename_stem,
    sanitize_folder_component,
    validate_zip_archive,
)
from modules.pipeline import generate_full_analysis


def _create_dummy_png_bytes() -> bytes:
    """Generate minimal valid PNG bytes."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests: normalize_for_collision and generate_safe_zip_entry_path
# ---------------------------------------------------------------------------
def test_normalize_for_collision_logic():
    """Verify normalize_for_collision normalizes Unicode to NFC and casefolds."""
    assert normalize_for_collision("Salary.png") == normalize_for_collision("salary.png")
    assert normalize_for_collision("Salary.png") == "salary.png"

    # Unicode precomposed vs combining characters
    nfc = "caf\u00e9.png"
    nfd = "cafe\u0301.png"
    assert normalize_for_collision(nfc) == normalize_for_collision(nfd)


def test_generate_safe_zip_entry_path_tracks_normalized_collisions():
    """Verify candidate_key = normalize_for_collision(zip_path) is tracked in used_names."""
    used_names: set[str] = set()

    p1 = generate_safe_zip_entry_path("01_dists", "Salary", used_names)
    assert p1 == "01_dists/Salary.png"
    assert "01_dists/salary.png" in used_names

    p2 = generate_safe_zip_entry_path("01_dists", "salary", used_names)
    assert p2 == "01_dists/salary_2.png"
    assert "01_dists/salary_2.png" in used_names

    p3 = generate_safe_zip_entry_path("01_dists", "SALARY", used_names)
    assert p3 == "01_dists/SALARY_3.png"
    assert "01_dists/salary_3.png" in used_names


# ---------------------------------------------------------------------------
# TEST 1: Columns Salary and salary (case-insensitive collision)
# ---------------------------------------------------------------------------
def test_1_case_insensitive_collision_salary():
    """TEST 1: Columns Salary and salary must produce two distinct PNGs with unique safe names."""
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
        casefolded = [normalize_for_collision(n) for n in namelist]
        assert len(casefolded) == len(set(casefolded))


# ---------------------------------------------------------------------------
# TEST 2: Columns Customer/Name and Customer:Name
# ---------------------------------------------------------------------------
def test_2_identical_names_after_sanitization():
    """TEST 2: Customer/Name and Customer:Name must produce distinct filenames."""
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
# TEST 3: Reserved names: CON, PRN, AUX, NUL, COM1, LPT1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reserved", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"])
def test_3_reserved_windows_names(reserved: str):
    """TEST 3: Reserved names must be safely escaped and extractable on Windows."""
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

        # Verify disk extraction on Windows
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            extracted = list(Path(tmpdir).rglob("*.png"))
            assert len(extracted) == 2


# ---------------------------------------------------------------------------
# TEST 4: Trailing dot: "Salary."
# ---------------------------------------------------------------------------
def test_4_trailing_dot():
    """TEST 4: Trailing dot must produce a safe filename without trailing period."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("01_distributions", "Salary.", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 1
        assert "01_distributions/Salary.png" in namelist
        for name in namelist:
            assert not Path(name).stem.endswith(".")


# ---------------------------------------------------------------------------
# TEST 5: Trailing space: "Salary "
# ---------------------------------------------------------------------------
def test_5_trailing_space():
    """TEST 5: Trailing space must produce a safe filename without trailing space."""
    raw_bytes = _create_dummy_png_bytes()
    items = [
        ("01_distributions", "Salary ", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    validate_zip_archive(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 1
        assert "01_distributions/Salary.png" in namelist
        for name in namelist:
            assert not Path(name).stem.endswith(" ")


# ---------------------------------------------------------------------------
# TEST 6: Very long column name
# ---------------------------------------------------------------------------
def test_6_very_long_column_name():
    """TEST 6: Very long column names must remain within the chosen safe length limit."""
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
# TEST 7: Unicode names
# ---------------------------------------------------------------------------
def test_7_unicode_column_names():
    """TEST 7: Unicode column names must produce valid ZIP entries with bit 11 UTF-8 set."""
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
# TEST 8: Unicode normalization equivalents (NFC vs NFD)
# ---------------------------------------------------------------------------
def test_8_unicode_normalization_collision():
    """TEST 8: NFC and NFD representations of the same word must not collide on Windows."""
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
        casefolded = [normalize_for_collision(n) for n in namelist]
        assert len(casefolded) == len(set(casefolded))


# ---------------------------------------------------------------------------
# TEST 9: Complete demo dataset
# ---------------------------------------------------------------------------
def test_9_complete_demo_dataset():
    """TEST 9: Complete demo dataset must produce a valid ZIP meeting all Windows criteria."""
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
        casefolded = [normalize_for_collision(n) for n in namelist]
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
# TEST 10: Awkward-column-name dataset
# ---------------------------------------------------------------------------
def test_10_awkward_dataset_pipeline_zip():
    """TEST 10: Awkward column dataset must produce valid ZIP with zero collisions."""
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
        casefolded = [normalize_for_collision(n) for n in namelist]
        assert len(casefolded) == len(set(casefolded))

        # Test actual extraction to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            extracted = list(Path(tmpdir).rglob("*.png"))
            assert len(extracted) == len(namelist)


# ---------------------------------------------------------------------------
# Validator Error Handling Tests
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
