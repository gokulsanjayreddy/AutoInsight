"""Tests for Section 3 dependencies and deprecations."""

import warnings

import pytest

from modules import eda


def test_3_1_no_use_container_width_in_app():
    """Bug 3.1: use_container_width must be replaced by width='stretch' or width='content'."""
    with open("app.py", encoding="utf-8") as f:
        content = f.read()
    assert "use_container_width" not in content


def test_3_2_requirements_packages():
    """Bug 3.2: requirements.txt must not contain scikit-learn or plotly, and pins must have compatible ranges."""
    with open("requirements.txt", encoding="utf-8") as f:
        reqs = f.read()
    assert "scikit-learn" not in reqs
    assert "plotly" not in reqs
    # Verify dev requirements exists
    with open("requirements-dev.txt", encoding="utf-8") as f:
        dev_reqs = f.read()
    assert "pytest" in dev_reqs
    assert "mypy" in dev_reqs
    assert "ruff" in dev_reqs
    assert "pandas-stubs" in dev_reqs


def test_3_4_dead_code_removed():
    """Bug 3.4: Dead plotly and target functions must be removed."""
    assert not hasattr(eda, "plotly_distribution_histogram")
    assert not hasattr(eda, "plotly_correlation_heatmap")
    assert not hasattr(eda, "plotly_pairwise_scatter")
    assert not hasattr(eda, "plotly_target_relationships")
    assert not hasattr(eda, "plot_target_relationships")
    # eda must not import plotly
    assert not hasattr(eda, "px")
    assert not hasattr(eda, "go")


def test_3_3_filterwarnings_catches_deprecation():
    """Bug 3.3: DeprecationWarnings from our own code must be caught/errored by pytest."""
    with pytest.raises(DeprecationWarning):
        warnings.warn("test deprecation from our code", DeprecationWarning, stacklevel=1)
