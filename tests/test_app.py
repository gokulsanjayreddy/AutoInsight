"""Streamlit AppTest integration tests for app.py."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from modules.data_loader import get_sample_dataset

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_smoke_demo_dataset():
    """Run app.py with the demo dataset in session_state and assert no exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["df"] = get_sample_dataset()
    at.session_state["source_name"] = "sample_employee_analytics.csv"
    at.session_state["uploaded_file_id"] = "demo"
    at.run()
    assert not at.exception


def test_app_reset_while_file_uploaded():
    """Resetting dataset while data exists clears state and increments key."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    at.session_state["df"] = df
    at.session_state["source_name"] = "data.csv"
    at.session_state["uploaded_file_id"] = "data.csv_100_123"
    at.session_state["uploader_key"] = 0
    at.run()
    assert not at.exception

    reset_btn = None
    for b in at.button:
        if b.label == "Reset Dataset":
            reset_btn = b
            break
    assert reset_btn is not None
    reset_btn.click().run()
    assert not at.exception
    assert at.session_state["df"] is None
    assert at.session_state["uploader_key"] == 1


def test_app_changing_data_refreshes_charts():
    """Changing data with same shape and name refreshes cached hash and charts."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    at.session_state["df"] = df1
    at.session_state["source_name"] = "data.csv"
    at.session_state["uploaded_file_id"] = "data_id"
    at.run()
    hash1 = at.session_state["cached_hash"]

    df2 = pd.DataFrame({"a": [10, 20, 30], "b": [40, 50, 60]})
    at.session_state["df"] = df2
    at.run()
    hash2 = at.session_state["cached_hash"]
    assert hash1 != hash2
