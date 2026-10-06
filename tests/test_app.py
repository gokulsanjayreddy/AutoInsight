"""Streamlit AppTest integration tests for app.py."""

from pathlib import Path

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
