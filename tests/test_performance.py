"""Tests for Section 2 performance and memory optimizations."""

import io

import matplotlib.pyplot as plt
from PIL import Image

from modules.data_loader import get_sample_dataset
from modules.eda import create_all_plots_zip


def test_2_1_create_all_plots_zip_accepts_prerendered_bytes():
    """create_all_plots_zip must accept pre-rendered (folder, name, bytes) tuples."""
    # Create simple 1x1 PNG bytes
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    items = [
        ("00_summary", "test_plot", raw_bytes),
        ("01_dists", "dist_a", raw_bytes),
    ]
    zip_bytes = create_all_plots_zip(items)
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0


def test_2_4_pipeline_module_has_no_streamlit_imports():
    """modules.pipeline must not import streamlit."""
    import modules.pipeline

    # Check that streamlit is not in pipeline's namespace
    assert not hasattr(modules.pipeline, "st")
    assert not hasattr(modules.pipeline, "streamlit")
    with open("modules/pipeline.py", encoding="utf-8") as f:
        content = f.read()
    assert "import streamlit" not in content
    assert "from streamlit" not in content


def test_2_1_and_2_3_pipeline_closes_figures_and_respects_cap():
    """generate_full_analysis in modules.pipeline closes figures and respects max_total_plots."""
    from modules.pipeline import generate_full_analysis

    df = get_sample_dataset()
    # Close any open figures first
    plt.close("all")

    progress_calls = []

    def on_progress(pct: float, msg: str):
        progress_calls.append((pct, msg))

    cap = 15
    res = generate_full_analysis(df, max_total_plots=cap, progress_callback=on_progress)

    # Total plots must be at or below cap
    assert res["total_count"] <= cap
    # Progress callback was called
    assert len(progress_calls) > 0
    # Matplotlib figures must all be closed (0 open figures)
    assert len(plt.get_fignums()) == 0
    # Results contain plain bytes, not Figure objects
    for card_list in res["card_data"].values():
        for item in card_list:
            # item is (title, fname, png_bytes)
            assert isinstance(item[2], bytes)
