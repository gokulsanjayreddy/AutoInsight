"""End-to-end pipeline simulation test for AutoInsight."""

import io
import unittest
import zipfile

from PIL import Image

from modules.data_loader import get_sample_dataset, profile_dataset
from modules.eda import (
    create_all_plots_zip,
    plot_cat_cat_relationships,
    plot_cat_num_relationships,
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_distributions,
    plot_missing_values,
    plot_numeric_boxplots,
    plot_pairwise_scatter,
    plot_summary_statistics,
)


class TestFullPipeline(unittest.TestCase):
    def test_pipeline_end_to_end(self):
        df = get_sample_dataset()
        profile = profile_dataset(df)
        self.assertGreater(profile["rows"], 0)

        all_named_figures = []

        fig_sum = plot_summary_statistics(df)
        all_named_figures.append(("00_summary", "summary_statistics", fig_sum))

        fig_miss = plot_missing_values(df)
        if fig_miss:
            all_named_figures.append(("00_summary", "missing_values", fig_miss))

        dists = plot_distributions(df)
        for col_name, fig in dists:
            all_named_figures.append(("01_distributions", f"dist_{col_name}", fig))

        boxes = plot_numeric_boxplots(df)
        for col, fig in boxes:
            all_named_figures.append(("02_boxplots", f"box_{col}", fig))

        cats = plot_categorical_bars(df)
        for i, fig in enumerate(cats):
            all_named_figures.append(("03_categorical", f"cat_{i}", fig))

        fig_corr = plot_correlation_heatmap(df)
        all_named_figures.append(("04_correlation", "corr_matrix", fig_corr))

        scatters = plot_pairwise_scatter(df)
        for i, fig in enumerate(scatters):
            all_named_figures.append(("05_scatter", f"scatter_{i}", fig))

        cat_nums = plot_cat_num_relationships(df)
        for label, fig in cat_nums:
            all_named_figures.append(("06_cat_num", f"cn_{label}", fig))

        cat_cats = plot_cat_cat_relationships(df)
        for label, fig in cat_cats:
            all_named_figures.append(("07_cat_cat", f"cc_{label}", fig))

        self.assertGreaterEqual(len(all_named_figures), 20)

        zip_bytes = create_all_plots_zip(all_named_figures)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            namelist = zf.namelist()
            self.assertGreaterEqual(len(namelist), 20)
            for fname in namelist:
                self.assertTrue(fname.endswith(".png"))
                png_data = zf.read(fname)
                self.assertTrue(png_data.startswith(b"\x89PNG\r\n\x1a\n"))
                # Confirm valid decodable image
                with Image.open(io.BytesIO(png_data)) as img:
                    self.assertGreater(img.width, 0)
                    self.assertGreater(img.height, 0)


if __name__ == "__main__":
    unittest.main()
