"""Automated unit tests for AutoInsight EDA engine and plot separation."""

import io
import unittest
import zipfile

import matplotlib.pyplot as plt
import pandas as pd

from modules.data_loader import get_sample_dataset, profile_dataset
from modules.eda import (
    create_all_plots_zip,
    fig_to_png_bytes,
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


class TestAutoInsightEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_sample_dataset()

    def tearDown(self):
        plt.close("all")

    def test_sample_dataset(self):
        self.assertFalse(self.df.empty)
        self.assertGreaterEqual(len(self.df), 100)
        profile = profile_dataset(self.df)
        self.assertIn("rows", profile)
        self.assertIn("column_types", profile)
        self.assertGreater(profile["rows"], 0)

    def test_summary_statistics_figure(self):
        fig = plot_summary_statistics(self.df)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_distributions_separation(self):
        figures = plot_distributions(self.df)
        self.assertIsInstance(figures, list)
        self.assertGreater(len(figures), 0)
        for fig in figures:
            self.assertIsInstance(fig, plt.Figure)
            self.assertGreaterEqual(len(fig.axes), 1)

    def test_plot_numeric_boxplots(self):
        results = plot_numeric_boxplots(self.df)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for col_name, fig in results:
            self.assertIsInstance(col_name, str)
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_categorical_bars(self):
        figures = plot_categorical_bars(self.df)
        self.assertIsInstance(figures, list)
        self.assertGreater(len(figures), 0)
        for fig in figures:
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_correlation_heatmap(self):
        fig = plot_correlation_heatmap(self.df)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_pairwise_scatter_separation(self):
        figures = plot_pairwise_scatter(self.df)
        self.assertIsInstance(figures, list)
        self.assertGreater(len(figures), 0)
        for fig in figures:
            self.assertIsInstance(fig, plt.Figure)
            self.assertGreaterEqual(len(fig.axes), 1)

    def test_plot_cat_num_relationships(self):
        results = plot_cat_num_relationships(self.df)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for label, fig in results:
            self.assertIsInstance(label, str)
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_cat_cat_relationships(self):
        results = plot_cat_cat_relationships(self.df)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for label, fig in results:
            self.assertIsInstance(label, str)
            self.assertIsInstance(fig, plt.Figure)

    def test_plot_missing_values(self):
        fig = plot_missing_values(self.df)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

        clean_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.assertIsNone(plot_missing_values(clean_df))

    def test_png_bytes_and_zip_creation(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        png_data = fig_to_png_bytes(fig)
        self.assertIsInstance(png_data, bytes)
        self.assertGreater(len(png_data), 100)

        named_figures = [
            ("01_distributions", "test_dist", fig),
            ("02_correlations", "test_corr", fig),
        ]
        zip_bytes = create_all_plots_zip(named_figures)
        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 200)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            file_list = zf.namelist()
            self.assertIn("01_distributions/test_dist.png", file_list)
            self.assertIn("02_correlations/test_corr.png", file_list)


if __name__ == "__main__":
    unittest.main()
