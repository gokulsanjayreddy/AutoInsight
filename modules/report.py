"""Report generation module for AutoInsight.

Assembles a downloadable Markdown report containing:
- Dataset summary
- Cleaning log
- Key EDA findings
- Chosen target and task type
- Model comparison table
- Best model's feature importance
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def generate_report(
    dataset_summary: dict,
    cleaning_log: list,
    eda_findings: dict,
    target_column: str,
    task_type: str,
    model_results: dict,
    best_model_name: str,
    feature_importance_fig: object,
) -> str:
    """Generate a Markdown-formatted summary report.

    Parameters
    ----------
    dataset_summary : dict
        Profiling info from data_loader.profile_dataset().
    cleaning_log : list[str]
        Human-readable list of cleaning step messages.
    eda_findings : dict
        Key EDA findings (correlations, missing summary, etc.).
    target_column : str
        The selected target column name.
    task_type : str
        Either "classification" or "regression".
    model_results : dict
        Model results from trainer.train_*_models().
    best_model_name : str
        Name of the best-performing model.
    feature_importance_fig : matplotlib.figure.Figure
        Feature importance plot figure.
    Returns
    -------
    str
        Complete Markdown report text.
    """
    lines = []

    # Title
    lines.append("# AutoInsight — Automated Data Analysis Report")
    lines.append("")

    # ---- Dataset Summary ----
    lines.append("## Dataset Summary")
    lines.append(f"- **Rows:** {dataset_summary.get('rows', 'N/A'):,}")
    lines.append(f"- **Columns:** {dataset_summary.get('columns', 'N/A')}")
    lines.append(f"- **Memory Usage:** {dataset_summary.get('memory_mb', 'N/A')} MB")
    lines.append(f"- **Duplicate Rows:** {dataset_summary.get('duplicate_rows', 'N/A')}")
    lines.append(f"- **Missing Values %:** {dataset_summary.get('missing_percent', {})}")
    lines.append("")

    # ---- Cleaning Log ----
    lines.append("## Cleaning Log")
    for entry in cleaning_log:
        lines.append(f"- {entry}")
    lines.append("")

    # ---- EDA Findings ----
    lines.append("## Key EDA Findings")
    if "correlation_highlights" in eda_findings:
        lines.append(eda_findings["correlation_highlights"])
    if "missing_summary" in eda_findings:
        lines.append(eda_findings["missing_summary"])
    if "distribution_notes" in eda_findings:
        lines.append(eda_findings["distribution_notes"])
    lines.append("")

    # ---- Target & Task Type ----
    lines.append("## Target Column & Task Type")
    lines.append(f"- **Target Column:** `{target_column}`")
    lines.append(f"- **Task Type:** {task_type.capitalize()}")
    lines.append("")

    # ---- Model Comparison ----
    lines.append("## Model Comparison")
    lines.append("| Model | Score |")
    lines.append("|---------|-------|")

    if task_type == "classification":
        for name, res in model_results.items():
            score = f"{res.get('accuracy', 0):.4f}"
            lines.append(f"| {name} | {score} |")
    else:
        for name, res in model_results.items():
            score = f"{res.get('r2', 0):.4f}"
            lines.append(f"| {name} | {score} |")

    lines.append("")

    # ---- Best Model Feature Importance ----
    lines.append("## Best Model Feature Importance")
    lines.append(
        f"*Best model: `{best_model_name}`*"
    )
    if feature_importance_fig is not None:
        lines.append("")
        lines.append("![Feature Importance]({feature_importance_placeholder})")
    lines.append("")

    # ---- Notes ----
    lines.append("---")
    lines.append("*Report generated automatically by AutoInsight.*")

    return "\n".join(lines)