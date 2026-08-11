"""Aggregation, figures and tables. Everything the paper shows comes from here."""

from .stats import ArmSummary, CurveStats, aggregate_curves, bootstrap_ci, iqm, summarize_arm

__all__ = [
    "ArmSummary",
    "CurveStats",
    "aggregate_curves",
    "bootstrap_ci",
    "iqm",
    "summarize_arm",
]
