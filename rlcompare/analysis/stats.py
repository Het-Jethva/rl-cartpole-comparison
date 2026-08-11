"""Aggregation statistics.

Reinforcement learning returns across seeds are skewed and often bimodal: a run
either solves the task or collapses. The mean and the standard deviation describe
neither case, and a single failed seed moves both a long way.

The interquartile mean (the mean of the middle 50% of the seeds) is used instead,
with confidence intervals from a bootstrap over the seeds. This follows Agarwal
et al. (2021). Every number in the paper's main table passes through this module,
which is why it is one of the tested seams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20250811


def iqm(values: Sequence[float] | np.ndarray) -> float:
    """Interquartile mean: the mean of the values between the 25th and 75th
    percentile, inclusive of the boundary values.

    With fewer than four values this degenerates to the ordinary mean, which is
    the honest behaviour - there is no interquartile range to speak of.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    if array.size < 4:
        return float(array.mean())
    low, high = np.percentile(array, [25, 75])
    middle = array[(array >= low) & (array <= high)]
    return float(middle.mean()) if middle.size else float(array.mean())


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = iqm,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval over the sample (here: over the seeds).

    Deterministic for a fixed `seed`, so the paper's intervals are reproducible.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, array.size, size=(n_resamples, array.size))
    samples = np.array([statistic(array[row]) for row in idx])
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(samples, [100 * tail, 100 * (1.0 - tail)])
    return (float(low), float(high))


@dataclass
class CurveStats:
    """An aggregated learning curve: point estimate plus interval per step."""

    steps: np.ndarray
    centre: np.ndarray
    low: np.ndarray
    high: np.ndarray
    n_seeds: int


def aggregate_curves(results, *, n_resamples: int = 2_000) -> CurveStats:
    """Aggregate several runs of the same arm into one curve with intervals.

    Fewer bootstrap resamples are used here than for the table: a curve has ~50
    points and the intervals are read visually, not quoted.
    """
    if not results:
        raise ValueError("no results to aggregate")
    steps = np.asarray(results[0].eval_steps, dtype=np.int64)
    matrix = np.asarray([r.eval_mean for r in results], dtype=np.float64)
    if matrix.shape[1] != steps.size:
        raise ValueError("runs in one arm have different evaluation schedules")

    centre = np.array([iqm(column) for column in matrix.T])
    bounds = np.array(
        [bootstrap_ci(column, n_resamples=n_resamples) for column in matrix.T]
    )
    return CurveStats(
        steps=steps,
        centre=centre,
        low=bounds[:, 0],
        high=bounds[:, 1],
        n_seeds=matrix.shape[0],
    )


@dataclass
class ArmSummary:
    """The row of the trade-off table for one arm."""

    arm: str
    label: str
    n_seeds: int
    final_iqm: float
    final_ci: tuple[float, float]
    steps_to_solve_median: float | None
    steps_to_solve_solved_count: int
    failure_rate: float
    wall_clock_mean: float
    wall_clock_std: float
    updates_mean: float
    diagnostics: dict


def summarize_arm(results, label: str | None = None) -> ArmSummary:
    """Reduce every run of one arm to the numbers the paper reports."""
    if not results:
        raise ValueError("no results to summarise")
    finals = np.array([r.final_mean for r in results], dtype=np.float64)
    solved = [r.steps_to_solve for r in results if r.steps_to_solve is not None]
    wall = np.array([r.wall_clock_seconds for r in results], dtype=np.float64)
    updates = np.array([r.update_count for r in results], dtype=np.float64)

    merged: dict = {}
    for key in results[0].diagnostics:
        values = [r.diagnostics.get(key) for r in results]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            merged[key] = float(np.mean(values))
        else:
            merged[key] = values[0]

    return ArmSummary(
        arm=results[0].arm,
        label=label or results[0].arm,
        n_seeds=len(results),
        final_iqm=iqm(finals),
        final_ci=bootstrap_ci(finals),
        steps_to_solve_median=float(np.median(solved)) if solved else None,
        steps_to_solve_solved_count=len(solved),
        failure_rate=1.0 - len(solved) / len(results),
        wall_clock_mean=float(wall.mean()),
        wall_clock_std=float(wall.std(ddof=1)) if wall.size > 1 else 0.0,
        updates_mean=float(updates.mean()),
        diagnostics=merged,
    )


def group_by_arm(results) -> dict[str, list]:
    """Split a flat list of runs into arms, preserving order of first appearance."""
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.arm, []).append(result)
    for runs in grouped.values():
        runs.sort(key=lambda r: r.seed)
    return grouped
