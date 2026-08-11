"""Figure generation.

Every figure in the paper is produced here from the raw result files. None is
drawn by hand, so rerunning the study updates the paper without manual work.

Figures are sized for an IEEE two-column page: a single-column figure is 3.4
inches wide, a full-width one 7.1 inches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..experiments.results import load_group
from .stats import aggregate_curves, group_by_arm, iqm

COLUMN_WIDTH = 3.4
PAGE_WIDTH = 7.16

METHOD_LABELS = {
    "tabular": "Tabular Q-learning",
    "dqn": "DQN",
    "reinforce": "REINFORCE + baseline",
}
METHOD_COLOURS = {
    "tabular": "#4C72B0",
    "dqn": "#DD8452",
    "reinforce": "#55A868",
}
ABLATION_COLOUR = "#C44E52"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _plot_curve(ax, stats, colour: str, label: str) -> None:
    ax.plot(stats.steps, stats.centre, color=colour, label=label)
    ax.fill_between(stats.steps, stats.low, stats.high, color=colour, alpha=0.18, linewidth=0)


def _save(fig, out: Path, name: str) -> Path:
    """Write the PDF the paper includes, and a PNG for the README.

    Returns the PDF path, which is what the paper references.
    """
    pdf = out / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    return pdf


def _legend_box(ax, **kwargs) -> None:
    """Place a legend in the empty lower-right corner, over a translucent patch.

    Every curve here trends upwards, so the upper-left corner that matplotlib
    prefers is exactly where the data is.
    """
    ax.legend(
        loc="lower right", frameon=True, framealpha=0.9,
        edgecolor="none", borderpad=0.4, **kwargs,
    )


def _thousands(ax) -> None:
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k")
    )


def figure_learning_curves(study: dict[str, Any], results_root: Path, out: Path) -> Path:
    """Figure 1: the headline comparison."""
    _apply_style()
    runs = group_by_arm(load_group("main", results_root))
    threshold = study["run"]["solved_threshold"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    for method in study["methods"]:
        if method not in runs:
            continue
        stats = aggregate_curves(runs[method])
        _plot_curve(ax, stats, METHOD_COLOURS[method], METHOD_LABELS[method])

    ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.8)
    ax.text(
        0.02, threshold + 8, "solved", ha="left", va="bottom",
        transform=ax.get_yaxis_transform(), color="0.35", fontsize=6.5,
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.set_ylim(0, 540)
    _thousands(ax)
    _legend_box(ax)

    return _save(fig, out, "learning_curves")


def figure_sweeps(study: dict[str, Any], results_root: Path, out: Path) -> Path:
    """Figure 2: one sensitivity sweep per method."""
    _apply_style()
    runs = group_by_arm(load_group("sweep", results_root))
    methods = list(study["sweeps"])

    fig, axes = plt.subplots(1, len(methods), figsize=(PAGE_WIDTH, 2.1), sharey=True)
    for ax, method in zip(np.atleast_1d(axes), methods):
        spec = study["sweeps"][method]
        parameter = spec["parameter"]
        values = spec["values"]
        colours = plt.cm.viridis(np.linspace(0.08, 0.9, len(values)))
        for colour, value in zip(colours, values):
            arm = f"{method}:{parameter}={_format(value)}"
            if arm not in runs:
                continue
            stats = aggregate_curves(runs[arm], n_resamples=500)
            ax.plot(stats.steps, stats.centre, color=colour, label=_legend(value))
        ax.axhline(study["run"]["solved_threshold"], color="0.35", linestyle="--", linewidth=0.7)
        ax.set_title(f"{METHOD_LABELS[method]}\n({spec['label']})")
        ax.set_xlabel("Environment steps")
        ax.set_ylim(0, 540)
        _thousands(ax)
        _legend_box(ax, ncols=2, handlelength=1.0)
    np.atleast_1d(axes)[0].set_ylabel("Evaluation return")

    return _save(fig, out, "sweeps")


def figure_ablations(study: dict[str, Any], results_root: Path, out: Path) -> Path:
    """Figure 3: each ablation against its full method."""
    _apply_style()
    ablation_runs = group_by_arm(load_group("ablation", results_root))
    main_runs = group_by_arm(load_group("main", results_root))
    names = list(study["ablations"])

    fig, axes = plt.subplots(1, len(names), figsize=(PAGE_WIDTH * 0.72, 2.1), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        spec = study["ablations"][name]
        method = spec["method"]
        if method in main_runs:
            stats = aggregate_curves(main_runs[method])
            _plot_curve(ax, stats, METHOD_COLOURS[method], "full method")
        arm = f"{method}:{name}"
        if arm in ablation_runs:
            stats = aggregate_curves(ablation_runs[arm])
            _plot_curve(ax, stats, ABLATION_COLOUR, "component removed")
        ax.axhline(study["run"]["solved_threshold"], color="0.35", linestyle="--", linewidth=0.7)
        ax.set_title(spec["label"])
        ax.set_xlabel("Environment steps")
        ax.set_ylim(0, 540)
        _thousands(ax)
        _legend_box(ax)
    np.atleast_1d(axes)[0].set_ylabel("Evaluation return")

    return _save(fig, out, "ablations")


def figure_seed_scatter(study: dict[str, Any], results_root: Path, out: Path) -> Path:
    """Figure 4: every seed's final performance, so bimodal outcomes are visible."""
    _apply_style()
    runs = group_by_arm(load_group("main", results_root))
    methods = [m for m in study["methods"] if m in runs]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    rng = np.random.default_rng(0)
    for position, method in enumerate(methods):
        finals = np.array([r.final_mean for r in runs[method]])
        jitter = rng.uniform(-0.14, 0.14, size=finals.size)
        ax.scatter(
            np.full(finals.size, position) + jitter, finals,
            s=16, color=METHOD_COLOURS[method], alpha=0.8, edgecolors="none",
        )
        centre = iqm(finals)
        ax.plot([position - 0.26, position + 0.26], [centre, centre], color="0.15", linewidth=1.4)

    ax.axhline(study["run"]["solved_threshold"], color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m].replace(" + ", "\n+ ") for m in methods])
    ax.set_ylabel("Final evaluation return")
    ax.set_ylim(0, 540)
    ax.set_xlim(-0.5, len(methods) - 0.5)

    return _save(fig, out, "seed_scatter")


def _format(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def _legend(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return f"{int(np.prod(value))} cells"
    return str(value)


def build_all_figures(study: dict[str, Any], results_root: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for builder in (
        figure_learning_curves,
        figure_sweeps,
        figure_ablations,
        figure_seed_scatter,
    ):
        try:
            written.append(builder(study, results_root, out))
        except (ValueError, KeyError, IndexError) as error:
            print(f"  skipped {builder.__name__}: {error}")
    return written
