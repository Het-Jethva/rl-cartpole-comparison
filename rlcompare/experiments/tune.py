"""Equal-budget random search over hyperparameters.

Every method gets the same number of trials and the same number of tuning seeds.
That equality is the defence against the most common objection to a comparison
paper: that the result reflects how much effort the author spent on each method.

The tuning seeds are disjoint from the evaluation seeds, so the reported results
are not tuned on their own test set.

Trials are scored by area under the evaluation curve, which rewards both reaching
a high return and reaching it early - the quantity the paper is actually about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import ConfigError, RunConfig
from . import results as results_io
from .matrix import _base_settings, run_all


def sample_trials(
    space: dict[str, list[Any]], n_trials: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Draw distinct hyperparameter combinations from a discrete search space.

    Duplicates are dropped rather than re-drawn forever: if the space is smaller
    than the trial budget, the search simply becomes exhaustive.
    """
    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(trials) < n_trials and attempts < n_trials * 50:
        attempts += 1
        trial = {}
        for name, choices in space.items():
            index = int(rng.integers(len(choices)))
            value = choices[index]
            trial[name] = tuple(value) if isinstance(value, list) else value
        signature = repr(sorted(trial.items()))
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
    return trials


def build_tuning_configs(study: dict[str, Any]) -> tuple[list[RunConfig], dict[str, list]]:
    """Build every tuning run, and return the trial table used to score them."""
    base = _base_settings(study)
    seeds = study["seeds"]["tuning"]
    n_trials = study["tuning"]["trials"]
    rng = np.random.default_rng(study["tuning"]["search_seed"])

    configs: list[RunConfig] = []
    trials_by_method: dict[str, list] = {}
    for method in study["methods"]:
        space = study["tuning"]["spaces"][method]
        trials = sample_trials(space, n_trials, rng)
        if len(trials) < n_trials:
            print(
                f"  note: {method} search space yields only {len(trials)} distinct "
                f"trials (requested {n_trials})"
            )
        trials_by_method[method] = trials
        for index, trial in enumerate(trials):
            for seed in seeds:
                configs.append(
                    RunConfig(
                        method=method,
                        seed=seed,
                        group="tuning",
                        variant=f"trial{index:02d}",
                        params=trial,
                        **base,
                    )
                )
    return configs, trials_by_method


def select_best(
    study: dict[str, Any],
    trials_by_method: dict[str, list],
    root: Path | str = results_io.RESULTS_ROOT,
) -> dict[str, dict[str, Any]]:
    """Score every trial by mean area under the curve and keep the winner."""
    base = _base_settings(study)
    seeds = study["seeds"]["tuning"]
    best: dict[str, dict[str, Any]] = {}
    scoreboard: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}

    for method, trials in trials_by_method.items():
        rows = []
        for index, trial in enumerate(trials):
            scores = []
            for seed in seeds:
                config = RunConfig(
                    method=method,
                    seed=seed,
                    group="tuning",
                    variant=f"trial{index:02d}",
                    params=trial,
                    **base,
                )
                if not results_io.exists(config, root):
                    raise ConfigError(f"missing tuning result for {config.key}")
                scores.append(results_io.load(config, root).area_under_curve)
            rows.append((float(np.mean(scores)), index, trial))
        rows.sort(key=lambda row: row[0], reverse=True)
        scoreboard[method] = rows
        best[method] = _jsonable(rows[0][2])
        print(
            f"  {method}: best trial {rows[0][1]:02d} "
            f"(score {rows[0][0]:.3g}) -> {best[method]}"
        )

    return best


def write_tuned(best: dict[str, dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Selected by rlcompare's equal-budget random search.\n"
        "# Regenerate with: python run.py tune\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(best, handle, sort_keys=True, default_flow_style=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def tune(
    study: dict[str, Any],
    *,
    workers: int,
    root: Path | str,
    output: Path | str,
) -> dict[str, dict[str, Any]]:
    configs, trials_by_method = build_tuning_configs(study)
    run_all(configs, workers=workers, root=root)
    best = select_best(study, trials_by_method, root)
    write_tuned(best, output)
    return best
