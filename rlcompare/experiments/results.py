"""Result records and their storage.

One run writes two files: a CSV of the evaluation curve (readable on its own, and
what a marker can check the figures against) and a JSON holding the full config,
the seed, the timings and the diagnostics.

Files are named by `RunConfig.key`, which is derived from the config. That is
what makes the matrix runner resume-safe.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import RunConfig

RESULTS_ROOT = Path("results")


@dataclass
class RunResult:
    """Everything one run produces."""

    config: dict[str, Any]
    key: str
    group: str
    arm: str
    method: str
    seed: int
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[list[float]] = field(default_factory=list)
    eval_mean: list[float] = field(default_factory=list)
    train_episode_returns: list[float] = field(default_factory=list)
    train_episode_end_steps: list[int] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    update_count: int = 0
    steps_to_solve: int | None = None
    final_mean: float = 0.0
    area_under_curve: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunResult:
        return cls(**data)


def run_directory(group: str, root: Path | str = RESULTS_ROOT) -> Path:
    return Path(root) / group


def result_paths(config: RunConfig, root: Path | str = RESULTS_ROOT) -> tuple[Path, Path]:
    directory = run_directory(config.group, root)
    return directory / f"{config.key}.json", directory / f"{config.key}.csv"


def exists(config: RunConfig, root: Path | str = RESULTS_ROOT) -> bool:
    json_path, _ = result_paths(config, root)
    return json_path.is_file()


def save(result: RunResult, config: RunConfig, root: Path | str = RESULTS_ROOT) -> None:
    json_path, csv_path = result_paths(config, root)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_json(), handle, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        n_episodes = len(result.eval_returns[0]) if result.eval_returns else 0
        writer.writerow(
            ["step", "mean_return"] + [f"episode_{i}" for i in range(n_episodes)]
        )
        for step, mean, returns in zip(
            result.eval_steps, result.eval_mean, result.eval_returns
        ):
            writer.writerow([step, mean, *returns])


def load(config: RunConfig, root: Path | str = RESULTS_ROOT) -> RunResult:
    json_path, _ = result_paths(config, root)
    with open(json_path, "r", encoding="utf-8") as handle:
        return RunResult.from_json(json.load(handle))


def load_group(group: str, root: Path | str = RESULTS_ROOT) -> list[RunResult]:
    """Load every result in a group, sorted by arm then seed."""
    directory = run_directory(group, root)
    if not directory.is_dir():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            results.append(RunResult.from_json(json.load(handle)))
    results.sort(key=lambda r: (r.arm, r.seed))
    return results
