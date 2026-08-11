"""Building and executing the run matrix.

The matrix is derived from two files: `configs/study.yaml` (the study design) and
`configs/tuned.yaml` (the hyperparameters the tuning stage selected). Nothing in
the matrix is hard-coded here, so changing the study means editing YAML rather
than editing code.

Execution is a process pool. Each worker runs one complete run and writes its own
files, so a stopped run costs at most one run's work: on restart, any config whose
result file already exists is skipped.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from ..config import ConfigError, RunConfig, load_yaml
from . import results as results_io
from .runner import run_experiment

DEFAULT_WORKERS = 12


def _base_settings(study: dict[str, Any]) -> dict[str, Any]:
    known = {"env_id", "total_steps", "eval_every", "eval_episodes", "solved_threshold"}
    unknown = set(study.get("run", {})) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in study.run: {sorted(unknown)}")
    return dict(study.get("run", {}))


def build_main_configs(study: dict[str, Any], tuned: dict[str, Any]) -> list[RunConfig]:
    """The headline comparison: every method, every evaluation seed."""
    base = _base_settings(study)
    seeds = study["seeds"]["evaluation"]
    configs = []
    for method in study["methods"]:
        params = dict(tuned.get(method, {}))
        for seed in seeds:
            configs.append(
                RunConfig(method=method, seed=seed, group="main", params=params, **base)
            )
    return configs


def build_sweep_configs(study: dict[str, Any], tuned: dict[str, Any]) -> list[RunConfig]:
    """One sensitivity sweep per method, over the knob that carries its trade-off."""
    base = _base_settings(study)
    seeds = study["seeds"]["sweep"]
    configs = []
    for method, spec in study["sweeps"].items():
        parameter = spec["parameter"]
        for value in spec["values"]:
            params = dict(tuned.get(method, {}))
            params[parameter] = value
            label = _format_value(value)
            for seed in seeds:
                configs.append(
                    RunConfig(
                        method=method,
                        seed=seed,
                        group="sweep",
                        variant=f"{parameter}={label}",
                        params=params,
                        **base,
                    )
                )
    return configs


def build_ablation_configs(study: dict[str, Any], tuned: dict[str, Any]) -> list[RunConfig]:
    """Component removals, each paired with its full method as the control."""
    base = _base_settings(study)
    seeds = study["seeds"]["evaluation"]
    configs = []
    for name, spec in study["ablations"].items():
        method = spec["method"]
        params = dict(tuned.get(method, {}))
        params.update(spec["params"])
        for seed in seeds:
            configs.append(
                RunConfig(
                    method=method,
                    seed=seed,
                    group="ablation",
                    variant=name,
                    params=params,
                    **base,
                )
            )
    return configs


def build_all_configs(study: dict[str, Any], tuned: dict[str, Any]) -> list[RunConfig]:
    return (
        build_main_configs(study, tuned)
        + build_sweep_configs(study, tuned)
        + build_ablation_configs(study, tuned)
    )


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def _execute(payload: tuple[dict[str, Any], str]) -> tuple[str, float, float]:
    """Pool worker: run one config and write its results."""
    config_data, root = payload
    config = RunConfig.from_dict(config_data)
    result = run_experiment(config)
    results_io.save(result, config, root)
    return config.key, result.final_mean, result.wall_clock_seconds


def run_all(
    configs: Iterable[RunConfig],
    *,
    workers: int = DEFAULT_WORKERS,
    root: Path | str = results_io.RESULTS_ROOT,
    force: bool = False,
) -> int:
    """Run every config that has no result yet. Returns the number executed."""
    configs = list(configs)
    pending = [c for c in configs if force or not results_io.exists(c, root)]
    skipped = len(configs) - len(pending)
    if skipped:
        print(f"  {skipped} run(s) already complete, skipping")
    if not pending:
        return 0

    workers = max(1, min(workers, os.cpu_count() or 1, len(pending)))
    print(f"  running {len(pending)} run(s) on {workers} worker(s)")

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute, (c.to_dict(), str(root))): c for c in pending
        }
        for future in as_completed(futures):
            config = futures[future]
            key, final_mean, seconds = future.result()
            completed += 1
            print(
                f"  [{completed:>4}/{len(pending)}] {key}  "
                f"final={final_mean:7.1f}  {seconds:6.1f}s",
                flush=True,
            )
    return completed
