"""Command line entry point for the whole study.

    python run.py tune       # equal-budget random search, writes configs/tuned.yaml
    python run.py matrix     # main comparison, sweeps and ablations
    python run.py analyse    # figures and tables into paper/
    python run.py all        # the three above, in order

`python run.py single` runs one configuration for debugging.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rlcompare.config import RunConfig, load_yaml

ROOT = Path(__file__).parent
STUDY = ROOT / "configs" / "study.yaml"
TUNED = ROOT / "configs" / "tuned.yaml"
RESULTS = ROOT / "results"


def _load_tuned() -> dict:
    if not TUNED.is_file():
        return {}
    return load_yaml(TUNED)


def command_single(args: argparse.Namespace) -> int:
    from rlcompare.experiments.runner import run_experiment

    study = load_yaml(STUDY)
    base = dict(study.get("run", {}))
    if args.steps:
        base["total_steps"] = args.steps
    if args.eval_every:
        base["eval_every"] = args.eval_every
    config = RunConfig(
        method=args.method,
        seed=args.seed,
        group="debug",
        params=_load_tuned().get(args.method, {}),
        **base,
    )
    started = time.perf_counter()
    result = run_experiment(config)
    print(f"method          {config.method}")
    print(f"seed            {config.seed}")
    print(f"final eval mean {result.final_mean:.1f}")
    print(f"steps to solve  {result.steps_to_solve}")
    print(f"updates         {result.update_count}")
    print(f"wall clock      {time.perf_counter() - started:.1f}s")
    print(f"diagnostics     {result.diagnostics}")
    return 0


def command_tune(args: argparse.Namespace) -> int:
    from rlcompare.experiments.tune import tune

    study = load_yaml(STUDY)
    print("Tuning: equal-budget random search")
    tune(study, workers=args.workers, root=RESULTS, output=TUNED)
    print(f"Wrote {TUNED}")
    return 0


def command_matrix(args: argparse.Namespace) -> int:
    from rlcompare.experiments.matrix import (
        build_ablation_configs,
        build_main_configs,
        build_sweep_configs,
        run_all,
    )

    study = load_yaml(STUDY)
    tuned = _load_tuned()
    if not tuned:
        print("No configs/tuned.yaml found. Run `python run.py tune` first.", file=sys.stderr)
        return 1

    stages = [
        ("main comparison", build_main_configs(study, tuned)),
        ("sensitivity sweeps", build_sweep_configs(study, tuned)),
        ("ablations", build_ablation_configs(study, tuned)),
    ]
    for name, configs in stages:
        print(f"{name}: {len(configs)} run(s)")
        run_all(configs, workers=args.workers, root=RESULTS)
    return 0


def command_analyse(args: argparse.Namespace) -> int:
    from rlcompare.analysis.figures import build_all_figures
    from rlcompare.analysis.tables import build_all_tables

    study = load_yaml(STUDY)
    figure_dir = ROOT / "paper" / "figures"
    table_dir = ROOT / "paper" / "tables"
    written = build_all_figures(study, RESULTS, figure_dir)
    written += build_all_tables(
        study, _load_tuned(), RESULTS, table_dir, package_root=ROOT / "rlcompare"
    )
    for path in written:
        print(f"  wrote {path.relative_to(ROOT)}")
    return 0


def command_all(args: argparse.Namespace) -> int:
    for command in (command_tune, command_matrix, command_analyse):
        code = command(args)
        if code != 0:
            return code
    return 0


def command_demo(args: argparse.Namespace) -> int:
    from rlcompare.demo import serve_demo

    serve_demo(ROOT, port=args.port, open_browser=not args.no_browser)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12, help="parallel worker processes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="run one configuration")
    single.add_argument("--method", required=True, choices=["tabular", "dqn", "reinforce"])
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--steps", type=int, default=None)
    single.add_argument("--eval-every", type=int, default=None, dest="eval_every")
    single.set_defaults(func=command_single)

    demo = subparsers.add_parser("demo", help="watch the agents train in a local browser")
    demo.add_argument("--port", type=int, default=8765, help="local HTTP port, default 8765")
    demo.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    demo.set_defaults(func=command_demo)

    subparsers.add_parser("tune", help="equal-budget random search").set_defaults(func=command_tune)
    subparsers.add_parser("matrix", help="main, sweeps and ablations").set_defaults(func=command_matrix)
    subparsers.add_parser("analyse", help="figures and tables").set_defaults(func=command_analyse)
    subparsers.add_parser("all", help="tune, matrix, analyse").set_defaults(func=command_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
