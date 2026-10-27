import csv
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from rlcompare.config import load_yaml
from rlcompare.experiments.matrix import build_all_configs
from rlcompare.experiments.results import load, result_paths
from rlcompare.experiments.tune import build_tuning_configs

study = load_yaml(root / 'configs/study.yaml')
tuned = load_yaml(root / 'configs/tuned.yaml')
tuning, _ = build_tuning_configs(study)
configs = tuning + build_all_configs(study, tuned)
if '--tuning-only' in sys.argv:
    configs = tuning
if '--before-ablations' in sys.argv:
    configs = [c for c in configs if c.group != 'ablation']
counts = Counter()
for config in configs:
    result = load(config, root / 'results')
    json_path, csv_path = result_paths(config, root / 'results')
    assert result.config == config.to_dict(), json_path
    assert result.key == config.key, json_path
    assert (result.group, result.arm, result.method, result.seed) == (config.group, config.arm, config.method, config.seed), json_path
    expected_steps = list(range(0, config.total_steps, config.eval_every)) + [config.total_steps]
    assert result.eval_steps == expected_steps, json_path
    assert len(result.eval_returns) == len(expected_steps), json_path
    assert len(result.eval_mean) == len(expected_steps), json_path
    assert math.isfinite(result.wall_clock_seconds) and result.wall_clock_seconds > 0, json_path
    assert math.isfinite(result.area_under_curve), json_path
    assert result.update_count > 0, json_path
    assert result.train_episode_returns, json_path
    assert len(result.train_episode_returns) == len(result.train_episode_end_steps), json_path
    assert result.train_episode_end_steps == sorted(set(result.train_episode_end_steps)), json_path
    assert 0 < result.train_episode_end_steps[-1] <= config.total_steps, json_path
    assert all(math.isfinite(v) and 0 < v <= 500 for v in result.train_episode_returns), json_path
    for values, mean in zip(result.eval_returns, result.eval_mean):
        assert len(values) == config.eval_episodes, json_path
        assert all(math.isfinite(v) and 0 <= v <= 500 for v in values), json_path
        assert math.isclose(sum(values) / len(values), mean), json_path
    assert result.final_mean == result.eval_mean[-1], json_path
    with csv_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row['step']) for row in rows] == expected_steps, csv_path
    for row, mean, values in zip(rows, result.eval_mean, result.eval_returns):
        assert float(row['mean_return']) == mean, csv_path
        assert [float(row[f'episode_{i}']) for i in range(config.eval_episodes)] == values, csv_path
    counts[config.group] += 1
summary = {
    'verified_runs': len(configs),
    'groups': dict(counts),
    'training_steps': sum(c.total_steps for c in configs),
    'evaluation_points_per_run': len(expected_steps),
    'platform': platform.platform(),
    'python': sys.version,
    'packages': {name: importlib.metadata.version(name) for name in ['torch', 'gymnasium', 'numpy', 'matplotlib', 'pyyaml', 'pytest']},
}
Path(__file__).with_name('verification.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
