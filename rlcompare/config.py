"""Configuration schemas.

One `RunConfig` fully determines one run. Every field that changes behaviour is
part of the config, and the config is written next to the results, so any curve
in the paper traces back to the exact settings that produced it.

Unknown keys are an error. A silently ignored hyperparameter name would produce
results for settings the paper does not describe.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any

DEFAULT_ENV = "CartPole-v1"


class ConfigError(ValueError):
    """Raised when a config file or override does not match the schema."""


@dataclass(frozen=True)
class TabularParams:
    """Tabular Q-learning over a binned state space.

    `clip_velocity` bounds the two unbounded CartPole dimensions (cart velocity
    and pole angular velocity) so that they can be binned at all.
    """

    bins: tuple[int, int, int, int] = (6, 6, 12, 12)
    clip_velocity: tuple[float, float] = (3.0, 3.5)
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000


@dataclass(frozen=True)
class DQNParams:
    """Canonical DQN (Mnih et al., 2015).

    `use_target_network=False` is the ablation arm: the bootstrap target is then
    computed with the online network.
    """

    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_interval: int = 500
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    use_target_network: bool = True


@dataclass(frozen=True)
class ReinforceParams:
    """REINFORCE with a learned state-value baseline (Williams, 1992).

    `use_baseline=False` is the ablation arm: the advantage is then the raw
    Monte-Carlo return. Returns are never whitened, because dividing by the
    standard deviation rescales the gradient and is therefore not a baseline.
    """

    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    baseline_lr: float = 1e-3
    gamma: float = 0.99
    episodes_per_update: int = 1
    grad_clip: float = 10.0
    use_baseline: bool = True


PARAM_CLASSES: dict[str, type] = {
    "tabular": TabularParams,
    "dqn": DQNParams,
    "reinforce": ReinforceParams,
}

# Fields that must be tuples once loaded from YAML (which produces lists).
_TUPLE_FIELDS = {"bins", "hidden", "clip_velocity"}


def build_params(method: str, overrides: dict[str, Any] | None = None):
    """Construct a method's parameter dataclass, rejecting unknown keys."""
    if method not in PARAM_CLASSES:
        raise ConfigError(
            f"unknown method {method!r}; expected one of {sorted(PARAM_CLASSES)}"
        )
    cls = PARAM_CLASSES[method]
    overrides = dict(overrides or {})
    known = {f.name for f in fields(cls)}
    unknown = set(overrides) - known
    if unknown:
        raise ConfigError(
            f"unknown parameter(s) for method {method!r}: {sorted(unknown)}; "
            f"known parameters are {sorted(known)}"
        )
    for key in list(overrides):
        if key in _TUPLE_FIELDS and isinstance(overrides[key], list):
            overrides[key] = tuple(overrides[key])
    return cls(**overrides)


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce one run."""

    method: str
    seed: int
    group: str = "main"
    variant: str = ""
    env_id: str = DEFAULT_ENV
    total_steps: int = 100_000
    eval_every: int = 2_000
    eval_episodes: int = 10
    solved_threshold: float = 475.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Constructing the params validates them and fails fast on a typo.
        build_params(self.method, self.params)
        if self.total_steps <= 0 or self.eval_every <= 0:
            raise ConfigError("total_steps and eval_every must be positive")

    @property
    def typed_params(self):
        return build_params(self.method, self.params)

    @property
    def arm(self) -> str:
        """The name of this run's arm, used for grouping in the analysis."""
        return f"{self.method}:{self.variant}" if self.variant else self.method

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["params"] = _jsonable(dataclasses.asdict(self.typed_params))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown run config key(s): {sorted(unknown)}; "
                f"known keys are {sorted(known)}"
            )
        return cls(**data)

    @property
    def key(self) -> str:
        """Deterministic identifier derived from the config.

        This is what makes the runner resume-safe: the same config always maps to
        the same result file, so a completed run is recognised and skipped.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        stem = self.variant or "default"
        return f"{self.method}__{stem}__seed{self.seed}__{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def load_yaml(path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data
