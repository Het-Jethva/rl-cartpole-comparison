"""Small shared helpers.

`discounted_returns` lives here because all three methods need it and because a
mistake in it is silent: it produces a plausible learning curve rather than an
error. It is one of the tested seams.
"""

from __future__ import annotations

import random

import numpy as np


def discounted_returns(rewards: np.ndarray | list[float], gamma: float) -> np.ndarray:
    """Return G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k} for every t.

    G has the same length as `rewards`. G[-1] == rewards[-1] for any gamma.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.empty_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def exponential_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Exponential decay from `start` to `end`, reaching `end` at `decay_steps`.

    The value is held at `end` after `decay_steps`.
    """
    if step >= decay_steps or start <= 0.0:
        return end
    ratio = end / start
    return float(max(end, start * (ratio ** (step / decay_steps))))


def seed_everything(seed: int) -> None:
    """Seed the global generators that PyTorch and Python use."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
