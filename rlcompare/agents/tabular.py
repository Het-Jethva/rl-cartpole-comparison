"""Tabular Q-learning over a discretised CartPole state space.

CartPole has four continuous dimensions, two of which (cart velocity and pole
angular velocity) are unbounded. They are clipped to hand-set limits before
binning; without clipping there is no finite table.

The agent also records state coverage - the fraction of table cells ever visited.
That number is the direct evidence for the function-approximation argument in the
paper, so it is a first-class output rather than a debug print.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import TabularParams
from ..utils import exponential_epsilon
from .base import Agent

# CartPole terminates outside these bounds, so they are the natural limits for
# the two bounded dimensions.
CART_POSITION_LIMIT = 2.4
POLE_ANGLE_LIMIT = 0.2095  # 12 degrees, in radians


class Discretizer:
    """Maps a continuous CartPole observation to a tuple of bin indices."""

    def __init__(
        self,
        bins: tuple[int, int, int, int],
        clip_velocity: tuple[float, float],
    ) -> None:
        if len(bins) != 4 or any(b < 1 for b in bins):
            raise ValueError("bins must be four positive integers")
        self.bins = tuple(int(b) for b in bins)
        cart_velocity_limit, angular_velocity_limit = clip_velocity
        self.low = np.array(
            [
                -CART_POSITION_LIMIT,
                -float(cart_velocity_limit),
                -POLE_ANGLE_LIMIT,
                -float(angular_velocity_limit),
            ],
            dtype=np.float64,
        )
        self.high = -self.low
        # Interior edges only: np.digitize then returns an index in [0, bins-1].
        self._edges = [
            np.linspace(self.low[d], self.high[d], self.bins[d] + 1)[1:-1]
            for d in range(4)
        ]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.bins))

    def index(self, observation: np.ndarray) -> tuple[int, int, int, int]:
        """Return the bin index per dimension.

        Observations outside the limits are clipped, so the result is always a
        valid index. This is the behaviour the paper describes as "clipping".
        """
        clipped = np.clip(np.asarray(observation, dtype=np.float64), self.low, self.high)
        return tuple(
            int(np.digitize(clipped[d], self._edges[d])) for d in range(4)
        )  # type: ignore[return-value]


class TabularQLearningAgent(Agent):
    """Q-learning with an epsilon-greedy behaviour policy."""

    def __init__(self, params: TabularParams, n_actions: int, seed: int) -> None:
        self.params = params
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.discretizer = Discretizer(params.bins, params.clip_velocity)
        self.q = np.zeros((*self.discretizer.bins, n_actions), dtype=np.float64)
        self._visited = np.zeros(self.discretizer.bins, dtype=bool)
        self.update_count = 0
        self._epsilon = params.eps_start

    def act(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        state = self.discretizer.index(observation)
        if not greedy and self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[state]
        # Random tie-break: with an all-zero table, argmax would always pick
        # action 0 and the greedy policy would never move the cart.
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        step: int,
    ) -> None:
        params = self.params
        self._epsilon = exponential_epsilon(
            step, params.eps_start, params.eps_end, params.eps_decay_steps
        )
        state = self.discretizer.index(observation)
        next_state = self.discretizer.index(next_observation)
        self._visited[state] = True

        bootstrap = 0.0 if terminated else float(self.q[next_state].max())
        target = reward + params.gamma * bootstrap
        self.q[(*state, action)] += params.alpha * (target - self.q[(*state, action)])
        self.update_count += 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "table_cells": self.discretizer.n_cells,
            "table_entries": self.discretizer.n_cells * self.n_actions,
            "state_coverage": float(self._visited.mean()),
            "epsilon_final": float(self._epsilon),
        }

    @property
    def hyperparameter_count(self) -> int:
        # bins, clip_velocity, alpha, gamma, eps_start, eps_end, eps_decay_steps
        return 7
