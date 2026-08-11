"""The shared agent interface.

All three methods expose the same four operations. The training loop is written
once against this interface, so the loop cannot favour one method over another.
Every difference between the methods lives in `observe` and `end_episode`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """A reinforcement learning agent for a discrete action space."""

    #: Number of value or gradient updates performed so far.
    update_count: int = 0

    @abstractmethod
    def act(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        """Choose an action. `greedy=True` disables exploration (evaluation)."""

    @abstractmethod
    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        step: int,
    ) -> None:
        """Accept one transition.

        `terminated` is the true end of the Markov chain, never a time-limit
        truncation. Bootstrapping through a truncation is correct; bootstrapping
        through a termination is not.
        """

    def end_episode(self) -> None:
        """Called after every episode, whether terminated or truncated."""

    def diagnostics(self) -> dict[str, Any]:
        """Method-specific numbers worth reporting in the paper."""
        return {}

    @property
    def hyperparameter_count(self) -> int:
        """Number of tunable hyperparameters, for the implementation-cost column."""
        return 0
