"""Agent registry.

One factory keeps the training loop free of any per-method branching.
"""

from __future__ import annotations

from ..config import DQNParams, ReinforceParams, TabularParams
from .base import Agent
from .dqn import DQNAgent, ReplayBuffer
from .reinforce import ReinforceAgent
from .tabular import Discretizer, TabularQLearningAgent

__all__ = [
    "Agent",
    "DQNAgent",
    "Discretizer",
    "ReinforceAgent",
    "ReplayBuffer",
    "TabularQLearningAgent",
    "make_agent",
]


def make_agent(method: str, params, obs_dim: int, n_actions: int, seed: int) -> Agent:
    if method == "tabular":
        assert isinstance(params, TabularParams)
        return TabularQLearningAgent(params, n_actions, seed)
    if method == "dqn":
        assert isinstance(params, DQNParams)
        return DQNAgent(params, obs_dim, n_actions, seed)
    if method == "reinforce":
        assert isinstance(params, ReinforceParams)
        return ReinforceAgent(params, obs_dim, n_actions, seed)
    raise ValueError(f"unknown method {method!r}")
