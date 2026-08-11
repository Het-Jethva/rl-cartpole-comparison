"""Canonical DQN (Mnih et al., 2015): replay buffer, target network, Huber loss.

The only departure from the original is scale: the network is a small MLP over a
four-dimensional observation rather than a convolutional stack over pixels.

`use_target_network=False` selects the ablation arm, where the bootstrap target
is computed with the online network.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DQNParams
from ..utils import count_parameters, exponential_epsilon
from .base import Agent


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions.

    Stored as preallocated NumPy arrays with explicit dtypes. An implicit dtype
    here would be silently cast by PyTorch rather than raising.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._cursor
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.terminated[i] = float(terminated)
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample uniformly with replacement from the stored transitions."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "terminated": self.terminated[idx],
        }


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden:
        layers.append(nn.Linear(last, width))
        layers.append(nn.ReLU())
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class DQNAgent(Agent):
    def __init__(
        self, params: DQNParams, obs_dim: int, n_actions: int, seed: int
    ) -> None:
        self.params = params
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator().manual_seed(seed)

        torch.manual_seed(seed)
        self.online = mlp(obs_dim, params.hidden, n_actions)
        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=params.lr)
        self.buffer = ReplayBuffer(params.buffer_size, obs_dim, seed + 1)
        self.update_count = 0
        self._epsilon = params.eps_start
        self._loss_sum = 0.0

    def act(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self._epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            values = self.online(torch.as_tensor(observation, dtype=torch.float32))
        return int(torch.argmax(values).item())

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
        self.buffer.add(observation, action, reward, next_observation, terminated)

        if len(self.buffer) < params.learning_starts:
            return
        if step % params.train_frequency != 0:
            return
        self._learn()

        if params.use_target_network and self.update_count % params.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _learn(self) -> None:
        params = self.params
        batch = self.buffer.sample(params.batch_size)
        observations = torch.from_numpy(batch["observations"])
        actions = torch.from_numpy(batch["actions"]).unsqueeze(1)
        rewards = torch.from_numpy(batch["rewards"])
        next_observations = torch.from_numpy(batch["next_observations"])
        terminated = torch.from_numpy(batch["terminated"])

        predicted = self.online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            network = self.target if params.use_target_network else self.online
            bootstrap = network(next_observations).max(dim=1).values
            target = rewards + params.gamma * (1.0 - terminated) * bootstrap

        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), params.grad_clip)
        self.optimizer.step()

        self.update_count += 1
        self._loss_sum += float(loss.item())

    def diagnostics(self) -> dict[str, Any]:
        return {
            "parameters": count_parameters(self.online),
            "mean_loss": self._loss_sum / max(self.update_count, 1),
            "epsilon_final": float(self._epsilon),
            "buffer_size": len(self.buffer),
        }

    @property
    def hyperparameter_count(self) -> int:
        # hidden, lr, gamma, buffer_size, batch_size, learning_starts,
        # train_frequency, target_update_interval, grad_clip,
        # eps_start, eps_end, eps_decay_steps
        return 12
