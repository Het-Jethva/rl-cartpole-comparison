"""REINFORCE with a learned state-value baseline (Williams, 1992).

The baseline is a separate network trained by regression on the Monte-Carlo
returns; the advantage is the return minus the predicted value.

Returns are deliberately NOT whitened. Dividing by the batch standard deviation
rescales the gradient as well as centring it, so it is not a baseline in the
sense the theory describes, and it would confound the ablation.

`use_baseline=False` selects the ablation arm, where the advantage is the raw
Monte-Carlo return.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ReinforceParams
from ..utils import count_parameters, discounted_returns
from .base import Agent
from .dqn import mlp


class ReinforceAgent(Agent):
    def __init__(
        self, params: ReinforceParams, obs_dim: int, n_actions: int, seed: int
    ) -> None:
        self.params = params
        self.n_actions = n_actions
        self.generator = torch.Generator().manual_seed(seed)

        torch.manual_seed(seed)
        self.policy = mlp(obs_dim, params.hidden, n_actions)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=params.lr)

        self.value: nn.Module | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        if params.use_baseline:
            self.value = mlp(obs_dim, params.hidden, 1)
            self.value_optimizer = torch.optim.Adam(
                self.value.parameters(), lr=params.baseline_lr
            )

        self.update_count = 0
        self._episode_observations: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._policy_loss_sum = 0.0
        self._value_loss_sum = 0.0

    def act(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        if greedy:
            return int(torch.argmax(logits).item())
        probabilities = F.softmax(logits, dim=-1)
        return int(
            torch.multinomial(probabilities, num_samples=1, generator=self.generator).item()
        )

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        step: int,
    ) -> None:
        self._episode_observations.append(np.asarray(observation, dtype=np.float32))
        self._episode_actions.append(int(action))
        self._episode_rewards.append(float(reward))

    def end_episode(self) -> None:
        if not self._episode_rewards:
            return
        returns = discounted_returns(self._episode_rewards, self.params.gamma)
        self._pending.append(
            (
                np.asarray(self._episode_observations, dtype=np.float32),
                np.asarray(self._episode_actions, dtype=np.int64),
                returns.astype(np.float32),
            )
        )
        self._episode_observations.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()

        if len(self._pending) >= self.params.episodes_per_update:
            self._learn()

    def _learn(self) -> None:
        params = self.params
        observations = torch.from_numpy(np.concatenate([p[0] for p in self._pending]))
        actions = torch.from_numpy(np.concatenate([p[1] for p in self._pending]))
        returns = torch.from_numpy(np.concatenate([p[2] for p in self._pending]))
        self._pending.clear()

        if self.value is not None and self.value_optimizer is not None:
            with torch.no_grad():
                baseline = self.value(observations).squeeze(1)
            advantage = returns - baseline
        else:
            advantage = returns

        logits = self.policy(observations)
        log_probabilities = F.log_softmax(logits, dim=-1)
        chosen = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(chosen * advantage).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), params.grad_clip)
        self.policy_optimizer.step()

        if self.value is not None and self.value_optimizer is not None:
            predicted = self.value(observations).squeeze(1)
            value_loss = F.mse_loss(predicted, returns)
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), params.grad_clip)
            self.value_optimizer.step()
            self._value_loss_sum += float(value_loss.item())

        self.update_count += 1
        self._policy_loss_sum += float(policy_loss.item())

    def diagnostics(self) -> dict[str, Any]:
        parameters = count_parameters(self.policy)
        if self.value is not None:
            parameters += count_parameters(self.value)
        return {
            "parameters": parameters,
            "mean_policy_loss": self._policy_loss_sum / max(self.update_count, 1),
            "mean_value_loss": self._value_loss_sum / max(self.update_count, 1),
        }

    @property
    def hyperparameter_count(self) -> int:
        # hidden, lr, baseline_lr, gamma, episodes_per_update, grad_clip
        return 6
