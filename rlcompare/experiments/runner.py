"""The single training loop, shared by all three methods.

`run_experiment` is the one entry point into the system: config in, result out.
Writing the loop once is what makes the comparison fair - no method gets a
different amount of experience, a different evaluation procedure, or a different
definition of "solved".

Two details in the loop matter more than they look:

* The loop bootstraps through a time-limit truncation but not through a genuine
  termination. CartPole truncates at 500 steps; treating that as a terminal state
  teaches every value-based agent that balancing forever is worth zero.
* Evaluation uses a fixed set of episode seeds, identical for every run and every
  method. The evaluation set is therefore a held-out test set rather than fresh
  noise at each measurement.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from ..agents import make_agent
from ..config import RunConfig
from ..utils import seed_everything
from .results import RunResult

# A fixed evaluation set. Identical start states for every method, every seed and
# every evaluation point, so differences in the curve come from the policy alone.
EVALUATION_SEEDS = tuple(90_000 + i for i in range(100))


def evaluate(env: gym.Env, agent, n_episodes: int) -> list[float]:
    """Run `n_episodes` greedy episodes and return their undiscounted returns."""
    returns: list[float] = []
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=EVALUATION_SEEDS[episode % len(EVALUATION_SEEDS)])
        total = 0.0
        while True:
            action = agent.act(observation, greedy=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return returns


def run_experiment(config: RunConfig) -> RunResult:
    """Train one agent under one config and return its complete result."""
    import torch

    # One thread per run: the matrix runner already uses every core, and letting
    # each worker spawn its own thread pool makes the wall-clock numbers noise.
    torch.set_num_threads(1)
    seed_everything(config.seed)

    train_env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    obs_dim = int(np.prod(train_env.observation_space.shape))
    n_actions = int(train_env.action_space.n)

    agent = make_agent(config.method, config.typed_params, obs_dim, n_actions, config.seed)

    result = RunResult(
        config=config.to_dict(),
        key=config.key,
        group=config.group,
        arm=config.arm,
        method=config.method,
        seed=config.seed,
    )

    started = time.perf_counter()
    observation, _ = train_env.reset(seed=config.seed)
    episode_return = 0.0

    for step in range(config.total_steps):
        if step % config.eval_every == 0:
            returns = evaluate(eval_env, agent, config.eval_episodes)
            _record_evaluation(result, config, step, returns)

        action = agent.act(observation)
        next_observation, reward, terminated, truncated, _ = train_env.step(action)
        episode_return += float(reward)
        agent.observe(observation, action, float(reward), next_observation, terminated, step)

        if terminated or truncated:
            agent.end_episode()
            result.train_episode_returns.append(episode_return)
            result.train_episode_end_steps.append(step + 1)
            episode_return = 0.0
            observation, _ = train_env.reset()
        else:
            observation = next_observation

    # A final measurement at the full budget, so every curve ends at total_steps.
    returns = evaluate(eval_env, agent, config.eval_episodes)
    _record_evaluation(result, config, config.total_steps, returns)

    result.wall_clock_seconds = time.perf_counter() - started
    result.update_count = int(agent.update_count)
    result.diagnostics = agent.diagnostics()
    result.diagnostics["hyperparameter_count"] = agent.hyperparameter_count
    result.final_mean = result.eval_mean[-1]
    result.area_under_curve = float(np.trapezoid(result.eval_mean, result.eval_steps))

    train_env.close()
    eval_env.close()
    return result


def _record_evaluation(
    result: RunResult, config: RunConfig, step: int, returns: list[float]
) -> None:
    mean = float(np.mean(returns))
    result.eval_steps.append(step)
    result.eval_returns.append(returns)
    result.eval_mean.append(mean)
    if result.steps_to_solve is None and mean >= config.solved_threshold:
        result.steps_to_solve = step
