"""Checks against silent numerical error.

These tests are not a refactoring safety net. They exist because the failure mode
of this project is quiet: a sign error in the advantage or an off-by-one in the
discounted return does not raise, it produces a plausible learning curve, and
every number in the paper is then wrong with no warning.

Each test targets a place where a wrong value would not raise. Nothing here
asserts on how a result was computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlcompare.agents import Discretizer, ReplayBuffer
from rlcompare.agents.tabular import CART_POSITION_LIMIT, POLE_ANGLE_LIMIT
from rlcompare.analysis.stats import bootstrap_ci, iqm
from rlcompare.config import ConfigError, RunConfig, build_params
from rlcompare.experiments.runner import run_experiment
from rlcompare.utils import discounted_returns, exponential_epsilon


# --------------------------------------------------------------------------
# discounted returns
# --------------------------------------------------------------------------


def test_discounted_returns_matches_hand_computed_example():
    # rewards 1, 2, 3 with gamma 0.5:
    #   G2 = 3
    #   G1 = 2 + 0.5*3   = 3.5
    #   G0 = 1 + 0.5*3.5 = 2.75
    result = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    assert np.allclose(result, [2.75, 3.5, 3.0])


def test_discounted_returns_with_gamma_one_is_the_reverse_cumulative_sum():
    rewards = [1.0] * 5
    assert np.allclose(discounted_returns(rewards, gamma=1.0), [5, 4, 3, 2, 1])


def test_discounted_returns_last_element_equals_last_reward():
    rewards = np.array([0.3, -1.0, 7.0, 2.5])
    for gamma in (0.0, 0.5, 0.99, 1.0):
        assert discounted_returns(rewards, gamma)[-1] == pytest.approx(rewards[-1])


def test_discounted_returns_length_matches_input():
    for n in (1, 2, 17):
        assert len(discounted_returns(np.ones(n), 0.99)) == n


# --------------------------------------------------------------------------
# discretizer
# --------------------------------------------------------------------------


def test_discretizer_maps_clip_limits_to_first_and_last_bin():
    bins = (6, 6, 12, 12)
    discretizer = Discretizer(bins, clip_velocity=(3.0, 3.5))

    at_low = discretizer.index(np.array([-CART_POSITION_LIMIT, -3.0, -POLE_ANGLE_LIMIT, -3.5]))
    at_high = discretizer.index(np.array([CART_POSITION_LIMIT, 3.0, POLE_ANGLE_LIMIT, 3.5]))

    assert at_low == (0, 0, 0, 0)
    assert at_high == tuple(b - 1 for b in bins)


def test_discretizer_clips_extreme_observations_into_range():
    bins = (6, 6, 12, 12)
    discretizer = Discretizer(bins, clip_velocity=(3.0, 3.5))
    extreme = discretizer.index(np.array([1e6, -1e6, 1e6, -1e6]))
    for index, size in zip(extreme, bins):
        assert 0 <= index < size


def test_discretizer_separates_states_within_range():
    discretizer = Discretizer((6, 6, 12, 12), clip_velocity=(3.0, 3.5))
    left = discretizer.index(np.array([-2.0, 0.0, 0.0, 0.0]))
    right = discretizer.index(np.array([2.0, 0.0, 0.0, 0.0]))
    assert left != right


def test_discretizer_cell_count_is_the_product_of_the_bins():
    assert Discretizer((3, 3, 6, 6), (3.0, 3.5)).n_cells == 3 * 3 * 6 * 6


# --------------------------------------------------------------------------
# replay buffer
# --------------------------------------------------------------------------


def test_replay_buffer_sample_has_expected_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=100, obs_dim=4, seed=0)
    for i in range(50):
        buffer.add(np.full(4, i, dtype=np.float32), i % 2, float(i), np.zeros(4), i % 7 == 0)

    batch = buffer.sample(16)
    assert batch["observations"].shape == (16, 4)
    assert batch["observations"].dtype == np.float32
    assert batch["actions"].shape == (16,)
    assert batch["actions"].dtype == np.int64
    assert batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == np.float32


def test_replay_buffer_overwrites_oldest_transitions_when_full():
    buffer = ReplayBuffer(capacity=10, obs_dim=1, seed=0)
    for i in range(25):
        buffer.add(np.array([i], dtype=np.float32), 0, 0.0, np.zeros(1), False)

    assert len(buffer) == 10
    stored = set(buffer.observations[:, 0].tolist())
    assert stored == set(float(i) for i in range(15, 25))


def test_replay_buffer_rejects_sampling_when_empty():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, obs_dim=4, seed=0).sample(4)


# --------------------------------------------------------------------------
# aggregation statistics
# --------------------------------------------------------------------------


def test_iqm_discards_the_outer_quartiles():
    # The middle 50% of 1..8 is 3,4,5,6 -> mean 4.5, while the mean of a set with
    # one catastrophic outlier would be dragged far below it.
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)
    assert iqm([500, 500, 500, 500, 0, 0]) == pytest.approx(500.0)


def test_iqm_falls_back_to_the_mean_for_small_samples():
    assert iqm([1.0, 5.0]) == pytest.approx(3.0)


def test_bootstrap_interval_is_deterministic_and_brackets_the_estimate():
    values = [420.0, 500.0, 500.0, 480.0, 350.0, 500.0, 500.0, 460.0]
    first = bootstrap_ci(values, n_resamples=500)
    second = bootstrap_ci(values, n_resamples=500)
    assert first == second
    assert first[0] <= iqm(values) <= first[1]


# --------------------------------------------------------------------------
# epsilon schedule
# --------------------------------------------------------------------------


def test_epsilon_starts_at_start_and_settles_at_end():
    assert exponential_epsilon(0, 1.0, 0.05, 1000) == pytest.approx(1.0)
    assert exponential_epsilon(1000, 1.0, 0.05, 1000) == pytest.approx(0.05)
    assert exponential_epsilon(9999, 1.0, 0.05, 1000) == pytest.approx(0.05)
    middle = exponential_epsilon(500, 1.0, 0.05, 1000)
    assert 0.05 < middle < 1.0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_unknown_hyperparameter_is_rejected():
    with pytest.raises(ConfigError):
        build_params("dqn", {"learnign_rate": 0.001})


def test_unknown_method_is_rejected():
    with pytest.raises(ConfigError):
        build_params("sarsa", {})


def test_config_key_changes_with_any_behavioural_difference():
    a = RunConfig(method="dqn", seed=0, total_steps=1000)
    b = RunConfig(method="dqn", seed=1, total_steps=1000)
    c = RunConfig(method="dqn", seed=0, total_steps=1000, params={"lr": 0.01})
    assert len({a.key, b.key, c.key}) == 3


# --------------------------------------------------------------------------
# the primary seam: run_experiment
# --------------------------------------------------------------------------

SHORT = dict(total_steps=600, eval_every=300, eval_episodes=2)


@pytest.mark.parametrize("method", ["tabular", "dqn", "reinforce"])
def test_short_run_produces_a_well_formed_result(method):
    config = RunConfig(method=method, seed=0, group="test", **SHORT)
    result = run_experiment(config)

    # 600 steps with an evaluation every 300, plus the final one at the budget.
    assert result.eval_steps == [0, 300, 600]
    assert len(result.eval_mean) == 3
    assert all(len(r) == 2 for r in result.eval_returns)
    assert all(1.0 <= value <= 500.0 for value in result.eval_mean)
    assert result.wall_clock_seconds > 0.0
    assert result.config["method"] == method
    assert result.diagnostics["hyperparameter_count"] > 0


@pytest.mark.parametrize("method", ["tabular", "dqn", "reinforce"])
def test_same_seed_reproduces_the_same_curve(method):
    config = RunConfig(method=method, seed=3, group="test", **SHORT)
    assert run_experiment(config).eval_mean == run_experiment(config).eval_mean


@pytest.mark.parametrize("method", ["tabular", "dqn", "reinforce"])
def test_different_seeds_produce_different_runs(method):
    # Guards against a seed that is accepted but never applied - the determinism
    # test above would pass happily in that case.
    a = run_experiment(RunConfig(method=method, seed=1, group="test", **SHORT))
    b = run_experiment(RunConfig(method=method, seed=2, group="test", **SHORT))
    assert a.train_episode_returns != b.train_episode_returns


def test_tabular_run_reports_state_coverage_below_one():
    config = RunConfig(method="tabular", seed=0, group="test", **SHORT)
    result = run_experiment(config)
    coverage = result.diagnostics["state_coverage"]
    assert 0.0 < coverage < 1.0
