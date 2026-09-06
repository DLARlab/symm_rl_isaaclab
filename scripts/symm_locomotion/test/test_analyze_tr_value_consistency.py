# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the offline complete-cycle TR value diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "analyze_tr_value_consistency.py"
    spec = importlib.util.spec_from_file_location("analyze_tr_value_consistency_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load_module()


def test_constant_reward_has_equal_future_and_past_return_at_every_phase():
    rewards = np.full(7, 2.5)

    future, past = analysis.complete_cycle_returns(rewards, gamma=0.83)
    report = analysis.analyze_complete_cycles([rewards], gamma=0.83, phase_bins=7)

    np.testing.assert_allclose(future, past)
    assert report["future_past_return_rmse"] == pytest.approx(0.0)
    assert report["future_past_return_normalized_rmse"] == pytest.approx(0.0)


def test_non_palindromic_discounted_cycle_generally_has_different_orientations():
    rewards = np.asarray([0.0, 1.0, 4.0, -2.0, 3.0])

    future, past = analysis.complete_cycle_returns(rewards, gamma=0.6)

    assert not np.allclose(future, past)
    assert np.sqrt(np.mean(np.square(future - past))) > 0.0


def test_undiscounted_full_cycle_sum_is_invariant_to_reversal():
    rewards = np.asarray([0.2, 5.0, -1.5, 3.25, 8.0])

    future, past = analysis.complete_cycle_returns(rewards, gamma=1.0, horizon=len(rewards))

    np.testing.assert_allclose(future, np.sum(rewards))
    np.testing.assert_allclose(past, np.sum(rewards))


def test_discounted_full_cycle_sequence_need_not_be_invariant_to_reversal():
    rewards = np.asarray([0.2, 5.0, -1.5, 3.25, 8.0])

    future, past = analysis.complete_cycle_returns(rewards, gamma=0.9, horizon=len(rewards))

    assert np.any(np.abs(future - past) > 1.0e-8)


def test_equal_scalar_returns_do_not_require_a_fixed_phase_state():
    # At phase 1/4, the immediate forward/backward neighbours happen to have
    # equal rewards, so the two four-step returns agree. The phase-reflection
    # map R_0(phi) = -phi mod 1 maps 1/4 to 3/4, not back to the same state.
    rewards = np.asarray([4.0, 1.0, 4.0, 7.0])
    future, past = analysis.complete_cycle_returns(rewards, gamma=0.5)
    phase = 0.25

    assert future[1] == pytest.approx(past[1])
    assert (-phase) % 1.0 != pytest.approx(phase)
    assert rewards[1] != rewards[3]


def test_matrix_cycles_wrap_independently_and_report_all_requested_strata(tmp_path):
    archive = tmp_path / "cycles.npz"
    np.savez_compressed(
        archive,
        step_rewards=np.asarray([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]]),
        phase=np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0]),
        gait_family=np.asarray(["trot", "pace"]),
        command_sign=np.asarray([1, -1]),
        robot=np.asarray(["go2", "x1"]),
        cycle_id=np.asarray([10, 20]),
        complete_cycle=np.asarray([True, True]),
    )

    cycles = analysis.load_complete_cycles(archive)
    report = analysis.analyze_complete_cycles(cycles, gamma=1.0, horizon=3, phase_bins=3)

    assert len(cycles) == 2
    assert report["cycle_summaries"][0]["future_return_mean"] == pytest.approx(6.0)
    assert report["cycle_summaries"][1]["future_return_mean"] == pytest.approx(600.0)
    assert set(report["future_past_return_gap_by_gait_family"]) == {"pace", "trot"}
    assert set(report["future_past_return_gap_by_command_sign"]) == {"negative", "positive"}
    assert set(report["future_past_return_gap_by_robot"]) == {"go2", "x1"}
    assert set(report["future_past_return_gap_by_phase"]) == {"0", "1", "2"}
    assert report["phase_source_counts"] == {"recorded": 2, "inferred_uniform_index": 0}


def test_optional_critic_diagnostics_compare_against_the_correct_return_orientation():
    rewards = np.asarray([1.0, 4.0, -2.0, 3.0])
    future, past = analysis.complete_cycle_returns(rewards, gamma=0.75)
    report = analysis.analyze_complete_cycles(
        [
            analysis.CompleteCycle(
                rewards=rewards,
                critic_values=future + 2.0,
                transformed_critic_values=past - 3.0,
            )
        ],
        gamma=0.75,
    )

    assert report["critic_value_minus_future_return"]["mean"] == pytest.approx(2.0)
    assert report["transformed_critic_value_minus_past_return"]["mean"] == pytest.approx(-3.0)
    assert report["transformed_minus_original_critic_value"]["mean"] == pytest.approx(-5.0)
    assert report["transformed_minus_original_critic_value"]["rmse"] > 0.0
    assert report["phase_source_counts"] == {"recorded": 0, "inferred_uniform_index": 1}
    assert report["cycle_summaries"][0]["phase_source"] == "inferred_uniform_index"


@pytest.mark.parametrize("boundary_field", ["episode_id", "command_interval_id", "gait_row"])
def test_vector_cycles_never_wrap_across_recorded_boundaries(tmp_path, boundary_field):
    archive = tmp_path / f"{boundary_field}.npz"
    np.savez_compressed(
        archive,
        step_rewards=np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
        complete_cycle=np.ones(6, dtype=bool),
        **{boundary_field: np.asarray([0, 0, 0, 1, 1, 1])},
    )

    cycles = analysis.load_complete_cycles(archive)
    report = analysis.analyze_complete_cycles(cycles, gamma=1.0, horizon=3)

    assert len(cycles) == 2
    assert report["cycle_summaries"][0]["future_return_mean"] == pytest.approx(6.0)
    assert report["cycle_summaries"][1]["future_return_mean"] == pytest.approx(60.0)


def test_aggregate_only_archive_fails_without_fabricating_per_step_rewards(tmp_path):
    archive = tmp_path / "aggregate_only.npz"
    np.savez_compressed(archive, mean_reward=np.asarray([12.0, 13.0]), episode_return=np.asarray([120.0, 130.0]))

    with pytest.raises(ValueError, match="sufficient per-step reward information"):
        analysis.load_complete_cycles(archive)


def test_generic_reward_alias_is_not_assumed_to_be_per_step_data(tmp_path):
    archive = tmp_path / "ambiguous_rewards.npz"
    np.savez_compressed(
        archive,
        rewards=np.asarray([120.0, 130.0]),
        cycle_id=np.asarray([0, 0]),
        complete_cycle=np.asarray([True, True]),
    )

    with pytest.raises(ValueError, match="sufficient per-step reward information"):
        analysis.load_complete_cycles(archive)


def test_vector_archive_requires_explicit_boundary_and_completeness_evidence(tmp_path):
    no_boundary = tmp_path / "no_boundary.npz"
    np.savez_compressed(
        no_boundary,
        step_rewards=np.asarray([1.0, 2.0, 3.0]),
        complete_cycle=np.asarray([True, True, True]),
    )
    with pytest.raises(ValueError, match="require cycle_id, episode_id, command_interval_id, or gait_row"):
        analysis.load_complete_cycles(no_boundary)

    no_completeness = tmp_path / "no_completeness.npz"
    np.savez_compressed(
        no_completeness,
        step_rewards=np.asarray([1.0, 2.0, 3.0]),
        cycle_id=np.asarray([0, 0, 0]),
    )
    with pytest.raises(ValueError, match="must include a true complete_cycle marker"):
        analysis.load_complete_cycles(no_completeness)


@pytest.mark.parametrize(
    "marker",
    [
        np.asarray([True, True, False]),
        np.asarray(["false", "false", "false"]),
        np.asarray([1.0, 1.0, np.nan]),
    ],
)
def test_incomplete_or_malformed_cycle_markers_are_rejected(tmp_path, marker):
    archive = tmp_path / "invalid_complete_marker.npz"
    np.savez_compressed(
        archive,
        step_rewards=np.asarray([1.0, 2.0, 3.0]),
        cycle_id=np.asarray([0, 0, 0]),
        complete_cycle=marker,
    )

    with pytest.raises(ValueError, match="complete-cycle marker|marks .* as incomplete"):
        analysis.load_complete_cycles(archive)


def test_requested_horizon_must_fit_every_recorded_cycle():
    with pytest.raises(ValueError, match="horizon .* recorded cycle length"):
        analysis.analyze_complete_cycles([np.asarray([1.0, 2.0, 3.0])], gamma=0.9, horizon=4)
