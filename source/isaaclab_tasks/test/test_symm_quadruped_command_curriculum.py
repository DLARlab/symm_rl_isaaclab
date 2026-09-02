# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the symmetric-quadruped TR-orbit command curriculum."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped import env as symm_quadruped_env
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.tr_orbit_curriculum import (
    TimeReversalOrbitCurriculum,
)


def _make_curriculum(**overrides) -> TimeReversalOrbitCurriculum:
    kwargs = {
        "num_envs": 4,
        "gait_partner_indices": SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS,
        "velocity_range": (-2.0, 2.0),
        "velocity_bin_count": 5,
        "ewma_coefficient": 1.0,
        "unlock_threshold": 0.5,
        "current_cell_increment": 1.0,
        "neighbor_increment": 0.25,
        "exploration_floor": 0.05,
        "maximum_weight": 3.0,
        "seed": 42,
        "initial_max_abs_speed": 0.5,
        "minimum_visits": 2,
        "locked_cell_weight": 0.0,
    }
    kwargs.update(overrides)
    return TimeReversalOrbitCurriculum(**kwargs)


def _make_command_term(num_envs: int = 4) -> symm_quadruped.GaitVelocityCommand:
    """Construct a CPU command term with its real curriculum boundary methods."""
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(-3.14, 3.14),
    )
    cfg.heading_command = False
    cfg.rel_standing_envs = 0.0
    cfg.resampling_time_range = (10.0, 10.0)
    cfg.gait_sequence_enabled = False
    cfg.calculate_from_sampling_curve = False
    cfg.add_noise_theta = False
    cfg.noise_level_theta = 0

    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        common_step_counter=0,
        step_dt=0.02,
        reset_buf=torch.zeros(num_envs, dtype=torch.bool),
        reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
        extras={},
    )
    command_term.vel_command_b = torch.zeros((num_envs, 3), dtype=torch.float32)
    command_term.heading_target = torch.zeros(num_envs, dtype=torch.float32)
    command_term.is_heading_env = torch.zeros(num_envs, dtype=torch.bool)
    command_term.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)
    command_term.time_left = torch.zeros(num_envs, dtype=torch.float32)
    command_term.command_counter = torch.zeros(num_envs, dtype=torch.long)
    command_term.init_foot_thetas = torch.tensor(
        symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS,
        dtype=torch.float32,
    )
    command_term.foot_theta_sampling_weights = None
    command_term.foot_thetas = torch.zeros((num_envs, 4), dtype=torch.float32)
    command_term.duty_factors = torch.full((num_envs,), cfg.duty_factor, dtype=torch.float32)
    command_term.gait_periods = torch.full((num_envs,), cfg.gait_period, dtype=torch.float32)
    command_term.kappa = torch.full((num_envs,), cfg.kappa, dtype=torch.float32)
    command_term.gait_time_left = torch.zeros(num_envs, dtype=torch.float32)
    command_term.gait_counter = torch.zeros(num_envs, dtype=torch.long)
    command_term.gait_row_indices = torch.full((num_envs,), -1, dtype=torch.long)
    command_term.gait_sequence_indices = torch.full((num_envs,), -1, dtype=torch.long)
    command_term._common_gait_phase_at_anchor = torch.zeros(num_envs, dtype=torch.float32)
    command_term._gait_phase_anchor_steps = torch.zeros(num_envs, dtype=torch.long)
    command_term._error_xy_sum = torch.zeros(num_envs, dtype=torch.float32)
    command_term._error_yaw_sum = torch.zeros(num_envs, dtype=torch.float32)
    command_term._step_count = torch.zeros(num_envs, dtype=torch.float32)
    command_term._evaluation_active = torch.zeros(num_envs, dtype=torch.bool)
    command_term._evaluation_forward_velocity = torch.zeros(num_envs, dtype=torch.float32)
    command_term._evaluation_gait_indices = torch.full((num_envs,), -1, dtype=torch.long)
    command_term._evaluation_deterministic_timing = torch.zeros(num_envs, dtype=torch.bool)
    command_term._curriculum_joint_resample_pending = torch.zeros(num_envs, dtype=torch.bool)
    command_term._curriculum_skip_next_gait_resample = torch.zeros(num_envs, dtype=torch.bool)
    command_term._training_iteration = 0
    command_term.command_curriculum = _make_curriculum(num_envs=num_envs, seed=91)
    command_term.metrics = {
        name: torch.zeros(num_envs, dtype=torch.float32)
        for name in (
            "error_vel_xy",
            "error_vel_yaw",
            "success_rate",
            "success_threshold_vel_xy",
            "success_threshold_vel_yaw",
            "gait_period",
            "duty_factor",
            "command_curriculum_segment_success",
            "command_curriculum_unlocked_fraction",
        )
    }
    return command_term


def _assert_nested_state_equal(actual, expected, path: str = "state") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for name in expected:
            _assert_nested_state_equal(actual[name], expected[name], f"{path}.{name}")
    elif isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(actual, expected), path
    else:
        assert actual == expected, path


def _assert_orbit_equal(curriculum: TimeReversalOrbitCurriculum, value: torch.Tensor) -> None:
    partner_gait = curriculum.gait_partner_indices[:, None].expand_as(value)
    partner_velocity = curriculum.velocity_partner_indices[None, :].expand_as(value)
    assert torch.equal(value, value[partner_gait, partner_velocity])


def test_partner_maps_are_involutions_and_velocity_bins_are_signed_symmetric():
    curriculum = _make_curriculum()

    gait_rows = torch.arange(curriculum.num_gaits)
    velocity_bins = torch.arange(curriculum.velocity_bin_count)
    assert torch.equal(curriculum.gait_partner_indices[curriculum.gait_partner_indices], gait_rows)
    assert torch.equal(curriculum.velocity_partner_indices[curriculum.velocity_partner_indices], velocity_bins)
    assert torch.equal(
        curriculum.velocity_bin_centers[curriculum.velocity_partner_indices],
        -curriculum.velocity_bin_centers,
    )
    assert curriculum.velocity_bin_indices(torch.tensor([-2.0, -0.8, 0.0, 0.8, 2.0])).tolist() == [0, 1, 2, 3, 4]


def test_initial_support_is_speed_limited_orbit_closed_and_excludes_zero_prior_rows():
    gait_prior = torch.ones(10)
    gait_prior[2:4] = 0.0
    curriculum = _make_curriculum(initial_gait_weights=gait_prior)

    expected_speed_support = torch.tensor([False, False, True, False, False])
    assert torch.equal(curriculum.eligible[0], expected_speed_support)
    assert not torch.any(curriculum.eligible[2:4])
    assert not torch.any(curriculum.weights[2:4])
    assert not torch.any(curriculum.weights[:, ~expected_speed_support])
    for value in (
        curriculum.priority,
        curriculum.eligible,
        curriculum.mastered,
        curriculum.ewma_success,
        curriculum.visit_counts,
        curriculum.weights,
    ):
        _assert_orbit_equal(curriculum, value)

    sampled_gaits, sampled_bins = curriculum.sample_joint(512)
    assert not torch.any((sampled_gaits == 2) | (sampled_gaits == 3))
    assert torch.equal(sampled_bins, torch.full_like(sampled_bins, 2))


def test_first_mastery_unlocks_only_the_next_larger_speed_orbits_once():
    curriculum = _make_curriculum(
        current_cell_increment=0.5,
        neighbor_increment=0.25,
        exploration_floor=0.1,
        maximum_weight=10.0,
    )

    # Rows 2 and 3 are TR partners. At zero speed they share one canonical orbit.
    curriculum.update_cells(torch.tensor([2]), torch.tensor([2]), torch.tensor([True]))
    assert not curriculum.mastered[2, 2]
    assert not curriculum.eligible[2, 1]
    curriculum.update_cells(torch.tensor([3]), torch.tensor([2]), torch.tensor([True]))

    assert curriculum.mastered[2, 2]
    assert curriculum.mastered[3, 2]
    assert curriculum.eligible[2, 1] and curriculum.eligible[3, 3]
    assert curriculum.eligible[2, 3] and curriculum.eligible[3, 1]
    assert not curriculum.eligible[2, 0] and not curriculum.eligible[2, 4]
    assert curriculum.priority[2, 2].item() == 1.5
    assert curriculum.priority[2, 1].item() == 1.25

    eligibility_after_first_mastery = curriculum.eligible.clone()
    newly_unlocked_priority = curriculum.priority[2, 1].item()
    curriculum.update_cells(torch.tensor([2]), torch.tensor([2]), torch.tensor([True]))

    assert torch.equal(curriculum.eligible, eligibility_after_first_mastery)
    assert curriculum.priority[2, 1].item() == newly_unlocked_priority

    # Mastering the positive next ring advances only its outward TR orbit.
    curriculum.update_cells(
        torch.tensor([2, 3]),
        torch.tensor([3, 1]),
        torch.tensor([True, True]),
    )
    assert curriculum.eligible[2, 4]
    assert curriculum.eligible[3, 0]
    assert not curriculum.eligible[2, 0]


def test_conditional_sampling_uses_the_selected_weight_row_or_column():
    curriculum = _make_curriculum(
        exploration_floor=1.0e-12,
        maximum_weight=100.0,
        initial_max_abs_speed=2.0,
    )
    curriculum.priority.fill_(1.0e-12)
    # Preserve orbit equality while making (g=2, k=4) and its partner dominant.
    curriculum.priority[2, 4] = 100.0
    curriculum.priority[3, 0] = 100.0

    velocity_choices = curriculum.sample_velocity_given_gait(torch.full((128,), 2))
    gait_choices = curriculum.sample_gait_given_velocity(torch.full((128,), 4))

    assert torch.equal(velocity_choices, torch.full((128,), 4))
    assert torch.equal(gait_choices, torch.full((128,), 2))


def test_iteration_dependent_gait_prior_preserves_learned_within_row_preferences():
    initial_prior = torch.tensor((1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 5.0, 6.0))
    curriculum = _make_curriculum(
        initial_gait_weights=initial_prior,
        exploration_floor=1.0e-8,
        maximum_weight=100.0,
        initial_max_abs_speed=2.0,
    )
    curriculum.update_cells(
        torch.tensor([2, 3]),
        torch.tensor([0, 4]),
        torch.tensor([True, True]),
    )
    learned_preferences = curriculum.priority / curriculum.gait_prior_weights[:, None]
    updated_prior = torch.tensor((2.0, 4.0, 1.5, 1.5, 6.0, 6.0, 7.0, 8.0, 7.0, 8.0))

    curriculum.update_gait_prior_weights(updated_prior)

    assert torch.equal(curriculum.gait_prior_weights, updated_prior.to(torch.float64))
    assert torch.allclose(
        curriculum.priority / curriculum.gait_prior_weights[:, None],
        learned_preferences,
        atol=1.0e-12,
        rtol=0.0,
    )
    _assert_orbit_equal(curriculum, curriculum.priority)


def test_gait_prior_updates_do_not_apply_exploration_floor_to_zero_rows():
    curriculum = _make_curriculum(initial_max_abs_speed=2.0)
    updated_prior = torch.ones(10)
    updated_prior[2:4] = 0.0

    curriculum.update_gait_prior_weights(updated_prior)

    assert not torch.any(curriculum.eligible[2:4])
    assert not torch.any(curriculum.mastered[2:4])
    assert not torch.any(curriculum.priority[2:4])
    assert not torch.any(curriculum.weights[2:4])

    updated_prior[2:4] = 0.5
    curriculum.update_gait_prior_weights(updated_prior)
    assert torch.equal(curriculum.eligible[2], torch.ones(5, dtype=torch.bool))
    assert torch.equal(curriculum.priority[2], torch.full((5,), 0.5, dtype=torch.float64))


def test_orbit_batch_aggregation_is_permutation_and_duplication_invariant():
    kwargs = {
        "ewma_coefficient": 0.25,
        "unlock_threshold": 0.1,
        "minimum_visits": 1,
        "current_cell_increment": 0.5,
        "neighbor_increment": 0.25,
    }
    gait = torch.tensor([2, 3, 2, 0])
    velocity_bin = torch.tensor([2, 2, 2, 2])
    success = torch.tensor([True, False, True, False])

    original = _make_curriculum(**kwargs)
    permuted = _make_curriculum(**kwargs)
    duplicated = _make_curriculum(**kwargs)
    original.update_cells(gait, velocity_bin, success)
    permutation = torch.tensor([3, 1, 0, 2])
    permuted.update_cells(gait[permutation], velocity_bin[permutation], success[permutation])
    duplicated.update_cells(
        torch.cat((gait, gait)),
        torch.cat((velocity_bin, velocity_bin)),
        torch.cat((success, success)),
    )

    for name in ("ewma_success", "eligible", "mastered", "priority"):
        assert torch.equal(getattr(permuted, name), getattr(original, name)), name
        assert torch.equal(getattr(duplicated, name), getattr(original, name)), name
    assert torch.equal(permuted.visit_counts, original.visit_counts)
    assert torch.equal(duplicated.visit_counts, 2 * original.visit_counts)
    assert original.ewma_success[2, 2].item() == pytest.approx(0.25 * (2.0 / 3.0))


def test_segment_success_uses_tracking_thresholds_and_rejects_termination():
    curriculum = _make_curriculum()
    curriculum.set_current_cells(
        torch.tensor([0, 1]),
        torch.tensor([2, 2]),
        torch.tensor([1, 1]),
    )
    curriculum.accumulate(torch.tensor([0.10, 0.10]), torch.tensor([0.02, 0.02]), torch.tensor([0, 1]))
    curriculum.accumulate(torch.tensor([0.20, 0.20]), torch.tensor([0.04, 0.04]), torch.tensor([0, 1]))

    success = curriculum.finish_segments(
        torch.tensor([0, 1]),
        xy_success_threshold=torch.tensor([0.16, 0.16]),
        yaw_success_threshold=0.05,
        terminated=torch.tensor([False, True]),
    )

    assert success.tolist() == [True, False]
    # Both reports belong to the same orbit cell, so both are counted in exact lockstep.
    assert curriculum.visit_counts[2, 1].item() == 2
    assert curriculum.visit_counts[3, 3].item() == 2
    assert torch.equal(curriculum.segment_step_count, torch.zeros(4, dtype=torch.long))


def test_curriculum_checkpoint_round_trip_restores_global_state_and_rng_but_clears_segments():
    curriculum = _make_curriculum(seed=17)
    gait, velocity_bin = curriculum.sample_joint(4)
    curriculum.set_current_cells(torch.arange(4), gait, velocity_bin)
    curriculum.accumulate(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0.01, 0.02, 0.03, 0.04]))
    curriculum.update_cells(torch.tensor([2, 3]), torch.tensor([2, 2]), torch.tensor([True, True]))
    state = curriculum.state_dict()

    restored = _make_curriculum(seed=17, num_envs=6)
    restored.set_current_cells(torch.arange(6), torch.zeros(6, dtype=torch.long), torch.full((6,), 2))
    restored.accumulate(torch.ones(6), torch.ones(6))
    restored.load_state_dict(state)

    restored_state = restored.state_dict()
    assert restored_state["config"] == state["config"]
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored_state[name], value), name
    assert torch.equal(restored.current_gait_indices, torch.full((6,), -1, dtype=torch.long))
    assert torch.equal(restored.current_velocity_bin_indices, torch.full((6,), -1, dtype=torch.long))
    assert torch.equal(restored.segment_error_xy_sum, torch.zeros(6, dtype=torch.float64))
    assert torch.equal(restored.segment_error_yaw_sum, torch.zeros(6, dtype=torch.float64))
    assert torch.equal(restored.segment_step_count, torch.zeros(6, dtype=torch.long))
    restored_gait, restored_velocity = restored.sample_joint(32)
    original_gait, original_velocity = curriculum.sample_joint(32)
    assert torch.equal(restored_gait, original_gait)
    assert torch.equal(restored_velocity, original_velocity)


def test_schema_one_migration_keeps_global_competence_and_discards_transients():
    curriculum = _make_curriculum(seed=17, minimum_visits=3)
    legacy_config = {
        "num_envs": 99,
        **{
            name: value
            for name, value in curriculum._config_state().items()
            if name not in {"initial_max_abs_speed", "minimum_visits", "locked_cell_weight"}
        },
    }
    legacy_weights = torch.ones_like(curriculum.priority)
    legacy_ewma = torch.zeros_like(curriculum.ewma_success)
    legacy_visits = torch.zeros_like(curriculum.visit_counts)
    legacy_unlocked = torch.zeros_like(curriculum.eligible)
    legacy_unlocked[2, 1] = True
    legacy_unlocked[3, 3] = True
    legacy_visits[2, 1] = 3
    legacy_visits[3, 3] = 3
    legacy_ewma[2, 1] = 0.9
    legacy_ewma[3, 3] = 0.9
    legacy_state = {
        "schema_version": 1,
        "config": legacy_config,
        "gait_prior_weights": curriculum.gait_prior_weights.clone(),
        "weights": legacy_weights,
        "ewma_success": legacy_ewma,
        "visit_counts": legacy_visits,
        "unlock_state": legacy_unlocked,
        "current_gait_indices": torch.zeros(99, dtype=torch.long),
        "current_velocity_bin_indices": torch.zeros(99, dtype=torch.long),
        "segment_error_xy_sum": torch.ones(99, dtype=torch.float64),
        "segment_error_yaw_sum": torch.ones(99, dtype=torch.float64),
        "segment_step_count": torch.ones(99, dtype=torch.long),
        "rng_state": curriculum._generator.get_state().clone(),
    }

    with pytest.warns(UserWarning, match="stale per-environment cells"):
        curriculum.load_state_dict(legacy_state)

    assert curriculum.eligible[2, 1] and curriculum.eligible[3, 3]
    assert curriculum.mastered[2, 1] and curriculum.mastered[3, 3]
    assert torch.equal(curriculum.current_gait_indices, torch.full((4,), -1, dtype=torch.long))
    assert torch.equal(curriculum.segment_step_count, torch.zeros(4, dtype=torch.long))
    assert curriculum.state_dict()["schema_version"] == 2


def test_schema_one_migration_never_reactivates_configured_zero_prior_rows():
    active_prior = torch.ones(10)
    active_prior[2:4] = 0.0
    curriculum = _make_curriculum(initial_gait_weights=active_prior)
    legacy_config = {
        name: value
        for name, value in curriculum._config_state().items()
        if name not in {"initial_max_abs_speed", "minimum_visits", "locked_cell_weight"}
    }
    legacy_state = {
        "schema_version": 1,
        "config": legacy_config,
        # Schema 1 applied its exploration floor to originally zero rows.
        "gait_prior_weights": torch.ones(10, dtype=torch.float64),
        "weights": torch.ones_like(curriculum.priority),
        "ewma_success": torch.ones_like(curriculum.ewma_success),
        "visit_counts": torch.full_like(curriculum.visit_counts, 100),
        "unlock_state": torch.ones_like(curriculum.eligible),
        "rng_state": curriculum._generator.get_state().clone(),
    }

    with pytest.warns(UserWarning, match="stale per-environment cells"):
        curriculum.load_state_dict(legacy_state, legacy_gait_prior_weights=active_prior)

    assert torch.equal(curriculum.gait_prior_weights[2:4], torch.zeros(2, dtype=torch.float64))
    assert not torch.any(curriculum.eligible[2:4])
    assert not torch.any(curriculum.mastered[2:4])
    assert not torch.any(curriculum.weights[2:4])


@pytest.mark.parametrize("invalid_schema", [True, 2.0])
def test_curriculum_checkpoint_schema_requires_exact_integer_type(invalid_schema):
    curriculum = _make_curriculum()
    state = curriculum.state_dict()
    state["schema_version"] = invalid_schema

    with pytest.raises(ValueError, match="Unsupported curriculum checkpoint schema"):
        curriculum.validate_state_dict(state)


def test_command_reset_uses_joint_cell_sampling_and_retains_its_joint_gait(monkeypatch):
    command_term = _make_command_term(num_envs=2)
    curriculum = command_term.command_curriculum
    sampled_gaits = torch.tensor([2, 7])
    sampled_bins = torch.tensor([4, 0])
    calls: dict[str, object] = {}

    def sample_joint(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        calls["joint_count"] = count
        return sampled_gaits.clone(), sampled_bins.clone()

    def sample_forward_velocity(bin_indices: torch.Tensor) -> torch.Tensor:
        calls["forward_bins"] = bin_indices.clone()
        return curriculum.velocity_bin_centers[bin_indices].clone()

    def fail_conditional(*_args, **_kwargs):
        pytest.fail("Reset must sample a joint curriculum cell, not a conditional cell.")

    def fake_parent_reset(instance, env_ids):
        instance._resample_random_command(instance._resolve_env_ids(env_ids))
        return {"success_rate": 0.0}

    monkeypatch.setattr(curriculum, "sample_joint", sample_joint)
    monkeypatch.setattr(curriculum, "sample_forward_velocity", sample_forward_velocity)
    monkeypatch.setattr(curriculum, "sample_velocity_given_gait", fail_conditional)
    monkeypatch.setattr(curriculum, "sample_gait_given_velocity", fail_conditional)
    monkeypatch.setattr(symm_quadruped.CommandTerm, "reset", fake_parent_reset)

    command_term.reset([0, 1])

    assert calls["joint_count"] == 2
    assert torch.equal(calls["forward_bins"], sampled_bins)
    assert torch.equal(command_term.gait_row_indices, sampled_gaits)
    assert torch.equal(curriculum.current_gait_indices, sampled_gaits)
    assert torch.equal(curriculum.current_velocity_bin_indices, sampled_bins)
    assert torch.equal(curriculum.velocity_bin_indices(command_term.vel_command_b[:, 0]), sampled_bins)
    assert not torch.any(command_term._curriculum_joint_resample_pending)
    assert not torch.any(command_term._curriculum_skip_next_gait_resample)


def test_velocity_only_boundary_samples_speed_conditioned_on_current_gait(monkeypatch):
    command_term = _make_command_term(num_envs=3)
    curriculum = command_term.command_curriculum
    env_ids = torch.arange(3)
    current_gaits = torch.tensor([2, 5, 8])
    current_bins = torch.tensor([2, 2, 2])
    sampled_bins = torch.tensor([4, 0, 3])
    command_term._assign_gait(env_ids, current_gaits, add_theta_noise=False)
    curriculum.set_current_cells(env_ids, current_gaits, current_bins)
    captured: dict[str, torch.Tensor] = {}

    def sample_velocity_given_gait(gait_indices: torch.Tensor) -> torch.Tensor:
        captured["gaits"] = gait_indices.clone()
        return sampled_bins.clone()

    monkeypatch.setattr(curriculum, "sample_velocity_given_gait", sample_velocity_given_gait)
    monkeypatch.setattr(
        curriculum,
        "sample_forward_velocity",
        lambda bin_indices: curriculum.velocity_bin_centers[bin_indices].clone(),
    )
    monkeypatch.setattr(curriculum, "sample_joint", lambda *_args: pytest.fail("Unexpected joint sampling."))

    command_term._resample_random_command(env_ids)

    assert torch.equal(captured["gaits"], current_gaits)
    assert torch.equal(command_term.gait_row_indices, current_gaits)
    assert torch.equal(curriculum.current_gait_indices, current_gaits)
    assert torch.equal(curriculum.current_velocity_bin_indices, sampled_bins)
    assert torch.equal(curriculum.velocity_bin_indices(command_term.vel_command_b[:, 0]), sampled_bins)


def test_gait_only_boundary_samples_gait_conditioned_on_current_speed(monkeypatch):
    command_term = _make_command_term(num_envs=3)
    curriculum = command_term.command_curriculum
    env_ids = torch.arange(3)
    initial_gaits = torch.tensor([0, 3, 6])
    current_bins = torch.tensor([0, 2, 4])
    sampled_gaits = torch.tensor([9, 4, 1])
    command_term._assign_gait(env_ids, initial_gaits, add_theta_noise=False)
    command_term.vel_command_b[:, 0] = curriculum.velocity_bin_centers[current_bins].to(torch.float32)
    curriculum.set_current_cells(env_ids, initial_gaits, current_bins)
    previous_velocity = command_term.vel_command_b.clone()
    captured: dict[str, torch.Tensor] = {}

    def sample_gait_given_velocity(bin_indices: torch.Tensor) -> torch.Tensor:
        captured["velocity_bins"] = bin_indices.clone()
        return sampled_gaits.clone()

    monkeypatch.setattr(curriculum, "sample_gait_given_velocity", sample_gait_given_velocity)
    monkeypatch.setattr(curriculum, "sample_joint", lambda *_args: pytest.fail("Unexpected joint sampling."))
    monkeypatch.setattr(
        curriculum,
        "sample_velocity_given_gait",
        lambda *_args: pytest.fail("Unexpected velocity-conditional sampling."),
    )

    command_term._resample_gait(env_ids)

    assert torch.equal(captured["velocity_bins"], current_bins)
    assert torch.equal(command_term.vel_command_b, previous_velocity)
    assert torch.equal(command_term.gait_row_indices, sampled_gaits)
    assert torch.equal(curriculum.current_gait_indices, sampled_gaits)
    assert torch.equal(curriculum.current_velocity_bin_indices, current_bins)


def test_evaluation_override_bypasses_all_curriculum_sampling(monkeypatch):
    command_term = _make_command_term(num_envs=2)
    curriculum = command_term.command_curriculum

    def fail_sampling(*_args, **_kwargs):
        pytest.fail("Evaluation must bypass curriculum sampling.")

    monkeypatch.setattr(curriculum, "sample_joint", fail_sampling)
    monkeypatch.setattr(curriculum, "sample_forward_velocity", fail_sampling)
    monkeypatch.setattr(curriculum, "sample_velocity_given_gait", fail_sampling)
    monkeypatch.setattr(curriculum, "sample_gait_given_velocity", fail_sampling)

    command_term.set_evaluation_scenario(vx_mps=-1.25, gait_index=8, env_ids=[1])
    command_term._resample_command([1])
    command_term._resample_gait([1])

    assert command_term._evaluation_active.tolist() == [False, True]
    assert command_term.vel_command_b[1].tolist() == [-1.25, 0.0, 0.0]
    assert command_term.gait_row_indices[1].item() == 8
    assert torch.equal(command_term.foot_thetas[1], command_term.init_foot_thetas[8])
    assert curriculum.current_gait_indices[1].item() == -1
    assert curriculum.current_velocity_bin_indices[1].item() == -1
    assert torch.isinf(command_term.time_left[1])
    assert torch.isinf(command_term.gait_time_left[1])


def test_environment_command_curriculum_state_restores_global_state_and_keeps_fresh_runtime():
    source = _make_command_term(num_envs=4)
    target = _make_command_term(num_envs=4)
    fresh_runtime_fields = (
        "vel_command_b",
        "heading_target",
        "is_heading_env",
        "is_standing_env",
        "foot_thetas",
        "gait_periods",
        "duty_factors",
        "kappa",
        "time_left",
        "command_counter",
        "gait_time_left",
        "gait_counter",
        "gait_row_indices",
        "gait_sequence_indices",
    )
    for index, name in enumerate(fresh_runtime_fields):
        tensor = getattr(target, name)
        values = torch.arange(tensor.numel(), dtype=torch.long).reshape(tensor.shape) + index + 1
        if tensor.dtype == torch.bool:
            tensor.copy_((values % 2).to(torch.bool))
        elif tensor.dtype.is_floating_point:
            tensor.copy_(values.to(tensor.dtype).mul(0.03125).add(index + 0.25))
        else:
            tensor.copy_(values.to(tensor.dtype))
    target._common_gait_phase_at_anchor.copy_(torch.tensor([0.125, 0.25, 0.5, 0.75]))
    target._gait_phase_anchor_steps.copy_(torch.tensor([2, 3, 4, 5]))
    target._error_xy_sum.fill_(7.0)
    target._error_yaw_sum.fill_(8.0)
    target._step_count.fill_(9.0)
    target._env.common_step_counter = 9
    fresh_runtime = {name: getattr(target, name).clone() for name in fresh_runtime_fields}
    fresh_phase = target.common_gait_phases().clone()

    active_gaits = torch.tensor([0, 2, 5, 9])
    active_bins = torch.tensor([2, 2, 2, 2])
    source.gait_row_indices.copy_(active_gaits)
    source.vel_command_b[:, 0] = source.command_curriculum.velocity_bin_centers[active_bins].to(torch.float32)
    source.foot_thetas.copy_(source.init_foot_thetas[active_gaits])
    source.command_curriculum.set_current_cells(torch.arange(4), active_gaits, active_bins)
    source.command_curriculum.accumulate(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([0.01, 0.02, 0.03, 0.04]),
    )
    source.command_curriculum.update_cells(
        torch.tensor([2, 3]),
        torch.tensor([2, 2]),
        torch.tensor([True, True]),
    )
    source.command_curriculum.sample_joint(11)
    source._training_iteration = 37

    source_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: source))
    target_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: target))
    state = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_command_curriculum_state(source_env)

    assert set(state) == {"schema_version", "training_iteration", "semantic_config", "curriculum"}
    assert state["schema_version"] == 3
    assert set(state["curriculum"]) == {
        "schema_version",
        "config",
        "gait_prior_weights",
        "priority",
        "eligible",
        "mastered",
        "ewma_success",
        "visit_counts",
        "rng_state",
    }

    symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.load_command_curriculum_state(target_env, state)
    assert target._training_iteration == 37
    for name, expected in fresh_runtime.items():
        assert torch.equal(getattr(target, name), expected), name
    assert torch.equal(target.common_gait_phases(), fresh_phase)
    assert torch.equal(target.command_curriculum.current_gait_indices, torch.full((4,), -1, dtype=torch.long))
    assert torch.equal(
        target.command_curriculum.current_velocity_bin_indices,
        torch.full((4,), -1, dtype=torch.long),
    )
    assert torch.equal(target.command_curriculum.segment_step_count, torch.zeros(4, dtype=torch.long))
    assert torch.equal(target._error_xy_sum, torch.zeros(4))
    assert torch.equal(target._error_yaw_sum, torch.zeros(4))
    assert torch.equal(target._step_count, torch.zeros(4))
    assert torch.all(target._curriculum_joint_resample_pending)
    assert not torch.any(target._curriculum_skip_next_gait_resample)
    restored = target.command_curriculum.state_dict()
    _assert_nested_state_equal(restored, state["curriculum"])
    source_gait, source_bin = source.command_curriculum.sample_joint(64)
    target_gait, target_bin = target.command_curriculum.sample_joint(64)
    assert torch.equal(target_gait, source_gait)
    assert torch.equal(target_bin, source_bin)


def test_environment_curriculum_preflight_is_type_exact_and_nonmutating():
    command_term = _make_command_term(num_envs=2)
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command_term))
    state = command_term.get_command_curriculum_state()
    before = command_term.command_curriculum.state_dict()
    state["semantic_config"]["gait_curriculum_iterations"] = float(
        state["semantic_config"]["gait_curriculum_iterations"]
    )

    with pytest.raises(ValueError, match="semantic configuration"):
        symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.validate_command_curriculum_state(
            environment,
            state,
        )

    _assert_nested_state_equal(command_term.command_curriculum.state_dict(), before)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("negative_priority", "priority must be finite"),
        ("asymmetric_eligible", "eligible is not equal"),
        ("mastered_ineligible", "mastered cells must remain eligible"),
        ("invalid_rng", "rng_state is not a valid CPU generator state"),
    ),
)
def test_environment_curriculum_preflight_rejects_invalid_values_without_mutation(corruption, message):
    command_term = _make_command_term(num_envs=2)
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command_term))
    state = command_term.get_command_curriculum_state()
    before = command_term.command_curriculum.state_dict()
    curriculum = command_term.command_curriculum
    curriculum_state = state["curriculum"]
    partner_gait = int(curriculum.gait_partner_indices[0])
    partner_bin = int(curriculum.velocity_partner_indices[0])

    if corruption == "negative_priority":
        curriculum_state["priority"][0, 0] = -1.0
        curriculum_state["priority"][partner_gait, partner_bin] = -1.0
    elif corruption == "asymmetric_eligible":
        curriculum_state["eligible"][0, 0] = ~curriculum_state["eligible"][0, 0]
    elif corruption == "mastered_ineligible":
        curriculum_state["eligible"][0, 0] = False
        curriculum_state["eligible"][partner_gait, partner_bin] = False
        curriculum_state["mastered"][0, 0] = True
        curriculum_state["mastered"][partner_gait, partner_bin] = True
    else:
        curriculum_state["rng_state"] = torch.zeros(1, dtype=torch.uint8)

    with pytest.raises(ValueError, match=message):
        symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.validate_command_curriculum_state(
            environment,
            state,
        )

    _assert_nested_state_equal(command_term.command_curriculum.state_dict(), before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("heading_command", True, "heading_command=False"),
        ("min_xy_command_norm", 0.1, "min_xy_command_norm=0"),
    ),
)
def test_tr_orbit_curriculum_rejects_command_postprocessing(monkeypatch, field, value, message):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.asset_name = "robot"
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(-torch.pi, torch.pi),
    )
    cfg.command_curriculum_mode = symm_quadruped.TR_ORBIT_COMMAND_CURRICULUM_MODE
    cfg.rel_standing_envs = 0.0
    setattr(cfg, field, value)

    def initialize_base(command_term, _cfg, _env):
        command_term.cfg = _cfg
        command_term._env = _env
        command_term.metrics = {}

    monkeypatch.setattr(symm_quadruped.CommandTerm, "__init__", initialize_base)
    environment = SimpleNamespace(
        scene={"robot": object()},
        common_step_counter=0,
        num_envs=1,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=message):
        symm_quadruped.GaitVelocityCommand(cfg, environment)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"velocity_range": (-1.0, 2.0)}, "symmetric"),
        ({"velocity_bin_count": 0}, "positive integer"),
        ({"ewma_coefficient": float("nan")}, "finite"),
        ({"unlock_threshold": 1.1}, "in [0, 1]"),
        ({"exploration_floor": 0.0}, "positive"),
        ({"maximum_weight": 0.01}, "greater than or equal"),
        ({"initial_max_abs_speed": -0.1}, "nonnegative"),
        ({"initial_max_abs_speed": 2.1}, "must not exceed"),
        (
            {"initial_max_abs_speed": 0.1, "velocity_bin_count": 4},
            "does not include any velocity-bin center",
        ),
        ({"minimum_visits": 0}, "positive integer"),
        ({"locked_cell_weight": -0.1}, "in [0, maximum_weight]"),
        ({"locked_cell_weight": 3.1}, "in [0, maximum_weight]"),
    ),
)
def test_invalid_curriculum_configuration_fails_fast(override, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _make_curriculum(**override)
