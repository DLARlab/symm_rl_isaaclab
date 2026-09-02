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


def test_arbitrary_updates_preserve_all_orbit_state_and_unlock_neighbors_symmetrically():
    curriculum = _make_curriculum()

    curriculum.update_cells(
        torch.tensor([2, 8, 1, 2]),
        torch.tensor([0, 3, 2, 0]),
        torch.tensor([True, False, True, True]),
    )

    for value in (
        curriculum.weights,
        curriculum.ewma_success,
        curriculum.visit_counts,
        curriculum.unlock_state,
        curriculum.sampling_eligibility,
    ):
        _assert_orbit_equal(curriculum, value)
    assert curriculum.unlock_state[2, 0]
    assert curriculum.unlock_state[3, 4]
    assert curriculum.unlock_state[2, 1]
    assert curriculum.unlock_state[3, 3]
    assert torch.all(curriculum.weights >= curriculum.exploration_floor)
    assert torch.all(curriculum.weights <= curriculum.maximum_weight)


def test_conditional_sampling_uses_the_selected_weight_row_or_column():
    curriculum = _make_curriculum(exploration_floor=1.0e-12, maximum_weight=100.0)
    curriculum.weights.fill_(1.0e-12)
    # Preserve orbit equality while making (g=2, k=4) and its partner dominant.
    curriculum.weights[2, 4] = 100.0
    curriculum.weights[3, 0] = 100.0

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
    )
    curriculum.update_cells(torch.tensor([2]), torch.tensor([0]), torch.tensor([True]))
    learned_preferences = curriculum.weights / curriculum.gait_prior_weights[:, None]
    updated_prior = torch.tensor((2.0, 4.0, 1.5, 1.5, 6.0, 6.0, 7.0, 8.0, 7.0, 8.0))

    curriculum.update_gait_prior_weights(updated_prior)

    assert torch.equal(curriculum.gait_prior_weights, updated_prior.to(torch.float64))
    assert torch.allclose(
        curriculum.weights / curriculum.gait_prior_weights[:, None],
        learned_preferences,
        atol=1.0e-12,
        rtol=0.0,
    )
    _assert_orbit_equal(curriculum, curriculum.weights)


def test_self_orbit_neighbor_is_incremented_once_when_both_adjacent_bins_share_it():
    curriculum = _make_curriculum(
        current_cell_increment=0.0,
        neighbor_increment=0.25,
        exploration_floor=1.0,
        maximum_weight=10.0,
    )

    curriculum.update_cells(torch.tensor([0]), torch.tensor([2]), torch.tensor([True]))

    assert curriculum.weights[0, 1].item() == 1.25
    assert curriculum.weights[0, 3].item() == 1.25
    assert curriculum.weights[0, 0].item() == 1.0
    assert curriculum.weights[0, 4].item() == 1.0


def test_even_bin_neighbor_does_not_reincrement_the_current_orbit():
    curriculum = _make_curriculum(
        velocity_bin_count=4,
        current_cell_increment=1.0,
        neighbor_increment=0.25,
        exploration_floor=1.0,
        maximum_weight=10.0,
    )

    curriculum.update_cells(torch.tensor([0]), torch.tensor([1]), torch.tensor([True]))

    assert torch.equal(curriculum.weights[0], torch.tensor([1.25, 2.0, 2.0, 1.25], dtype=torch.float64))


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


def test_curriculum_checkpoint_round_trip_restores_exact_rng_and_current_segments():
    curriculum = _make_curriculum(seed=17)
    gait, velocity_bin = curriculum.sample_joint(4)
    curriculum.set_current_cells(torch.arange(4), gait, velocity_bin)
    curriculum.accumulate(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0.01, 0.02, 0.03, 0.04]))
    curriculum.update_cells(torch.tensor([2, 6]), torch.tensor([0, 4]), torch.tensor([True, False]))
    state = curriculum.state_dict()

    restored = _make_curriculum(seed=17)
    restored.load_state_dict(state)

    restored_state = restored.state_dict()
    assert restored_state["config"] == state["config"]
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored_state[name], value), name
    assert torch.equal(restored.sample_joint(32)[0], curriculum.sample_joint(32)[0])
    # Compare a fresh pair from the now-equally advanced generators.
    restored_gait, restored_velocity = restored.sample_joint(32)
    original_gait, original_velocity = curriculum.sample_joint(32)
    assert torch.equal(restored_gait, original_gait)
    assert torch.equal(restored_velocity, original_velocity)


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


def test_environment_command_curriculum_state_round_trip_restores_every_runtime_tensor_and_rng():
    source = _make_command_term(num_envs=4)
    target = _make_command_term(num_envs=4)
    runtime_fields = source._COMMAND_CURRICULUM_RUNTIME_TENSOR_FIELDS
    phase_field = source._COMMAND_CURRICULUM_RUNTIME_PHASE_FIELD

    for index, name in enumerate(runtime_fields):
        if name == phase_field:
            continue
        tensor = getattr(source, name)
        values = torch.arange(tensor.numel(), dtype=torch.long).reshape(tensor.shape) + index
        if tensor.dtype == torch.bool:
            tensor.copy_((values % 2).to(torch.bool))
        elif tensor.dtype.is_floating_point:
            tensor.copy_(values.to(tensor.dtype).mul(0.03125).add(index + 0.5))
        else:
            tensor.copy_(values.to(tensor.dtype))

    active_gaits = torch.tensor([0, 2, 5, 9])
    active_bins = torch.tensor([0, 1, 3, 4])
    source.gait_row_indices.copy_(active_gaits)
    source.vel_command_b[:, 0] = source.command_curriculum.velocity_bin_centers[active_bins].to(torch.float32)
    source.foot_thetas.copy_(source.init_foot_thetas[active_gaits])
    source.command_curriculum.set_current_cells(torch.arange(4), active_gaits, active_bins)
    source.command_curriculum.accumulate(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([0.01, 0.02, 0.03, 0.04]),
    )
    source.command_curriculum.update_cells(torch.tensor([2]), torch.tensor([0]), torch.tensor([True]))
    source.command_curriculum.sample_joint(11)
    source._training_iteration = 37
    source._env.common_step_counter = 123
    source._common_gait_phase_at_anchor.copy_(torch.tensor([0.125, 0.25, 0.5, 0.75]))
    source._gait_phase_anchor_steps.copy_(torch.tensor([90, 91, 92, 93]))
    target._env.common_step_counter = 7

    source_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: source))
    target_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: target))
    state = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_command_curriculum_state(source_env)

    assert set(state) == {"schema_version", "training_iteration", "curriculum", "runtime"}
    assert set(state["runtime"]) == set(runtime_fields)
    for name in runtime_fields:
        expected = source.common_gait_phases() if name == phase_field else getattr(source, name)
        assert torch.equal(state["runtime"][name], expected), name

    symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.load_command_curriculum_state(target_env, state)
    assert torch.equal(target.common_gait_phases(), source.common_gait_phases())
    source._env.common_step_counter += 1
    target._env.common_step_counter += 1
    assert torch.equal(target.common_gait_phases(), source.common_gait_phases())
    restored = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_command_curriculum_state(target_env)
    advanced_source_state = symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv.get_command_curriculum_state(source_env)

    _assert_nested_state_equal(restored, advanced_source_state)
    source_gait, source_bin = source.command_curriculum.sample_joint(64)
    target_gait, target_bin = target.command_curriculum.sample_joint(64)
    assert torch.equal(target_gait, source_gait)
    assert torch.equal(target_bin, source_bin)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"velocity_range": (-1.0, 2.0)}, "symmetric"),
        ({"velocity_bin_count": 0}, "positive integer"),
        ({"ewma_coefficient": float("nan")}, "finite"),
        ({"unlock_threshold": 1.1}, "in [0, 1]"),
        ({"exploration_floor": 0.0}, "positive"),
        ({"maximum_weight": 0.01}, "greater than or equal"),
    ),
)
def test_invalid_curriculum_configuration_fails_fast(override, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _make_curriculum(**override)
