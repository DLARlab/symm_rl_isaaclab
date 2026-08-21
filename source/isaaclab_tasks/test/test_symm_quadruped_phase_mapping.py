# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared symmetric quadruped gait and time-reversal phase mappings."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

_LAYOUT = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_LAYOUT
_OBS_DIM = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_DIM


def _wrap(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase, 1.0)


def _encode_phase(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = 2.0 * torch.pi * phase
    return torch.sin(angle), torch.cos(angle)


def _decode_phase(phase_sin: torch.Tensor, phase_cos: torch.Tensor) -> torch.Tensor:
    return _wrap(torch.atan2(phase_sin, phase_cos) / (2.0 * torch.pi))


def _assert_phases_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1.0e-10) -> None:
    circular_error = torch.remainder(actual - expected + 0.5, 1.0) - 0.5
    assert torch.allclose(circular_error, torch.zeros_like(circular_error), atol=atol, rtol=0.0)


def _make_valid_observations(
    leading_shape: tuple[int, ...],
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn((*leading_shape, _OBS_DIM), generator=generator, dtype=dtype)
    phase = torch.rand((*leading_shape, 4), generator=generator, dtype=dtype)
    theta = torch.rand((*leading_shape, 4), generator=generator, dtype=dtype)
    beta = 0.2 + 0.6 * torch.rand((*leading_shape, 1), generator=generator, dtype=dtype)
    swing_ratio = 1.0 - beta

    phase_sin, phase_cos = _encode_phase(phase)
    theta_sin, theta_cos = _encode_phase(theta)
    obs[..., _LAYOUT.foot_phase_sin] = phase_sin
    obs[..., _LAYOUT.foot_phase_cos] = phase_cos
    obs[..., _LAYOUT.foot_theta_sin] = theta_sin
    obs[..., _LAYOUT.foot_theta_cos] = theta_cos
    obs[..., _LAYOUT.phase_ratios] = torch.cat((swing_ratio, beta), dim=-1)
    return obs, phase, theta, beta


def _configured_gaits() -> torch.Tensor:
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    return torch.tensor(cfg.init_foot_thetas, dtype=torch.float64)


def _initialize_gait_clock(
    command_term: symm_quadruped.GaitVelocityCommand,
    *,
    anchor_phase: float | torch.Tensor = 0.0,
    anchor_step: int | torch.Tensor = 0,
) -> None:
    """Initialize the lazy piecewise-integrated gait clock for a test command term."""
    phase = torch.as_tensor(
        anchor_phase,
        dtype=command_term.gait_periods.dtype,
        device=command_term.gait_periods.device,
    )
    steps = torch.as_tensor(anchor_step, dtype=torch.long, device=command_term.gait_periods.device)
    command_term._common_gait_phase_at_anchor = torch.broadcast_to(phase, command_term.gait_periods.shape).clone()
    command_term._gait_phase_anchor_steps = torch.broadcast_to(steps, command_term.gait_periods.shape).clone()


def _leg_permutation_penalty(
    foot_thetas: torch.Tensor,
    joint_pos: torch.Tensor,
    *,
    leg_pairs: tuple[tuple[str, str], ...],
    phase_sync_tolerance: float = 0.02,
) -> torch.Tensor:
    gait_command = SimpleNamespace(foot_thetas=foot_thetas)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_pos)))},
        command_manager=SimpleNamespace(get_term=lambda _: gait_command),
        num_envs=joint_pos.shape[0],
        device=joint_pos.device,
    )
    return symm_quadruped.leg_permutation_symmetry_penalty(
        env,
        command_name="base_velocity",
        joint_cfg=SimpleNamespace(joint_ids=list(range(12))),
        leg_pairs=leg_pairs,
        phase_sync_tolerance=phase_sync_tolerance,
    )


def test_policy_observation_layout_remains_72d_and_in_original_order():
    assert _OBS_DIM == 72
    assert (
        _LAYOUT.measured_base_twist,
        _LAYOUT.projected_gravity,
        _LAYOUT.desired_base_twist,
        _LAYOUT.joint_position,
        _LAYOUT.joint_velocity,
        _LAYOUT.previous_action,
        _LAYOUT.foot_phase_sin,
        _LAYOUT.foot_phase_cos,
        _LAYOUT.foot_theta_sin,
        _LAYOUT.foot_theta_cos,
        _LAYOUT.phase_ratios,
        _LAYOUT.sagittal_plane_state,
    ) == (
        slice(0, 6),
        slice(6, 9),
        slice(9, 15),
        slice(15, 27),
        slice(27, 39),
        slice(39, 51),
        slice(51, 55),
        slice(55, 59),
        slice(59, 63),
        slice(63, 67),
        slice(67, 69),
        slice(69, 72),
    )
    assert _LAYOUT.swing_ratio == slice(67, 68)
    assert _LAYOUT.stance_ratio == slice(68, 69)


def test_duty_aware_phase_reflection_is_an_involution():
    generator = torch.Generator().manual_seed(1)
    beta = 0.15 + 0.7 * torch.rand((2, 3, 1), generator=generator, dtype=torch.float64)
    swing_ratio = 1.0 - beta
    mode_is_swing = torch.rand((2, 3, 4), generator=generator) < 0.5
    interior = 0.05 + 0.9 * torch.rand((2, 3, 4), generator=generator, dtype=torch.float64)
    phase = torch.where(mode_is_swing, interior * swing_ratio, swing_ratio + interior * beta)
    phase_sin, phase_cos = _encode_phase(phase)

    phase_sin_tr, phase_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(phase_sin, phase_cos, swing_ratio)
    phase_sin_tt, phase_cos_tt = symm_quadruped.time_reverse_phase_sin_cos(phase_sin_tr, phase_cos_tr, swing_ratio)

    assert phase_sin_tt.shape == phase.shape
    assert phase_sin_tt.dtype == phase.dtype
    assert phase_sin_tt.device == phase.device
    assert torch.allclose(phase_sin_tt, phase_sin, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(phase_cos_tt, phase_cos, atol=1.0e-12, rtol=0.0)


def test_duty_aware_phase_reflection_preserves_contact_mode_away_from_boundaries():
    generator = torch.Generator().manual_seed(2)
    beta = 0.15 + 0.7 * torch.rand((128, 1), generator=generator, dtype=torch.float64)
    swing_ratio = 1.0 - beta
    mode_is_swing = torch.rand((128, 4), generator=generator) < 0.5
    interior = 0.05 + 0.9 * torch.rand((128, 4), generator=generator, dtype=torch.float64)
    phase = torch.where(mode_is_swing, interior * swing_ratio, swing_ratio + interior * beta)
    phase_sin, phase_cos = _encode_phase(phase)

    phase_sin_tr, phase_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(phase_sin, phase_cos, swing_ratio)
    phase_tr = _decode_phase(phase_sin_tr, phase_cos_tr)

    assert torch.equal(phase < swing_ratio, phase_tr < swing_ratio)
    assert torch.equal(phase >= swing_ratio, phase_tr >= swing_ratio)


def test_duty_aware_phase_reflection_exchanges_liftoff_and_touchdown_boundaries():
    beta = torch.tensor([[0.45], [0.30], [0.60]], dtype=torch.float64)
    swing_ratio = 1.0 - beta

    zero = torch.zeros((3, 4), dtype=torch.float64)
    zero_sin, zero_cos = _encode_phase(zero)
    zero_sin_tr, zero_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(zero_sin, zero_cos, swing_ratio)
    _assert_phases_close(_decode_phase(zero_sin_tr, zero_cos_tr), swing_ratio.expand(-1, 4))

    touchdown = swing_ratio.expand(-1, 4)
    touchdown_sin, touchdown_cos = _encode_phase(touchdown)
    touchdown_sin_tr, touchdown_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(
        touchdown_sin, touchdown_cos, swing_ratio
    )
    _assert_phases_close(_decode_phase(touchdown_sin_tr, touchdown_cos_tr), zero)


def test_duty_aware_phase_reflection_preserves_unit_circle_norm():
    phase = torch.linspace(0.01, 0.99, 24, dtype=torch.float64).reshape(2, 3, 4)
    swing_ratio = torch.tensor([[[0.55]], [[0.62]]], dtype=torch.float64)
    phase_sin, phase_cos = _encode_phase(phase)

    phase_sin_tr, phase_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(phase_sin, phase_cos, swing_ratio)

    norm = phase_sin.square() + phase_cos.square()
    norm_tr = phase_sin_tr.square() + phase_cos_tr.square()
    assert torch.allclose(norm_tr, norm, atol=1.0e-12, rtol=0.0)


def test_duty_aware_phase_reflection_does_not_silently_clamp_ratio_values():
    phase = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float64)
    phase_sin, phase_cos = _encode_phase(phase)
    swing_ratio = torch.tensor([[1.2]], dtype=torch.float64)

    phase_sin_tr, phase_cos_tr = symm_quadruped.time_reverse_phase_sin_cos(phase_sin, phase_cos, swing_ratio)
    expected_sin, expected_cos = _encode_phase(_wrap(swing_ratio - phase))

    assert torch.allclose(phase_sin_tr, expected_sin, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(phase_cos_tr, expected_cos, atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("phase_shape", "cos_shape", "swing_shape", "message"),
    [
        ((2, 3), (2, 3), (2, 1), "four leg channels"),
        ((2, 4), (3, 4), (2, 1), "match phase_sin shape"),
        ((2, 4), (2, 4), (2,), "singleton final dimension"),
        ((2, 4), (2, 4), (3, 1), "broadcastable"),
    ],
)
def test_duty_aware_phase_reflection_rejects_incompatible_shapes(
    phase_shape: tuple[int, ...],
    cos_shape: tuple[int, ...],
    swing_shape: tuple[int, ...],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        symm_quadruped.time_reverse_phase_sin_cos(
            torch.zeros(phase_shape),
            torch.ones(cos_shape),
            torch.full(swing_shape, 0.55),
        )


def test_full_observation_time_reversal_is_an_involution_for_arbitrary_leading_dimensions():
    obs, _, _, _ = _make_valid_observations((2, 3), seed=3)

    obs_tt = symm_quadruped.time_reverse_observations(symm_quadruped.time_reverse_observations(obs))

    assert obs_tt.shape == (2, 3, _OBS_DIM)
    assert torch.allclose(obs_tt, obs, atol=1.0e-11, rtol=0.0)


def test_time_reversed_observation_closes_on_one_common_transformed_gait_clock():
    generator = torch.Generator().manual_seed(4)
    phi = torch.rand((16, 1), generator=generator, dtype=torch.float64)
    theta = torch.rand((16, 4), generator=generator, dtype=torch.float64)
    beta = 0.2 + 0.6 * torch.rand((16, 1), generator=generator, dtype=torch.float64)
    swing_ratio = 1.0 - beta
    phase = _wrap(phi + theta)
    obs = torch.zeros((16, _OBS_DIM), dtype=torch.float64)
    obs[..., _LAYOUT.foot_phase_sin], obs[..., _LAYOUT.foot_phase_cos] = _encode_phase(phase)
    obs[..., _LAYOUT.foot_theta_sin], obs[..., _LAYOUT.foot_theta_cos] = _encode_phase(theta)
    obs[..., _LAYOUT.phase_ratios] = torch.cat((swing_ratio, beta), dim=-1)

    obs_tr = symm_quadruped.time_reverse_observations(obs)
    phase_tr = _decode_phase(
        obs_tr[..., _LAYOUT.foot_phase_sin],
        obs_tr[..., _LAYOUT.foot_phase_cos],
    )
    theta_tr = _decode_phase(
        obs_tr[..., _LAYOUT.foot_theta_sin],
        obs_tr[..., _LAYOUT.foot_theta_cos],
    )

    _assert_phases_close(phase_tr, _wrap(swing_ratio - phase))
    _assert_phases_close(theta_tr, _wrap(-theta))
    common_phase_tr = _wrap(phase_tr - theta_tr)
    _assert_phases_close(common_phase_tr, _wrap(swing_ratio - phi).expand_as(common_phase_tr))


def test_same_gait_phase_helper_is_independent_of_command_sign_and_preserves_tensor_properties():
    assert tuple(inspect.signature(symm_quadruped.compute_same_gait_foot_phases).parameters) == (
        "common_phase",
        "foot_thetas",
    )
    commands = (-2.0, -1.0e-3, 0.0, 1.0e-3, 2.0)
    common_phase = torch.tensor([[0.37]], dtype=torch.float64)
    foot_thetas = torch.tensor([[0.13, -0.13, 0.5, 0.5]], dtype=torch.float64)

    phases_by_command = [symm_quadruped.compute_same_gait_foot_phases(common_phase, foot_thetas) for _ in commands]

    assert all(torch.equal(phases, phases_by_command[0]) for phases in phases_by_command[1:])
    assert phases_by_command[0].dtype == common_phase.dtype
    assert phases_by_command[0].device == common_phase.device
    assert torch.allclose(phases_by_command[0], _wrap(common_phase + foot_thetas))


def test_environment_foot_phases_use_continuous_clock_for_every_command_sign():
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    commands = torch.tensor([-2.0, -1.0e-3, 0.0, 1.0e-3, 2.0], dtype=torch.float64)
    command_term._env = SimpleNamespace(
        common_step_counter=37,
        step_dt=0.02,
    )
    command_term.gait_periods = torch.full((len(commands),), 0.5, dtype=torch.float64)
    _initialize_gait_clock(command_term)
    command_term.foot_thetas = torch.tensor(
        [[0.13, -0.13, 0.5, 0.5]] * len(commands),
        dtype=torch.float64,
    )
    command_term.vel_command_b = torch.stack((commands, torch.zeros_like(commands), torch.zeros_like(commands)), dim=-1)

    phases = command_term.foot_phases()

    assert phases.dtype == command_term.gait_periods.dtype
    assert all(torch.equal(phase, phases[0]) for phase in phases[1:])
    common_phase = torch.tensor([[37 * 0.02 / 0.5]], dtype=torch.float64)
    assert torch.allclose(
        phases,
        symm_quadruped.compute_same_gait_foot_phases(common_phase, command_term.foot_thetas),
    )


def test_environment_foot_phases_add_offsets_before_wrapping_continuous_clock():
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term._env = SimpleNamespace(
        common_step_counter=16,
        step_dt=0.02,
    )
    command_term.gait_periods = torch.tensor([0.2001], dtype=torch.float32)
    _initialize_gait_clock(command_term)
    command_term.foot_thetas = torch.tensor([[0.13, -0.13, 0.5, 0.5]], dtype=torch.float32)

    unwrapped_clock = (
        torch.tensor([command_term._env.common_step_counter], dtype=torch.float32)
        * command_term._env.step_dt
        / command_term.gait_periods
    )
    expected = torch.remainder(unwrapped_clock.unsqueeze(-1) + command_term.foot_thetas, 1.0)
    prewrapped = torch.remainder(
        torch.remainder(unwrapped_clock, 1.0).unsqueeze(-1) + command_term.foot_thetas,
        1.0,
    )

    assert not torch.equal(expected, prewrapped)
    assert torch.equal(command_term.common_gait_phases(), unwrapped_clock)
    assert torch.equal(command_term.foot_phases(), expected)


def test_velocity_resample_refreshes_gait_timing_from_final_command():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_period = False
    cfg.heading_command = False
    cfg.rel_standing_envs = 0.0
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(1.2, 1.2),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(0.0, 0.0),
    )
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=25,
        step_dt=0.02,
    )
    command_term.vel_command_b = torch.zeros((1, 3))
    command_term.heading_target = torch.zeros(1)
    command_term.is_heading_env = torch.zeros(1, dtype=torch.bool)
    command_term.is_standing_env = torch.zeros(1, dtype=torch.bool)
    command_term.gait_periods = torch.full((1,), 0.5)
    command_term.duty_factors = torch.full((1,), 0.45)
    command_term.foot_thetas = torch.tensor(((0.0, 0.5, 0.5, 0.0),))
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term)

    common_phase_before = command_term.common_gait_phases().clone()
    phase_before = command_term.foot_phases().clone()
    command_term._resample_command([0])
    command_term._update_velocity_resampled_gait_timing(torch.tensor([0]), torch.empty(0, dtype=torch.long))
    phase_after = command_term.foot_phases()

    expected_period = command_term._compute_period_from_forward_velocity(torch.tensor([1.2]))
    expected_duty_factor = command_term._compute_duty_factor_from_forward_velocity(torch.tensor([1.2]))
    assert command_term.vel_command_b[0, 0].item() == pytest.approx(1.2)
    assert torch.allclose(command_term.gait_periods, expected_period)
    assert torch.allclose(command_term.duty_factors, expected_duty_factor)
    assert torch.allclose(
        command_term.phase_ratios(),
        torch.stack((1.0 - expected_duty_factor, expected_duty_factor), dim=-1),
    )
    _assert_phases_close(command_term.common_gait_phases(), common_phase_before)
    _assert_phases_close(phase_after, phase_before)

    command_term._env.common_step_counter += 10
    expected_common_phase = torch.remainder(common_phase_before, 1.0) + 10 * command_term._env.step_dt / expected_period
    expected_phase = symm_quadruped.compute_same_gait_foot_phases(
        expected_common_phase.unsqueeze(-1), command_term.foot_thetas
    )
    _assert_phases_close(command_term.foot_phases(), expected_phase, atol=2.0e-7)


def test_standing_velocity_resample_refreshes_timing_from_final_zero_command():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_period = False
    cfg.heading_command = False
    cfg.rel_standing_envs = 1.0
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(1.2, 1.2),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(0.0, 0.0),
    )
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=25,
        step_dt=0.02,
    )
    command_term.vel_command_b = torch.zeros((1, 3))
    command_term.heading_target = torch.zeros(1)
    command_term.is_heading_env = torch.zeros(1, dtype=torch.bool)
    command_term.is_standing_env = torch.zeros(1, dtype=torch.bool)
    command_term.gait_periods = torch.full((1,), 0.5)
    command_term.duty_factors = torch.full((1,), 0.45)
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term)

    phase_before = command_term.common_gait_phases().clone()
    command_term._resample_command([0])
    command_term._update_velocity_resampled_gait_timing(torch.tensor([0]), torch.empty(0, dtype=torch.long))

    expected_period = command_term._compute_period_from_forward_velocity(torch.tensor([0.0]))
    expected_duty_factor = command_term._compute_duty_factor_from_forward_velocity(torch.tensor([0.0]))
    assert torch.count_nonzero(command_term.vel_command_b) == 0
    assert torch.allclose(command_term.gait_periods, expected_period)
    assert torch.allclose(command_term.duty_factors, expected_duty_factor)
    _assert_phases_close(command_term.common_gait_phases(), phase_before)


def test_dimensionless_velocity_curve_is_sign_symmetric_and_has_correct_zero_speed_scale():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_period = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(device=torch.device("cpu"))
    velocities = torch.tensor((-1.2, 0.0, 1.2), dtype=torch.float64)

    periods = command_term._compute_period_from_forward_velocity(velocities)
    duty_factors = command_term._compute_duty_factor_from_forward_velocity(velocities)

    characteristic_length = sum(cfg.base_height_range) * 0.5
    assert periods[1].item() == pytest.approx(2.55 * math.sqrt(characteristic_length / 9.81))
    assert duty_factors[1].item() == pytest.approx(0.5588)
    assert periods[0].item() == pytest.approx(periods[2].item())
    assert duty_factors[0].item() == pytest.approx(duty_factors[2].item())


@pytest.mark.parametrize(
    "base_height_range",
    ((0.0, 0.4), (-0.4, 0.4), (0.5, 0.4), (float("nan"), 0.4), (0.4, float("inf"))),
)
def test_dimensionless_velocity_curve_rejects_invalid_characteristic_lengths(base_height_range):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_period = False
    cfg.base_height_range = base_height_range
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg

    with pytest.raises(ValueError, match="base_height_range"):
        command_term._compute_period_from_forward_velocity(torch.tensor([1.0]))


def test_gait_assignment_recomputes_timing_without_jumping_common_phase():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_theta = False
    cfg.add_noise_period = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=20,
        step_dt=0.02,
    )
    command_term.init_foot_thetas = _configured_gaits().to(dtype=torch.float32)
    command_term.foot_thetas = command_term.init_foot_thetas[[0]].clone()
    command_term.gait_periods = torch.full((1,), 0.42)
    command_term.duty_factors = torch.full((1,), 0.47)
    command_term.kappa = torch.full((1,), cfg.kappa)
    command_term.vel_command_b = torch.tensor(((1.8, 0.0, 0.0),))
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term, anchor_phase=0.17, anchor_step=3)
    common_phase_before = command_term.common_gait_phases().clone()
    command_term._assign_gait(torch.tensor([0]), torch.tensor([2]), add_theta_noise=False)
    phase_after = command_term.foot_phases()

    expected_period = command_term._compute_period_from_forward_velocity(torch.tensor([1.8]))
    expected_duty_factor = command_term._compute_duty_factor_from_forward_velocity(torch.tensor([1.8]))
    assert torch.allclose(command_term.gait_periods, expected_period)
    assert torch.allclose(command_term.duty_factors, expected_duty_factor)
    _assert_phases_close(command_term.common_gait_phases(), common_phase_before)
    expected_phase = symm_quadruped.compute_same_gait_foot_phases(
        common_phase_before.unsqueeze(-1), command_term.init_foot_thetas[[2]]
    )
    _assert_phases_close(phase_after, expected_phase, atol=1.0e-6)


def test_selective_gait_timing_update_leaves_other_environment_unchanged():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.add_noise_period = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        common_step_counter=20,
        step_dt=0.02,
    )
    command_term.vel_command_b = torch.tensor(((1.2, 0.0, 0.0), (1.8, 0.0, 0.0)))
    command_term.gait_periods = torch.tensor((0.5, 0.6))
    command_term.duty_factors = torch.tensor((0.45, 0.55))
    command_term.metrics = {"gait_period": torch.zeros(2), "duty_factor": torch.zeros(2)}
    _initialize_gait_clock(
        command_term,
        anchor_phase=torch.tensor((0.1, 0.7)),
        anchor_step=torch.tensor((4, 11)),
    )
    phase_before = command_term.common_gait_phases().clone()

    command_term._update_gait_timing([0])

    _assert_phases_close(command_term.common_gait_phases(), phase_before)
    assert command_term.gait_periods[1].item() == pytest.approx(0.6)
    assert command_term.duty_factors[1].item() == pytest.approx(0.55)


def test_velocity_timing_refresh_excludes_environments_already_refreshed_by_gait():
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term._env = SimpleNamespace(num_envs=4)
    timing_updates = []
    command_term._update_gait_timing = lambda env_ids: timing_updates.append(env_ids.clone())

    command_term._update_velocity_resampled_gait_timing(
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 3]),
    )

    assert len(timing_updates) == 1
    assert torch.equal(timing_updates[0], torch.tensor([0, 2]))


def test_once_after_reset_command_resamples_at_only_the_first_timer_expiry():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.resample_once_after_reset = True
    cfg.gait_sequence_enabled = False
    cfg.resampling_time_gait = 0.0
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term.time_left = torch.zeros(1)
    command_term.command_counter = torch.ones(1, dtype=torch.long)
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    resampled_env_ids = []
    timing_env_ids = []

    def resample(env_ids):
        resampled_env_ids.append(env_ids.clone())
        command_term.time_left[env_ids] = 10.0
        command_term.command_counter[env_ids] += 1

    command_term._resample = resample
    command_term._update_gait_timing = lambda env_ids: timing_env_ids.append(env_ids.clone())

    command_term.compute(0.02)
    command_term.compute(0.02)

    assert len(resampled_env_ids) == 1
    assert torch.equal(resampled_env_ids[0], torch.tensor([0]))
    assert len(timing_env_ids) == 1
    assert torch.equal(timing_env_ids[0], torch.tensor([0]))
    assert command_term.command_counter.item() == 2
    assert torch.isinf(command_term.time_left).all()


def test_post_reward_scheduling_skips_environments_that_are_about_to_reset():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.resample_once_after_reset = True
    cfg.gait_sequence_enabled = False
    cfg.resampling_time_gait = 0.0
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(reset_buf=torch.tensor((False, True)))
    command_term.time_left = torch.zeros(2)
    command_term.command_counter = torch.ones(2, dtype=torch.long)
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    resampled_env_ids = []

    def resample(env_ids):
        resampled_env_ids.append(env_ids.clone())
        command_term.command_counter[env_ids] += 1

    command_term._resample = resample
    command_term._update_gait_timing = lambda _: None

    command_term.compute(0.02)

    assert len(resampled_env_ids) == 1
    assert torch.equal(resampled_env_ids[0], torch.tensor([0]))
    assert command_term.command_counter.tolist() == [2, 1]
    assert torch.isinf(command_term.time_left[0])
    assert command_term.time_left[1].item() == pytest.approx(0.0)


def test_partial_reset_restarts_only_the_selected_environment_clock():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        common_step_counter=123,
        extras={},
    )
    command_term._common_gait_phase_at_anchor = torch.tensor((0.25, 0.75))
    command_term._gait_phase_anchor_steps = torch.tensor((50, 80), dtype=torch.long)
    command_term._error_xy_sum = torch.zeros(2)
    command_term._error_yaw_sum = torch.zeros(2)
    command_term._step_count = torch.ones(2)
    command_term.vel_command_b = torch.zeros((2, 3))
    command_term.command_counter = torch.ones(2, dtype=torch.long)
    command_term.gait_counter = torch.ones(2, dtype=torch.long)
    command_term.gait_sequence_indices = torch.zeros(2, dtype=torch.long)
    command_term.metrics = {
        "error_vel_xy": torch.zeros(2),
        "error_vel_yaw": torch.zeros(2),
        "success_threshold_vel_xy": torch.zeros(2),
        "success_threshold_vel_yaw": torch.zeros(2),
        "success_rate": torch.zeros(2),
        "gait_period": torch.zeros(2),
        "duty_factor": torch.zeros(2),
    }
    phases_seen_during_reset = []

    def resample_command(env_ids):
        phases_seen_during_reset.append(command_term._common_gait_phase_at_anchor[env_ids].clone())
        command_term.command_counter[env_ids] += 1

    def resample_gait(env_ids):
        phases_seen_during_reset.append(command_term._common_gait_phase_at_anchor[env_ids].clone())
        command_term.gait_counter[env_ids] += 1

    command_term._resample = resample_command
    command_term._resample_gait = resample_gait

    command_term.reset(torch.tensor([1]))

    assert command_term._common_gait_phase_at_anchor.tolist() == pytest.approx((0.25, 0.0))
    assert command_term._gait_phase_anchor_steps.tolist() == [50, 123]
    assert len(phases_seen_during_reset) == 2
    assert all(torch.equal(phase, torch.zeros(1)) for phase in phases_seen_during_reset)


def test_once_after_reset_gait_resamples_at_only_the_first_timer_expiry():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.gait_sequence_enabled = False
    cfg.resample_gait_once_after_reset = True
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term.time_left = torch.full((1,), torch.inf)
    command_term.command_counter = torch.ones(1, dtype=torch.long)
    command_term.gait_time_left = torch.zeros(1)
    command_term.gait_counter = torch.ones(1, dtype=torch.long)
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    resampled_env_ids = []

    def resample_gait(env_ids):
        resampled_env_ids.append(env_ids.clone())
        command_term.gait_time_left[env_ids] = 20.0
        command_term.gait_counter[env_ids] += 1

    command_term._resample_gait = resample_gait

    command_term.compute(0.02)
    command_term.compute(0.02)

    assert len(resampled_env_ids) == 1
    assert torch.equal(resampled_env_ids[0], torch.tensor([0]))
    assert command_term.gait_counter.item() == 2
    assert torch.isinf(command_term.gait_time_left).all()


def test_training_one_shot_resampling_uses_ten_and_twenty_second_schedule():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.resampling_time_range = (10.0, 10.0)
    cfg.resample_once_after_reset = True
    cfg.resampling_time_gait = 20.0
    cfg.resample_gait_once_after_reset = True
    cfg.gait_sequence_enabled = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term.time_left = torch.full((1,), 10.0)
    command_term.command_counter = torch.ones(1, dtype=torch.long)
    command_term.gait_time_left = torch.full((1,), 20.0)
    command_term.gait_counter = torch.ones(1, dtype=torch.long)
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    events = []
    current_step = 0

    def resample_velocity(env_ids):
        events.append((current_step, "velocity"))
        command_term.time_left[env_ids] = 10.0
        command_term.command_counter[env_ids] += 1

    def resample_gait(env_ids):
        events.append((current_step, "gait"))
        command_term.gait_time_left[env_ids] = 20.0
        command_term.gait_counter[env_ids] += 1
        command_term._update_gait_timing(env_ids)

    def update_timing(env_ids):
        events.append((current_step, "timing"))

    command_term._resample = resample_velocity
    command_term._resample_gait = resample_gait
    command_term._update_gait_timing = update_timing

    for current_step in range(1, 1501):
        command_term.compute(0.02)

    assert events == [(500, "velocity"), (500, "timing"), (1000, "gait"), (1000, "timing")]


def test_training_boundaries_refresh_timing_with_velocity_then_gait(monkeypatch):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.resampling_time_range = (10.0, 10.0)
    cfg.resample_once_after_reset = True
    cfg.resampling_time_gait = 20.0
    cfg.resample_gait_once_after_reset = True
    cfg.gait_sequence_enabled = False
    cfg.add_noise_period = False
    cfg.add_noise_theta = False
    cfg.heading_command = False
    cfg.rel_standing_envs = 0.0
    cfg.ranges = cfg.Ranges(
        lin_vel_x=(1.2, 1.2),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(0.0, 0.0),
    )
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        step_dt=0.02,
    )
    command_term.time_left = torch.full((1,), 10.0)
    command_term.command_counter = torch.ones(1, dtype=torch.long)
    command_term.gait_time_left = torch.full((1,), 20.0)
    command_term.gait_counter = torch.ones(1, dtype=torch.long)
    command_term.vel_command_b = torch.tensor(((0.4, 0.0, 0.0),))
    command_term.heading_target = torch.zeros(1)
    command_term.is_heading_env = torch.zeros(1, dtype=torch.bool)
    command_term.is_standing_env = torch.zeros(1, dtype=torch.bool)
    command_term.gait_periods = torch.full((1,), 0.5)
    command_term.duty_factors = torch.full((1,), 0.45)
    command_term.init_foot_thetas = _configured_gaits().to(dtype=torch.float32)
    command_term.foot_theta_sampling_weights = None
    command_term.foot_thetas = command_term.init_foot_thetas[[0]].clone()
    command_term.kappa = torch.full((1,), cfg.kappa)
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    _initialize_gait_clock(command_term, anchor_phase=0.17)

    def sample_third_gait(*, low, high, size, device):
        assert (low, high, size) == (0, len(command_term.init_foot_thetas), (1,))
        return torch.full(size, 2, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", sample_third_gait)

    command_term._env.common_step_counter = 500
    command_term._env.episode_length_buf.fill_(500)
    phase_before_velocity = command_term.foot_phases().clone()
    theta_before_velocity = command_term.foot_thetas.clone()
    command_term.compute(10.0)

    assert command_term.vel_command_b[0, 0].item() == pytest.approx(1.2)
    expected_period = command_term._compute_period_from_forward_velocity(torch.tensor([1.2]))
    expected_duty_factor = command_term._compute_duty_factor_from_forward_velocity(torch.tensor([1.2]))
    assert torch.allclose(command_term.gait_periods, expected_period)
    assert torch.allclose(command_term.duty_factors, expected_duty_factor)
    assert torch.equal(command_term.foot_thetas, theta_before_velocity)
    _assert_phases_close(command_term.foot_phases(), phase_before_velocity)

    command_term._env.common_step_counter = 1000
    command_term._env.episode_length_buf.fill_(1000)
    common_phase_before_gait = command_term.common_gait_phases().clone()
    command_term.compute(10.0)

    assert torch.allclose(command_term.gait_periods, expected_period)
    assert torch.allclose(command_term.duty_factors, expected_duty_factor)
    assert torch.equal(command_term.foot_thetas, command_term.init_foot_thetas[[2]])
    _assert_phases_close(command_term.common_gait_phases(), common_phase_before_gait)
    expected_foot_phases = symm_quadruped.compute_same_gait_foot_phases(
        torch.remainder(common_phase_before_gait, 1.0).unsqueeze(-1), command_term.init_foot_thetas[[2]]
    )
    _assert_phases_close(command_term.foot_phases(), expected_foot_phases)
    assert command_term.command_counter.item() == 2
    assert command_term.gait_counter.item() == 2
    assert torch.isinf(command_term.time_left).all()
    assert torch.isinf(command_term.gait_time_left).all()


def test_play_sequence_uses_one_shot_velocity_before_shared_gait_boundary():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.resampling_time_range = (10.0, 10.0)
    cfg.resample_once_after_reset = True
    cfg.gait_sequence_enabled = True
    cfg.gait_sequence_duration_s = 5.0
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=0,
        step_dt=0.02,
    )
    command_term.time_left = torch.full((1,), 10.0)
    command_term.command_counter = torch.ones(1, dtype=torch.long)
    command_term.init_foot_thetas = torch.tensor(
        symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_ROWS, dtype=torch.float32
    )
    command_term.foot_thetas = command_term.init_foot_thetas[[0]].clone()
    command_term.gait_sequence_indices = torch.zeros(1, dtype=torch.long)
    command_term.gait_time_left = torch.full((1,), torch.inf)
    command_term._update_metrics = lambda: None
    command_term._update_command = lambda: None
    events = []
    current_step = 0

    def resample_velocity(env_ids):
        events.append((current_step, "velocity"))
        command_term.time_left[env_ids] = 10.0
        command_term.command_counter[env_ids] += 1

    def assign_gait(env_ids, choices, *, add_theta_noise):
        assert add_theta_noise is False
        events.append((current_step, "gait"))
        command_term.foot_thetas[env_ids] = command_term.init_foot_thetas[choices]
        command_term._update_gait_timing(env_ids)

    def update_timing(env_ids):
        events.append((current_step, "timing"))

    command_term._resample = resample_velocity
    command_term._assign_gait = assign_gait
    command_term._update_gait_timing = update_timing

    for current_step in range(1, 1001):
        command_term._env.common_step_counter = current_step
        command_term.compute(0.02)

    assert events == [
        (250, "gait"),
        (250, "timing"),
        (500, "velocity"),
        (500, "gait"),
        (500, "timing"),
        (750, "gait"),
        (750, "timing"),
        (1000, "gait"),
        (1000, "timing"),
    ]


def test_zero_command_remains_zero_and_observation_transform_remains_involutive():
    obs, _, _, _ = _make_valid_observations((8,), seed=5)
    obs[..., _LAYOUT.desired_base_twist] = 0.0

    obs_tr = symm_quadruped.time_reverse_observations(obs)
    obs_tt = symm_quadruped.time_reverse_observations(obs_tr)

    assert torch.count_nonzero(obs_tr[..., _LAYOUT.desired_base_twist]) == 0
    assert torch.allclose(obs_tt, obs, atol=1.0e-11, rtol=0.0)


def test_half_bound_rows_have_required_geometry_and_are_negative_offset_partners():
    gaits = _configured_gaits()
    half_bound_1 = gaits[2]
    half_bound_2 = gaits[3]

    assert half_bound_1.tolist() == pytest.approx((0.13, -0.13, 0.5, 0.5))
    assert half_bound_2.tolist() == pytest.approx((-0.13, 0.13, 0.5, 0.5))
    assert half_bound_1[2] == half_bound_1[3]
    assert half_bound_2[2] == half_bound_2[3]
    front_separation = torch.abs(half_bound_1[0] - half_bound_1[1])
    front_circular_separation = torch.minimum(front_separation, 1.0 - front_separation)
    assert front_circular_separation.item() == pytest.approx(0.26)
    _assert_phases_close(_wrap(-half_bound_1), _wrap(half_bound_2))
    _assert_phases_close(_wrap(-half_bound_2), _wrap(half_bound_1))


def test_same_gait_backward_keeps_each_half_bound_row_while_physical_tr_exchanges_them():
    half_bounds = _configured_gaits()[2:4]
    phi = torch.tensor([[0.21], [0.21]], dtype=torch.float64)
    phases = symm_quadruped.compute_same_gait_foot_phases(phi, half_bounds)
    _assert_phases_close(_wrap(phases - phi), _wrap(half_bounds))

    beta = torch.full((2, 1), 0.45, dtype=torch.float64)
    swing_ratio = 1.0 - beta
    obs = torch.zeros((2, _OBS_DIM), dtype=torch.float64)
    obs[..., _LAYOUT.foot_phase_sin], obs[..., _LAYOUT.foot_phase_cos] = _encode_phase(phases)
    obs[..., _LAYOUT.foot_theta_sin], obs[..., _LAYOUT.foot_theta_cos] = _encode_phase(half_bounds)
    obs[..., _LAYOUT.phase_ratios] = torch.cat((swing_ratio, beta), dim=-1)

    obs_tr = symm_quadruped.time_reverse_observations(obs)
    phase_tr = _decode_phase(
        obs_tr[..., _LAYOUT.foot_phase_sin],
        obs_tr[..., _LAYOUT.foot_phase_cos],
    )
    theta_tr = _decode_phase(
        obs_tr[..., _LAYOUT.foot_theta_sin],
        obs_tr[..., _LAYOUT.foot_theta_cos],
    )
    phi_tr = _wrap(swing_ratio - phi)
    _assert_phases_close(phase_tr, _wrap(phi_tr + half_bounds.flip(0)))
    _assert_phases_close(theta_tr, _wrap(half_bounds.flip(0)))


def test_half_bound_touchdown_order_reverses_between_partner_rows():
    half_bound_1, half_bound_2 = _configured_gaits()[2:4]
    swing_ratio = torch.tensor(0.55, dtype=torch.float64)

    def anchored_touchdown_times(theta: torch.Tensor) -> torch.Tensor:
        touchdown_phase = _wrap(swing_ratio - theta)
        return _wrap(touchdown_phase - touchdown_phase[2])

    h1_events = anchored_touchdown_times(half_bound_1)
    h2_events = anchored_touchdown_times(half_bound_2)

    assert torch.allclose(h1_events[2:], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(h2_events[2:], torch.zeros(2, dtype=torch.float64))
    assert 0.0 < h1_events[0] < h1_events[1] < 1.0
    assert 0.0 < h2_events[1] < h2_events[0] < 1.0


def test_current_gait_library_time_reversal_closure_is_explicit():
    gaits = _wrap(_configured_gaits())
    partner_indices = (0, 1, 3, 2, 5, 4, 8, 9, 6, 7)
    expected_gaits = (
        (0.0, 0.5, 0.5, 0.0),
        (0.0, 0.0, 0.5, 0.5),
        (0.13, -0.13, 0.5, 0.5),
        (-0.13, 0.13, 0.5, 0.5),
        (0.0, 0.0, 0.63, 0.37),
        (0.0, 0.0, 0.37, 0.63),
        (-0.13, 0.13, 0.63, 0.37),
        (0.13, -0.13, 0.63, 0.37),
        (0.13, -0.13, 0.37, 0.63),
        (-0.13, 0.13, 0.37, 0.63),
    )

    assert torch.allclose(
        _configured_gaits(),
        torch.tensor(expected_gaits, dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    for gait_index, partner_index in enumerate(partner_indices):
        _assert_phases_close(_wrap(-gaits[gait_index]), gaits[partner_index])


def test_training_gait_distribution_is_time_reversal_invariant_and_family_balanced():
    weights = torch.tensor(symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS, dtype=torch.float64)
    probabilities = weights / weights.sum()

    assert probabilities.tolist() == pytest.approx((0.25, 0.25, *(0.0625,) * 8))
    assert probabilities.tolist() == pytest.approx(
        tuple(probabilities[index] for index in (0, 1, 3, 2, 5, 4, 8, 9, 6, 7))
    )
    family_probabilities = torch.stack(
        (probabilities[0], probabilities[1], probabilities[2:6].sum(), probabilities[6:].sum())
    )
    assert family_probabilities.tolist() == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_leg_permutation_penalty_ignores_non_synchronous_pairs():
    joint_ranges = torch.tensor(symm_quadruped.SYMM_QUADRUPED_JOINT_RANGES)
    joint_pos = torch.zeros((2, 12))
    joint_pos[:, :3] = 0.1 * joint_ranges
    foot_thetas = torch.tensor(
        (
            (0.0, 0.02, 0.3, 0.4),
            (0.0, 0.020001, 0.3, 0.4),
        )
    )

    penalty = _leg_permutation_penalty(
        foot_thetas,
        joint_pos,
        leg_pairs=(("FL", "FR"),),
    )

    assert penalty[0] < 0.0
    assert penalty[1] == 0.0


def test_leg_permutation_penalty_returns_zero_without_configured_pairs():
    penalty = _leg_permutation_penalty(
        torch.zeros((2, 4)),
        torch.ones((2, 12)),
        leg_pairs=(),
    )

    assert torch.equal(penalty, torch.zeros_like(penalty))


def test_leg_permutation_penalty_preserves_joint_tensor_dtype():
    penalty = _leg_permutation_penalty(
        torch.zeros((1, 4), dtype=torch.float64),
        torch.ones((1, 12), dtype=torch.float64),
        leg_pairs=(("FL", "FR"),),
    )

    assert penalty.dtype == torch.float64


def test_leg_permutation_penalty_wraps_synchronous_offsets_in_cycle_units():
    joint_ranges = torch.tensor(symm_quadruped.SYMM_QUADRUPED_JOINT_RANGES)
    joint_pos = torch.zeros((2, 12))
    joint_pos[:, :3] = 0.1 * joint_ranges
    foot_thetas = torch.tensor(
        (
            (0.999, 0.001, 0.3, 0.4),
            (-0.001, 0.001, 0.3, 0.4),
        )
    )

    penalty = _leg_permutation_penalty(
        foot_thetas,
        joint_pos,
        leg_pairs=(("FL", "FR"),),
    )

    assert penalty[0] < 0.0
    assert torch.allclose(penalty[0], penalty[1], atol=1.0e-7, rtol=0.0)


def test_leg_permutation_penalty_normalizes_by_synchronous_pair_count():
    normalized_error = 0.1
    joint_ranges = torch.tensor(symm_quadruped.SYMM_QUADRUPED_JOINT_RANGES)
    joint_pos = torch.zeros((2, 12))
    joint_pos[:, :3] = normalized_error * joint_ranges
    joint_pos[:, 6:9] = normalized_error * joint_ranges
    foot_thetas = torch.tensor(
        (
            (0.13, -0.13, 0.5, 0.5),
            (0.0, 0.0, 0.5, 0.5),
        )
    )

    penalty = _leg_permutation_penalty(
        foot_thetas,
        joint_pos,
        leg_pairs=(("FL", "FR"), ("RL", "RR")),
    )

    expected = -(1.0 - torch.exp(torch.tensor(-5.0 * normalized_error)))
    assert torch.allclose(penalty, expected.expand_as(penalty), atol=1.0e-6, rtol=0.0)


def test_leg_permutation_penalty_shapes_the_mean_of_different_active_pair_errors():
    joint_ranges = torch.tensor(symm_quadruped.SYMM_QUADRUPED_JOINT_RANGES)
    joint_pos = torch.zeros((1, 12))
    joint_pos[:, :3] = 0.1 * joint_ranges
    joint_pos[:, 6:9] = 0.3 * joint_ranges

    penalty = _leg_permutation_penalty(
        torch.zeros((1, 4)),
        joint_pos,
        leg_pairs=(("FL", "FR"), ("RL", "RR")),
    )

    expected = -(1.0 - torch.exp(torch.tensor(-5.0 * 0.2)))
    assert penalty.item() == pytest.approx(expected.item(), abs=1.0e-6)


def test_foot_phase_penalty_sums_over_feet():
    foot_speeds = torch.ones((1, 4, 3))
    gait_command = SimpleNamespace(
        periodic_force_weights=lambda: -torch.ones((1, 4)),
        periodic_speed_weights=lambda: -torch.ones((1, 4)),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=SimpleNamespace(torch=foot_speeds)))},
        command_manager=SimpleNamespace(get_term=lambda _: gait_command),
    )

    penalty = symm_quadruped.foot_phase_penalty(
        env,
        command_name="base_velocity",
        feet_cfg=SimpleNamespace(body_ids=[0, 1, 2, 3]),
    )

    expected_per_foot_penalty = -(1.0 - math.exp(-2.0 * math.sqrt(3.0)))
    assert penalty.item() == pytest.approx(4.0 * expected_per_foot_penalty)


def test_hip_action_penalty_is_already_normalized_over_hip_actions():
    actions = torch.zeros((2, 12))
    actions[1, symm_quadruped.SYMM_QUADRUPED_HIP_ACTION_IDS] = 0.4
    env = SimpleNamespace(action_manager=SimpleNamespace(action=actions))

    penalty = symm_quadruped.hip_action_penalty(env)

    assert penalty[0].item() == 0.0
    assert penalty[1].item() == pytest.approx(-(1.0 - math.exp(-0.5 * 0.4)))
    assert torch.all((penalty >= -1.0) & (penalty <= 0.0))


@pytest.mark.parametrize("phase_sync_tolerance", (-0.001, 0.501, float("nan"), float("inf")))
def test_leg_permutation_penalty_rejects_invalid_phase_tolerance(phase_sync_tolerance):
    with pytest.raises(ValueError, match="phase_sync_tolerance"):
        _leg_permutation_penalty(
            torch.zeros((1, 4)),
            torch.zeros((1, 12)),
            leg_pairs=(),
            phase_sync_tolerance=phase_sync_tolerance,
        )


def test_default_theta_noise_cannot_change_synchronous_pair_classification():
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    noise_support = torch.arange(-cfg.noise_level_theta, cfg.noise_level_theta) * cfg.noise_scale_theta
    pairs = symm_quadruped.SYMM_QUADRUPED_LEG_PAIRS
    phase_index = symm_quadruped.SYMM_QUADRUPED_LEG_PHASE_INDEX
    gaits = _configured_gaits()

    assert cfg.add_noise_theta is True
    assert noise_support.tolist() == pytest.approx((-0.002, -0.001, 0.0, 0.001))

    nominal_active_counts = []
    for gait in gaits:
        active_count = 0
        for tag_a, tag_b in pairs:
            phase_a = gait[phase_index[tag_a]]
            phase_b = gait[phase_index[tag_b]]
            nominal_distance = torch.abs(torch.remainder(phase_a - phase_b + 0.5, 1.0) - 0.5)
            nominal_is_active = nominal_distance <= 0.02
            active_count += int(nominal_is_active)
            for noise_a in noise_support:
                for noise_b in noise_support:
                    distance = torch.abs(torch.remainder(phase_a + noise_a - phase_b - noise_b + 0.5, 1.0) - 0.5)
                    assert bool(distance <= 0.02) is bool(nominal_is_active)
        nominal_active_counts.append(active_count)

    assert nominal_active_counts == [2, 2, 1, 1, 1, 1, 0, 0, 0, 0]


def test_gait_resampling_adds_configured_phase_noise(monkeypatch):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.calculate_from_sampling_curve = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=0,
        step_dt=0.02,
    )
    command_term.init_foot_thetas = _configured_gaits().to(dtype=torch.float32)
    command_term.foot_theta_sampling_weights = torch.tensor(
        symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS, dtype=torch.float32
    )
    command_term.foot_theta_sampling_weights /= command_term.foot_theta_sampling_weights.sum()
    command_term.foot_thetas = torch.zeros((1, 4))
    command_term.gait_periods = torch.zeros(1)
    command_term.duty_factors = torch.zeros(1)
    command_term.kappa = torch.zeros(1)
    command_term.gait_time_left = torch.zeros(1)
    command_term.gait_counter = torch.zeros(1, dtype=torch.long)
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term)
    multinomial_calls = []
    randint_calls = []

    def fake_multinomial(input, num_samples, replacement):
        multinomial_calls.append((input.clone(), num_samples, replacement))
        return torch.zeros((num_samples,), dtype=torch.long, device=input.device)

    def fake_randint(*, low, high, size, device):
        randint_calls.append((low, high, size, device))
        return torch.tensor(((-2, -1, 0, 1),), dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    monkeypatch.setattr(torch, "randint", fake_randint)
    command_term._resample_gait([0])

    assert len(multinomial_calls) == 1
    sampled_weights, num_samples, replacement = multinomial_calls[0]
    assert sampled_weights.tolist() == pytest.approx((0.25, 0.25, *(0.0625,) * 8))
    assert (num_samples, replacement) == (1, True)
    assert randint_calls == [(-2, 2, (1, 4), torch.device("cpu"))]
    expected = command_term.init_foot_thetas[0] + torch.tensor((-0.002, -0.001, 0.0, 0.001))
    assert torch.allclose(command_term.foot_thetas[0], expected)


def test_gait_resampling_without_weights_preserves_uniform_sampling(monkeypatch):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.calculate_from_sampling_curve = False
    cfg.add_noise_theta = False
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        common_step_counter=0,
        step_dt=0.02,
    )
    command_term.init_foot_thetas = _configured_gaits().to(dtype=torch.float32)
    command_term.foot_theta_sampling_weights = None
    command_term.foot_thetas = torch.zeros((1, 4))
    command_term.gait_periods = torch.zeros(1)
    command_term.duty_factors = torch.zeros(1)
    command_term.kappa = torch.zeros(1)
    command_term.gait_time_left = torch.zeros(1)
    command_term.gait_counter = torch.zeros(1, dtype=torch.long)
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term)
    randint_calls = []

    def fake_randint(*, low, high, size, device):
        randint_calls.append((low, high, size, device))
        return torch.zeros(size, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: pytest.fail("uniform sampling used weights"))
    monkeypatch.setattr(torch, "randint", fake_randint)
    command_term._resample_gait([0])

    assert randint_calls == [(0, 10, (1,), torch.device("cpu"))]
    assert torch.equal(command_term.foot_thetas[0], command_term.init_foot_thetas[0])


def test_fixed_gait_sequence_uses_all_rows_in_order_for_five_seconds(monkeypatch):
    cfg = symm_quadruped.GaitVelocityCommandCfg()
    cfg.calculate_from_sampling_curve = False
    cfg.gait_period = 0.42
    cfg.duty_factor = 0.47
    cfg.gait_sequence_enabled = True
    cfg.gait_sequence_duration_s = 5.0
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    command_term.cfg = cfg
    command_term._env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=0,
    )
    command_term.init_foot_thetas = torch.tensor(
        symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_ROWS, dtype=torch.float32
    )
    command_term.foot_theta_sampling_weights = None
    command_term.foot_thetas = torch.zeros((1, 4))
    command_term.gait_periods = torch.full((1,), 0.42)
    command_term.duty_factors = torch.full((1,), 0.47)
    command_term.kappa = torch.zeros(1)
    command_term.gait_time_left = torch.zeros(1)
    command_term.gait_counter = torch.zeros(1, dtype=torch.long)
    command_term.gait_sequence_indices = torch.full((1,), -1, dtype=torch.long)
    command_term.metrics = {"gait_period": torch.zeros(1), "duty_factor": torch.zeros(1)}
    _initialize_gait_clock(command_term)

    expected_gaits = torch.tensor(symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_ROWS, dtype=torch.float32)
    assert torch.equal(command_term.init_foot_thetas, expected_gaits)

    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: pytest.fail("fixed sequence sampled randomly"))
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: pytest.fail("fixed sequence sampled randomly"))

    boundaries = (
        (0, 0),
        (249, 0),
        (250, 1),
        (499, 1),
        (500, 2),
        (749, 2),
        (750, 3),
        (999, 3),
        (1000, 4),
        (1249, 4),
        (1250, 5),
        (1499, 5),
        (1500, 0),
    )
    for common_step_counter, expected_gait_index in boundaries:
        command_term._env.common_step_counter = common_step_counter
        command_term._update_gait_sequence([0])
        assert command_term.gait_sequence_indices.item() == expected_gait_index
        assert torch.equal(command_term.foot_thetas[0], command_term.init_foot_thetas[expected_gait_index])
        assert command_term.gait_periods.item() == pytest.approx(0.42)
        assert command_term.duty_factors.item() == pytest.approx(0.47)

    command_term._env.common_step_counter = 2231
    command_term.foot_thetas.fill_(9.0)
    command_term._resample_gait([0])
    assert command_term.gait_sequence_indices.item() == 2
    assert torch.equal(command_term.foot_thetas[0], command_term.init_foot_thetas[2])

    for invalid_duration_s in (0.0, -1.0, 0.001, float("nan"), float("inf")):
        cfg.gait_sequence_duration_s = invalid_duration_s
        with pytest.raises(ValueError, match="gait_sequence_duration_s"):
            command_term._update_gait_sequence([0], force=True)
