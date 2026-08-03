# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared symmetric quadruped gait and time-reversal phase mappings."""

from __future__ import annotations

import inspect
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


def test_environment_foot_phases_use_same_episode_clock_for_every_command_sign():
    command_term = object.__new__(symm_quadruped.GaitVelocityCommand)
    commands = torch.tensor([-2.0, -1.0e-3, 0.0, 1.0e-3, 2.0], dtype=torch.float64)
    command_term._env = SimpleNamespace(
        episode_length_buf=torch.full((len(commands),), 37, dtype=torch.long),
        step_dt=0.02,
    )
    command_term.gait_periods = torch.full((len(commands),), 0.5, dtype=torch.float64)
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

    _assert_phases_close(_wrap(-gaits[0]), gaits[0])
    _assert_phases_close(_wrap(-gaits[1]), gaits[1])
    _assert_phases_close(_wrap(-gaits[2]), gaits[3])
    _assert_phases_close(_wrap(-gaits[3]), gaits[2])
    for gallop in gaits[4:]:
        partner = _wrap(-gallop)
        assert not any(torch.allclose(partner, gait, atol=1.0e-12, rtol=0.0) for gait in gaits)
