# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the migrated Go2 time-reversal transform."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm

_LAYOUT = go2_symm.SYMM_QUADRUPED_POLICY_OBS_LAYOUT
_OBS_DIM = go2_symm.SYMM_QUADRUPED_POLICY_OBS_DIM


def _make_valid_observations(batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    obs = torch.randn(batch_size, _OBS_DIM, generator=generator)
    phase = torch.rand(batch_size, 4, generator=generator)
    theta = torch.rand(batch_size, 4, generator=generator)
    beta = 0.2 + 0.6 * torch.rand(batch_size, 1, generator=generator)

    obs[..., _LAYOUT.foot_phase_sin] = torch.sin(2.0 * torch.pi * phase)
    obs[..., _LAYOUT.foot_phase_cos] = torch.cos(2.0 * torch.pi * phase)
    obs[..., _LAYOUT.foot_theta_sin] = torch.sin(2.0 * torch.pi * theta)
    obs[..., _LAYOUT.foot_theta_cos] = torch.cos(2.0 * torch.pi * theta)
    obs[..., _LAYOUT.phase_ratios] = torch.cat((1.0 - beta, beta), dim=-1)
    return obs


def test_time_reverse_observations_is_duty_aware_involution_with_expected_parities():
    obs = _make_valid_observations(3)
    obs_tr = go2_symm.time_reverse_observations(obs)
    obs_tt = go2_symm.time_reverse_observations(obs_tr)

    assert torch.allclose(obs_tt, obs, atol=1.0e-6, rtol=0.0)
    for odd_slice in (
        _LAYOUT.measured_base_twist,
        _LAYOUT.desired_base_twist,
        _LAYOUT.joint_velocity,
        _LAYOUT.foot_theta_sin,
    ):
        assert torch.equal(obs_tr[..., odd_slice], -obs[..., odd_slice])
    for even_slice in (
        _LAYOUT.projected_gravity,
        _LAYOUT.joint_position,
        _LAYOUT.previous_action,
        _LAYOUT.foot_theta_cos,
        _LAYOUT.phase_ratios,
        _LAYOUT.sagittal_plane_state,
    ):
        assert torch.equal(obs_tr[..., even_slice], obs[..., even_slice])

    alpha = 2.0 * torch.pi * obs[..., _LAYOUT.swing_ratio]
    expected_phase_sin = (
        torch.sin(alpha) * obs[..., _LAYOUT.foot_phase_cos] - torch.cos(alpha) * obs[..., _LAYOUT.foot_phase_sin]
    )
    expected_phase_cos = (
        torch.cos(alpha) * obs[..., _LAYOUT.foot_phase_cos] + torch.sin(alpha) * obs[..., _LAYOUT.foot_phase_sin]
    )
    assert torch.allclose(obs_tr[..., _LAYOUT.foot_phase_sin], expected_phase_sin)
    assert torch.allclose(obs_tr[..., _LAYOUT.foot_phase_cos], expected_phase_cos)


def test_time_reverse_actions_is_identity_involution():
    actions = torch.randn(5, 12)

    actions_tr = go2_symm.time_reverse_actions(actions)
    actions_tt = go2_symm.time_reverse_actions(actions_tr)

    assert torch.allclose(actions_tr, actions)
    assert torch.allclose(actions_tt, actions)


def test_compute_time_reversal_states_augments_observations_and_actions():
    batch_size = 4
    obs = TensorDict({"policy": _make_valid_observations(batch_size)}, batch_size=[batch_size])
    actions = torch.randn(batch_size, 12)

    obs_aug, actions_aug = go2_symm.compute_time_reversal_states(env=None, obs=obs, actions=actions)

    assert obs_aug.batch_size == torch.Size([2 * batch_size])
    assert actions_aug.shape == (2 * batch_size, 12)
    assert torch.allclose(obs_aug["policy"][:batch_size], obs["policy"])
    assert torch.allclose(obs_aug["policy"][batch_size:], go2_symm.time_reverse_observations(obs["policy"]))
    assert torch.allclose(actions_aug[:batch_size], actions)
    assert torch.allclose(actions_aug[batch_size:], go2_symm.time_reverse_actions(actions))
