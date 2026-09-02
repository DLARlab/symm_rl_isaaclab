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
    beta = 0.2 + 0.6 * torch.rand(batch_size, 1, generator=generator)

    obs[..., _LAYOUT.foot_phase_sin] = torch.sin(2.0 * torch.pi * phase)
    obs[..., _LAYOUT.foot_phase_cos] = torch.cos(2.0 * torch.pi * phase)
    obs[..., _LAYOUT.gait_period] = 1.5
    obs[..., _LAYOUT.duty_factor] = beta
    return obs


def test_time_reverse_observations_is_duty_aware_involution_with_expected_parities():
    obs = _make_valid_observations(3)
    obs_tr = go2_symm.time_reverse_observations(obs)
    obs_tt = go2_symm.time_reverse_observations(obs_tr)

    assert torch.allclose(obs_tt, obs, atol=1.0e-6, rtol=0.0)
    for odd_slice in (
        _LAYOUT.velocity_command,
        _LAYOUT.joint_velocity,
    ):
        assert torch.equal(obs_tr[..., odd_slice], -obs[..., odd_slice])
    for even_slice in (
        _LAYOUT.projected_gravity,
        _LAYOUT.joint_position,
        _LAYOUT.previous_action,
        _LAYOUT.second_previous_action,
        _LAYOUT.gait_period,
        _LAYOUT.duty_factor,
    ):
        assert torch.equal(obs_tr[..., even_slice], obs[..., even_slice])

    alpha = 2.0 * torch.pi * (1.0 - obs[..., _LAYOUT.duty_factor])
    expected_phase_sin = (
        torch.sin(alpha) * obs[..., _LAYOUT.foot_phase_cos] - torch.cos(alpha) * obs[..., _LAYOUT.foot_phase_sin]
    )
    expected_phase_cos = (
        torch.cos(alpha) * obs[..., _LAYOUT.foot_phase_cos] + torch.sin(alpha) * obs[..., _LAYOUT.foot_phase_sin]
    )
    assert torch.allclose(obs_tr[..., _LAYOUT.foot_phase_sin], expected_phase_sin)
    assert torch.allclose(obs_tr[..., _LAYOUT.foot_phase_cos], expected_phase_cos)


def test_time_reverse_term_major_history_transforms_every_frame_without_reordering():
    frames = _make_valid_observations(6).reshape(2, 3, _OBS_DIM)
    frames[:, 0, _LAYOUT.velocity_command] = 1.0
    frames[:, 1, _LAYOUT.velocity_command] = 2.0
    frames[:, 2, _LAYOUT.velocity_command] = 3.0
    packed = go2_symm.pack_term_major_policy_history(frames)

    transformed = go2_symm.time_reverse_observations(packed)
    transformed_frames = go2_symm.unpack_term_major_policy_history(transformed)

    assert torch.equal(transformed_frames[..., _LAYOUT.velocity_command], -frames[..., _LAYOUT.velocity_command])
    assert torch.equal(transformed_frames[..., _LAYOUT.previous_action], frames[..., _LAYOUT.previous_action])
    assert torch.allclose(go2_symm.time_reverse_observations(transformed), packed, atol=1.0e-6, rtol=0.0)


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
