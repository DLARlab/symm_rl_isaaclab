# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the migrated Go2 time-reversal transform."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm

FRAME_DIM = 56
FOOT_THETA_SLICE = slice(50, 54)


def _canonicalize_foot_theta(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta + 0.25, 1.0) - 0.25


def test_time_reverse_observations_is_involution_with_expected_parities():
    obs = torch.randn(3, FRAME_DIM)
    obs[:, FOOT_THETA_SLICE].uniform_(-0.24, 0.74)

    obs_tt = go2_symm.time_reverse_observations(go2_symm.time_reverse_observations(obs))
    assert torch.allclose(obs_tt, obs)

    expected = obs.clone()
    expected[:, 3:6] *= -1.0
    expected[:, 18:30] *= -1.0
    expected[:, FOOT_THETA_SLICE] = _canonicalize_foot_theta(-obs[:, FOOT_THETA_SLICE])
    obs_tr = go2_symm.time_reverse_observations(obs)
    assert torch.allclose(obs_tr, expected)


def test_time_reverse_observations_repeats_parity_for_frame_major_history():
    history_length = 20
    obs_frames = torch.randn(3, history_length, FRAME_DIM)
    obs_frames[..., FOOT_THETA_SLICE].uniform_(-0.24, 0.74)
    obs = obs_frames.reshape(3, -1)
    expected_frames = obs_frames.clone()
    expected_frames[..., 3:6] *= -1.0
    expected_frames[..., 18:30] *= -1.0
    expected_frames[..., FOOT_THETA_SLICE] = _canonicalize_foot_theta(-obs_frames[..., FOOT_THETA_SLICE])

    obs_tr = go2_symm.time_reverse_observations(obs)

    assert obs_tr.shape == (3, history_length * FRAME_DIM)
    assert torch.allclose(obs_tr.reshape(3, history_length, FRAME_DIM), expected_frames)
    assert torch.allclose(go2_symm.time_reverse_observations(obs_tr), obs)


def test_time_reverse_observations_rejects_incompatible_layout():
    with pytest.raises(ValueError, match="multiple of the 56D symmetric quadruped policy frame"):
        go2_symm.time_reverse_observations(torch.randn(3, 59))


def test_time_reverse_actions_is_identity_involution():
    actions = torch.randn(5, 12)

    actions_tr = go2_symm.time_reverse_actions(actions)
    actions_tt = go2_symm.time_reverse_actions(actions_tr)

    assert torch.allclose(actions_tr, actions)
    assert torch.allclose(actions_tt, actions)


def test_time_reverse_observation_history_reverses_frames_and_realigns_actions():
    batch_size = 2
    history_length = 3
    obs_frames = torch.arange(batch_size * history_length * FRAME_DIM, dtype=torch.float32).reshape(
        batch_size, history_length, FRAME_DIM
    )
    obs_frames[..., FOOT_THETA_SLICE] = 0.0
    previous_actions = torch.stack(
        [torch.full((batch_size, 12), float(action_index)) for action_index in range(history_length)],
        dim=1,
    )
    obs_frames[..., 30:42] = previous_actions
    obs = obs_frames.reshape(batch_size, -1)
    current_actions = torch.full((batch_size, 12), float(history_length))

    obs_tr, action_targets_tr = go2_symm.time_reverse_observation_history(obs, current_actions)

    expected_frames = go2_symm.time_reverse_observations(obs).reshape(batch_size, history_length, FRAME_DIM).flip(1)
    expected_frames[..., 30:42] = torch.stack(
        (current_actions, previous_actions[:, 2], previous_actions[:, 1]),
        dim=1,
    )
    assert torch.equal(obs_tr.reshape(batch_size, history_length, FRAME_DIM), expected_frames)
    assert torch.equal(action_targets_tr, previous_actions[:, 0])

    obs_tt, action_targets_tt = go2_symm.time_reverse_observation_history(obs_tr, action_targets_tr)
    assert torch.equal(obs_tt, obs)
    assert torch.equal(action_targets_tt, current_actions)


def test_compute_time_reversal_states_augments_observations_and_actions():
    batch_size = 4
    obs = TensorDict({"policy": torch.randn(batch_size, 20 * FRAME_DIM)}, batch_size=[batch_size])
    obs["policy"].reshape(batch_size, 20, FRAME_DIM)[..., FOOT_THETA_SLICE].uniform_(-0.24, 0.74)
    actions = torch.randn(batch_size, 12)

    obs_aug, actions_aug = go2_symm.compute_time_reversal_states(env=None, obs=obs, actions=actions)
    expected_obs_tr, expected_actions_tr = go2_symm.time_reverse_observation_history(obs["policy"], actions)

    assert obs_aug.batch_size == torch.Size([2 * batch_size])
    assert actions_aug.shape == (2 * batch_size, 12)
    assert torch.allclose(obs_aug["policy"][:batch_size], obs["policy"])
    assert torch.allclose(obs_aug["policy"][batch_size:], expected_obs_tr)
    assert torch.allclose(actions_aug[:batch_size], actions)
    assert torch.allclose(actions_aug[batch_size:], expected_actions_tr)


def test_compute_time_reversal_states_requires_current_actions_for_history():
    obs = TensorDict({"policy": torch.randn(2, 20 * FRAME_DIM)}, batch_size=[2])

    with pytest.raises(ValueError, match="current rollout actions are required"):
        go2_symm.compute_time_reversal_states(env=None, obs=obs, actions=None)
