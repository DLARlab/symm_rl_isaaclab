# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the migrated Go2 time-reversal transform."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm


def test_time_reverse_observations_is_involution_with_expected_parities():
    obs = torch.randn(3, 72)

    obs_tt = go2_symm.time_reverse_observations(go2_symm.time_reverse_observations(obs))
    assert torch.allclose(obs_tt, obs)

    parity = torch.ones(72)
    parity[0:6] = -1.0
    parity[9:15] = -1.0
    parity[27:39] = -1.0
    parity[51:55] = -1.0
    parity[59:63] = -1.0

    obs_tr = go2_symm.time_reverse_observations(obs)
    assert torch.allclose(obs_tr, obs * parity)


def test_time_reverse_actions_is_identity_involution():
    actions = torch.randn(5, 12)

    actions_tr = go2_symm.time_reverse_actions(actions)
    actions_tt = go2_symm.time_reverse_actions(actions_tr)

    assert torch.allclose(actions_tr, actions)
    assert torch.allclose(actions_tt, actions)


def test_compute_time_reversal_states_augments_observations_and_actions():
    batch_size = 4
    obs = TensorDict({"policy": torch.randn(batch_size, 72)}, batch_size=[batch_size])
    actions = torch.randn(batch_size, 12)

    obs_aug, actions_aug = go2_symm.compute_time_reversal_states(env=None, obs=obs, actions=actions)

    assert obs_aug.batch_size == torch.Size([2 * batch_size])
    assert actions_aug.shape == (2 * batch_size, 12)
    assert torch.allclose(obs_aug["policy"][:batch_size], obs["policy"])
    assert torch.allclose(obs_aug["policy"][batch_size:], go2_symm.time_reverse_observations(obs["policy"]))
    assert torch.allclose(actions_aug[:batch_size], actions)
    assert torch.allclose(actions_aug[batch_size:], go2_symm.time_reverse_actions(actions))
