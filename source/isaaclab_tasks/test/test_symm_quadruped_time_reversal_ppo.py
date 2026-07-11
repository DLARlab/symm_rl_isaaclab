# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped time-reversal PPO."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo import TimeReversalPPO


class _DummyDistribution:
    def __init__(self):
        self.std_param = torch.nn.Parameter(torch.tensor([-1.0, float("nan"), float("inf"), 0.5]))


class _DummyActor:
    def __init__(self):
        self.distribution = _DummyDistribution()


def test_clamp_actor_std_keeps_scalar_gaussian_std_positive_and_finite():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.actor = _DummyActor()

    algorithm._clamp_actor_std()

    std = algorithm.actor.distribution.std_param
    assert torch.all(torch.isfinite(std))
    assert torch.all(std >= algorithm._MIN_ACTOR_STD)
    assert torch.all(std <= algorithm._MAX_ACTOR_STD)
