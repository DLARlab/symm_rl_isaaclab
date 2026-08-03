# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped time-reversal PPO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo import TimeReversalPPO


class _DummyDistribution:
    def __init__(self):
        self.std_param = torch.nn.Parameter(torch.tensor([-1.0, float("nan"), float("inf"), 5.0]))


class _DummyActor:
    def __init__(self):
        self.distribution = _DummyDistribution()


def test_zeroed_no_trs_configuration_disables_time_reversal_update_path():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "use_time_reversal_regularization": True,
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
    }

    assert not algorithm._time_reversal_enabled()


def test_clamp_actor_std_keeps_scalar_gaussian_std_positive_and_finite():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.actor = _DummyActor()

    algorithm._clamp_actor_std()

    std = algorithm.actor.distribution.std_param
    assert torch.all(torch.isfinite(std))
    assert torch.all(std >= algorithm._MIN_ACTOR_STD)
    assert torch.all(std <= algorithm._MAX_ACTOR_STD)
    assert algorithm._MAX_ACTOR_STD == 1.0


def test_action_diagnostics_use_exact_rollout_samples_and_distribution_means():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.storage = SimpleNamespace(
        actions=torch.tensor([[[-2.0, 1.0]], [[0.0, 1.0]]]),
        distribution_params=(torch.tensor([[[-0.5, 0.25]], [[0.0, 0.25]]]),),
    )

    diagnostics = algorithm._action_diagnostics_from_storage()

    assert diagnostics["diagnostics/action_abs_mean"] == pytest.approx(1.0)
    assert diagnostics["diagnostics/action_abs_max"] == pytest.approx(2.0)
    assert diagnostics["diagnostics/actor_mean_abs_mean"] == pytest.approx(0.25)
    assert diagnostics["diagnostics/actor_mean_abs_max"] == pytest.approx(0.5)


def test_actor_mean_bound_loss_only_penalizes_extreme_policy_means():
    actor_mean = torch.tensor([[0.0, 10.0, 12.0, -14.0]])

    loss = TimeReversalPPO._actor_mean_bound_loss(actor_mean)

    assert loss.item() == pytest.approx(5.0)
    assert TimeReversalPPO._ACTOR_MEAN_BOUND == 10.0
    assert TimeReversalPPO._ACTOR_MEAN_BOUND_LOSS_COEFF == 1.0e-2


def test_actor_mean_safety_aborts_sustained_divergence_and_recovers_after_safe_update():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm._actor_mean_abort_count = 0

    for _ in range(algorithm._ACTOR_MEAN_ABORT_PATIENCE - 1):
        algorithm._update_actor_mean_safety(algorithm._ACTOR_MEAN_ABORT_BOUND + 1.0)

    with pytest.raises(RuntimeError, match="actor mean diverged"):
        algorithm._update_actor_mean_safety(algorithm._ACTOR_MEAN_ABORT_BOUND + 1.0)

    algorithm._update_actor_mean_safety(0.0)
    assert algorithm._actor_mean_abort_count == 0


def test_time_reversal_mask_includes_zero_velocity_commands():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "command_observation_index": 9,
        "command_observation_scale": 2.0,
        "min_abs_command_velocity": 0.0,
    }
    observations = {"policy": torch.zeros(3, 72)}

    mask = algorithm._time_reversal_mask(observations)

    assert torch.equal(mask, torch.ones(3, 1))
