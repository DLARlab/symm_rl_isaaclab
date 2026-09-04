# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped time-reversal PPO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo import TimeReversalPPO
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm

FRAME_DIM = 56


class _DummyDistribution:
    def __init__(self):
        self.std_param = torch.nn.Parameter(torch.tensor([-1.0, float("nan"), float("inf"), 5.0]))


class _DummyActor:
    def __init__(self):
        self.distribution = _DummyDistribution()


def test_clamp_actor_std_keeps_scalar_gaussian_std_positive_finite_and_unbounded_above():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.actor = _DummyActor()

    algorithm._clamp_actor_std()

    std = algorithm.actor.distribution.std_param
    assert torch.all(torch.isfinite(std))
    assert torch.all(std >= algorithm._MIN_ACTOR_STD)
    assert torch.equal(
        std,
        torch.tensor(
            [
                algorithm._MIN_ACTOR_STD,
                algorithm._MIN_ACTOR_STD,
                algorithm._NONFINITE_ACTOR_STD_FALLBACK,
                5.0,
            ]
        ),
    )


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
        "command_observation_index": 3,
        "command_observation_scale": 2.0,
        "observation_frame_dim": FRAME_DIM,
        "min_abs_command_velocity": 0.0,
    }
    observations = {"policy": torch.zeros(3, 20 * FRAME_DIM)}

    mask = algorithm._time_reversal_mask(observations)

    assert torch.equal(mask, torch.ones(3, 1))


def test_time_reversal_mask_uses_forward_command_from_newest_history_frame():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "command_observation_index": 3,
        "command_observation_scale": 2.0,
        "observation_frame_dim": FRAME_DIM,
        "min_abs_command_velocity": 0.2,
    }
    policy = torch.zeros(3, 20 * FRAME_DIM)
    policy[:, 3] = 10.0
    policy[:, -FRAME_DIM + 3] = torch.tensor([0.0, 0.4, -0.6])

    mask = algorithm._time_reversal_mask({"policy": policy})

    assert torch.equal(mask, torch.tensor([[0.0], [1.0], [1.0]]))


def test_time_reversal_mask_rejects_partial_observation_frame():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = {
        "command_observation_index": 3,
        "observation_frame_dim": FRAME_DIM,
    }

    with pytest.raises(ValueError, match="multiple of the configured frame dimension"):
        algorithm._time_reversal_mask({"policy": torch.zeros(1, 61)})


def _make_temporal_pair_algorithm(*, history_length: int = 3, num_steps: int = 5, num_envs: int = 2):
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.actor = SimpleNamespace(is_recurrent=False)
    algorithm.critic = SimpleNamespace(is_recurrent=False)
    algorithm.symmetry = {
        "_env": None,
        "data_augmentation_func": go2_symm.compute_time_reversal_states,
        "observation_frame_dim": FRAME_DIM,
        "command_observation_index": 3,
        "command_observation_scale": 2.0,
        "min_abs_command_velocity": 0.0,
    }
    policy = torch.ones(num_steps, num_envs, history_length * FRAME_DIM)
    policy_frames = policy.reshape(num_steps, num_envs, history_length, FRAME_DIM)
    policy_frames[..., 50:54] = 0.0
    for step in range(num_steps):
        for frame in range(history_length):
            policy_frames[step, :, frame, 30:42] = 100.0 * step + 10.0 * frame
    actions = torch.stack(
        [torch.full((num_envs, 12), 1000.0 + step) for step in range(num_steps)],
        dim=0,
    )
    algorithm.storage = SimpleNamespace(
        observations=TensorDict({"policy": policy}, batch_size=[num_steps, num_envs]),
        actions=actions,
        dones=torch.zeros(num_steps, num_envs, 1, dtype=torch.uint8),
    )
    return algorithm


def test_time_reversal_pairs_use_rollout_time_offset_before_flattening():
    algorithm = _make_temporal_pair_algorithm()

    reversed_observations, reference_observations, pair_mask = algorithm._time_reversal_pairs()

    expected_reference = algorithm.storage.observations[:-3].flatten(0, 1)
    current_observations = algorithm.storage.observations[3:].flatten(0, 1)
    current_actions = algorithm.storage.actions[3:].flatten(0, 1)
    expected_reversed_policy, _ = go2_symm.time_reverse_observation_history(
        current_observations["policy"], current_actions
    )
    assert torch.equal(reference_observations["policy"], expected_reference["policy"])
    assert torch.equal(reversed_observations["policy"], expected_reversed_policy)
    assert torch.equal(pair_mask, torch.ones(4, 1))


def test_time_reversal_pairs_mask_episode_boundaries_and_zero_padded_histories():
    algorithm = _make_temporal_pair_algorithm()
    algorithm.storage.dones[2, 0] = 1
    policy_frames = algorithm.storage.observations["policy"].reshape(5, 2, 3, FRAME_DIM)
    policy_frames[0, 1, 0] = 0.0

    _, _, pair_mask = algorithm._time_reversal_pairs()

    assert torch.equal(pair_mask, torch.tensor([[0.0], [0.0], [0.0], [1.0]]))


def test_time_reversal_pairs_do_not_mask_terminal_after_current_observation():
    algorithm = _make_temporal_pair_algorithm()
    algorithm.storage.dones[3, 0] = 1

    _, _, pair_mask = algorithm._time_reversal_pairs()

    assert torch.equal(pair_mask, torch.tensor([[1.0], [1.0], [0.0], [1.0]]))


def test_time_reversal_pairs_require_rollout_longer_than_history():
    algorithm = _make_temporal_pair_algorithm(history_length=5, num_steps=5)

    with pytest.raises(ValueError, match=r"num_steps_per_env \(5\).*history length \(5\)"):
        algorithm._time_reversal_pairs()


def test_update_uses_temporally_aligned_time_reversal_pairs():
    num_envs = 2
    num_steps = 24
    action_dim = 12
    policy_dim = 20 * FRAME_DIM
    observations = TensorDict(
        {"policy": torch.ones(num_envs, policy_dim)},
        batch_size=[num_envs],
    )
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        observations,
        obs_groups,
        "actor",
        action_dim,
        hidden_dims=[16],
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(observations, obs_groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", num_envs, num_steps, observations, [action_dim])
    algorithm = TimeReversalPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        desired_kl=None,
        symmetry_cfg={
            "_env": None,
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
            "data_augmentation_func": go2_symm.compute_time_reversal_states,
            "use_time_reversal_regularization": True,
            "value_loss_coeff": 0.05,
            "warmup_iterations": 0,
            "observation_frame_dim": FRAME_DIM,
            "command_observation_index": 3,
            "command_observation_scale": 2.0,
            "min_abs_command_velocity": 0.0,
        },
    )

    dones = torch.zeros(num_envs, dtype=torch.bool)
    for _ in range(num_steps):
        algorithm.act(observations)
        algorithm.process_env_step(observations, torch.ones(num_envs), dones, {})
    algorithm.compute_returns(observations)

    losses = algorithm.update()

    assert torch.isfinite(torch.tensor(losses["symmetry"]))
    assert torch.isfinite(torch.tensor(losses["tr_value"]))
    assert losses["trs/valid_pair_fraction"] == pytest.approx(1.0)
    assert storage.step == 0
