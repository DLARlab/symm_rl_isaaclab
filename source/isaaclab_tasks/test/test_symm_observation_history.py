# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for frame-major symmetric-quadruped observation history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped import env as symm_quadruped_env
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.observation_history import (
    FrameMajorObservationHistoryWrapper,
    PolicyObservationHistoryCfg,
)


class _FakeObservationManager:
    """Minimal observation manager used to exercise the history wrapper."""

    def __init__(self, policy: torch.Tensor):
        self.policy = policy
        self.critic = torch.full((policy.shape[0], 1), -1.0)
        self.group_obs_dim = {"policy": (policy.shape[1],), "critic": (1,)}
        self.reset_calls = []

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor]:
        del update_history
        return {"policy": self.policy.clone(), "critic": self.critic.clone()}

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        self.reset_calls.append(env_ids)
        return {}


def test_frame_major_history_stacks_complete_observation_frames():
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    wrapper = FrameMajorObservationHistoryWrapper(
        manager,
        num_envs=1,
        device="cpu",
        history_length=3,
    )

    first = wrapper.compute(update_history=True)["policy"].clone()
    manager.policy[:] = torch.tensor([[3.0, 4.0]])
    second = wrapper.compute(update_history=True)["policy"].clone()
    manager.policy[:] = torch.tensor([[5.0, 6.0]])
    third = wrapper.compute(update_history=True)

    assert torch.equal(first, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]]))
    assert torch.equal(second, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(third["policy"], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
    assert torch.equal(third["critic"], torch.tensor([[-1.0]]))
    assert wrapper.group_obs_dim == {"policy": (6,), "critic": (1,)}


def test_history_read_without_update_does_not_duplicate_latest_frame():
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    wrapper = FrameMajorObservationHistoryWrapper(
        manager,
        num_envs=1,
        device="cpu",
        history_length=2,
    )
    wrapper.compute(update_history=True)
    manager.policy[:] = torch.tensor([[3.0, 4.0]])

    observation = wrapper.compute(update_history=False)

    assert torch.equal(observation["policy"], torch.tensor([[0.0, 0.0, 1.0, 2.0]]))


@pytest.mark.parametrize("history_length", [1, 3])
def test_returned_history_is_not_mutated_by_later_updates_or_resets(history_length):
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    wrapper = FrameMajorObservationHistoryWrapper(
        manager,
        num_envs=1,
        device="cpu",
        history_length=history_length,
    )
    first = wrapper.compute(update_history=True)["policy"]
    expected_first = first.clone()

    manager.policy[:] = torch.tensor([[3.0, 4.0]])
    wrapper.compute(update_history=True)
    wrapper.reset(torch.tensor([0]))

    assert torch.equal(first, expected_first)


def test_initial_history_read_places_current_frame_last():
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    wrapper = FrameMajorObservationHistoryWrapper(
        manager,
        num_envs=1,
        device="cpu",
        history_length=2,
    )

    first = wrapper.compute(update_history=False)
    manager.policy[:] = torch.tensor([[3.0, 4.0]])
    second = wrapper.compute(update_history=False)

    assert torch.equal(first["policy"], torch.tensor([[0.0, 0.0, 1.0, 2.0]]))
    assert torch.equal(second["policy"], torch.tensor([[0.0, 0.0, 1.0, 2.0]]))


def test_history_reset_clears_only_selected_environments():
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0], [10.0, 20.0]]))
    wrapper = FrameMajorObservationHistoryWrapper(
        manager,
        num_envs=2,
        device="cpu",
        history_length=2,
    )
    wrapper.compute(update_history=True)
    manager.policy[:] = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
    wrapper.compute(update_history=True)

    reset_ids = torch.tensor([1])
    wrapper.reset(reset_ids)
    manager.policy[:] = torch.tensor([[5.0, 6.0], [50.0, 60.0]])
    observation = wrapper.compute(update_history=True)

    assert torch.equal(observation["policy"][0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(observation["policy"][1], torch.tensor([0.0, 0.0, 50.0, 60.0]))
    assert manager.reset_calls == [reset_ids]


@pytest.mark.parametrize("history_length", [0, -1])
def test_history_length_must_be_positive(history_length):
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="history_length must be positive"):
        FrameMajorObservationHistoryWrapper(
            manager,
            num_envs=1,
            device="cpu",
            history_length=history_length,
        )


def test_history_wrapper_requires_a_flat_concatenated_group():
    manager = SimpleNamespace(group_obs_dim={"policy": [(2,), (3,)]})

    with pytest.raises(ValueError, match="concatenated observation group"):
        FrameMajorObservationHistoryWrapper(
            manager,
            num_envs=1,
            device="cpu",
            history_length=2,
        )


def test_symmetric_environment_installs_history_wrapper(monkeypatch):
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    env = object.__new__(symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        policy_observation_history=PolicyObservationHistoryCfg(history_length=3),
    )
    env.scene = SimpleNamespace(num_envs=1)
    env.sim = SimpleNamespace(device="cpu")
    env.observation_manager = manager
    configured_spaces = []
    env._configure_gym_env_spaces = lambda: configured_spaces.append(True)
    monkeypatch.setattr(symm_quadruped_env.ManagerBasedRLEnv, "load_managers", lambda _self: None)

    env.load_managers()

    assert isinstance(env.observation_manager, FrameMajorObservationHistoryWrapper)
    assert env.observation_manager.group_obs_dim["policy"] == (6,)
    assert configured_spaces == [True]


def test_symmetric_environment_uses_native_observations_for_history_length_one(monkeypatch):
    manager = _FakeObservationManager(torch.tensor([[1.0, 2.0]]))
    env = object.__new__(symm_quadruped_env.SymmQuadrupedManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        policy_observation_history=PolicyObservationHistoryCfg(history_length=1),
    )
    env.scene = SimpleNamespace(num_envs=1)
    env.sim = SimpleNamespace(device="cpu")
    env.observation_manager = manager
    configured_spaces = []
    env._configure_gym_env_spaces = lambda: configured_spaces.append(True)
    monkeypatch.setattr(symm_quadruped_env.ManagerBasedRLEnv, "load_managers", lambda _self: None)

    env.load_managers()

    assert env.observation_manager is manager
    assert configured_spaces == []
