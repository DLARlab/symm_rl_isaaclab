# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for symmetric quadruped rollout plotting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _load_plotter_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "symm_rollout_plotter.py"
    spec = importlib.util.spec_from_file_location("symm_rollout_plotter_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


def _make_env():
    robot = MagicMock()
    robot.find_bodies.return_value = ([0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    robot.data.root_pos_w = _proxy(torch.tensor([[1.0, 2.0, 0.5]]))
    robot.data.heading_w = _proxy(torch.tensor([0.0]))
    robot.data.root_lin_vel_b = _proxy(torch.tensor([[0.4, 0.1, 0.0]]))
    robot.data.root_ang_vel_b = _proxy(torch.tensor([[0.0, 0.0, 0.2]]))
    robot.data.body_lin_vel_w = _proxy(
        torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.3],
                    [0.4, 0.0, 0.0],
                ]
            ]
        )
    )

    command_term = SimpleNamespace(
        command=torch.tensor([[0.5, 0.0, 0.1]]),
        periodic_force_weights=lambda: torch.tensor([[-1.0, -0.5, 0.0, -0.25]]),
        periodic_speed_weights=lambda: torch.tensor([[0.0, -0.5, -1.0, -0.75]]),
        robot=robot,
    )
    command_manager = MagicMock()
    command_manager.get_term.return_value = command_term

    sensors = {}
    for index, (sensor_name, body_name) in enumerate(
        zip(
            ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"),
            ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            strict=True,
        )
    ):
        sensors[sensor_name] = SimpleNamespace(
            num_sensors=1,
            body_names=[body_name],
            data=SimpleNamespace(net_forces_w=_proxy(torch.tensor([[[float(index + 1), 0.0, 0.0]]]))),
        )

    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        command_manager=command_manager,
        scene=SimpleNamespace(sensors=sensors),
    )
    return env, command_term, robot


def test_plotter_records_symmetric_rollout_signals(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)

    plotter.record()

    assert plotter._data["true_lin_vel"][0].tolist() == pytest.approx([0.4, 0.1, 0.2])
    assert plotter._data["desired_lin_vel"][0].tolist() == pytest.approx([0.5, 0.0, 0.1])
    assert plotter._data["E_C_frc"][0].tolist() == pytest.approx([-1.0, -0.5, 0.0, -0.25])
    assert plotter._data["E_C_spd"][0].tolist() == pytest.approx([0.0, -0.5, -1.0, -0.75])
    assert plotter._data["foot_forces"][0].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert plotter._data["foot_velocities"][0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_plotter_saves_isaacgym_compatible_outputs(tmp_path):
    module = _load_plotter_module()
    env, command_term, robot = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path)
    plotter.record()
    command_term.command = torch.tensor([[1.0, 0.0, 0.0]])
    robot.data.root_pos_w = _proxy(torch.tensor([[1.01, 2.0, 0.5]]))
    plotter.record()

    saved_paths = plotter.save()

    expected_names = {
        "sim_data.npz",
        "figure1_linear_velocities_and_position.png",
        "figure2_E_C_frc_and_contact_forces.png",
        "figure3_E_C_spd_and_foot_velocities.png",
        "figure4_agg_E_C_frc_vs_contact.png",
    }
    assert {path.name for path in saved_paths} == expected_names
    assert all(path.stat().st_size > 0 for path in saved_paths)
    with np.load(tmp_path / "sim_data.npz") as data:
        assert data["true_lin_vel"].shape == (2, 3)
        assert data["desired_positions"].shape == (2, 2)
        assert data["foot_forces"].shape == (2, 4)


def test_plotter_rejects_non_symmetric_command_term(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    env.command_manager.get_term.return_value = SimpleNamespace(command=torch.zeros(1, 3))

    with pytest.raises(ValueError, match="does not support symmetric plots"):
        module.SymmetricRolloutPlotter(env, tmp_path)


def test_plotter_limits_samples_for_long_play_sessions(tmp_path):
    module = _load_plotter_module()
    env, _, _ = _make_env()
    plotter = module.SymmetricRolloutPlotter(env, tmp_path, max_samples=1)

    plotter.record()
    plotter.record()

    assert len(plotter._data["time_steps"]) == 1
