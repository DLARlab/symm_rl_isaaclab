# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for batched velocity-command tracking-error evaluation."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_tracking_error_grid_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "tracking_error_grid.py"
    spec = importlib.util.spec_from_file_location("tracking_error_grid_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_fake_env(num_envs: int):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=SimpleNamespace(torch=torch.zeros((num_envs, 3))),
            root_ang_vel_b=SimpleNamespace(torch=torch.zeros((num_envs, 3))),
        )
    )
    command_term = SimpleNamespace(
        vel_command_b=torch.zeros((num_envs, 3)),
        is_standing_env=torch.ones(num_envs, dtype=torch.bool),
        is_heading_env=torch.ones(num_envs, dtype=torch.bool),
        robot=robot,
        _resample_command=lambda env_ids: None,
    )
    command_manager = SimpleNamespace(get_term=lambda name: command_term)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", command_manager=command_manager)
    return env, command_term


def test_tracking_error_grid_contains_requested_23_command_groups():
    tracking_grid = _load_tracking_error_grid_module()

    tests = tracking_grid.TRACKING_ERROR_COMMAND_TESTS

    assert len(tests) == 23
    assert [test[1] for test in tests[:13]] == [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    assert [test[2] for test in tests[13:16]] == [-0.5, 0.0, 0.5]
    assert [test[3] for test in tests[16:]] == [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def test_tracking_error_grid_assigns_commands_and_writes_mae_table(tmp_path):
    tracking_grid = _load_tracking_error_grid_module()
    envs_per_command = 2
    env, command_term = _make_fake_env(len(tracking_grid.TRACKING_ERROR_COMMAND_TESTS) * envs_per_command)
    output_path = tmp_path / "tracking_errors.csv"
    evaluator = tracking_grid.TrackingErrorGridEvaluator(
        env,
        envs_per_command=envs_per_command,
        warmup_steps=1,
        measurement_steps=2,
        output_path=output_path,
    )

    evaluator.install_command_override()
    assigned_commands = command_term.vel_command_b.clone()
    command_term.vel_command_b.zero_()
    command_term._resample_command(slice(None))
    assert torch.equal(command_term.vel_command_b, assigned_commands)
    assert not torch.any(command_term.is_standing_env)
    assert not torch.any(command_term.is_heading_env)

    command_term.robot.data.root_lin_vel_b.torch[:, :2] = assigned_commands[:, :2] + torch.tensor([1.0, -2.0])
    command_term.robot.data.root_ang_vel_b.torch[:, 2] = assigned_commands[:, 2] + 0.5
    dones = torch.zeros(env.num_envs, dtype=torch.long)
    evaluator.record(dones)
    evaluator.record(dones)
    evaluator.record(dones)
    assert evaluator.is_complete
    assert evaluator.write_csv() == output_path.resolve()

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 23
    assert set(rows[0]) == {
        "sweep",
        "command_x_mps",
        "command_y_mps",
        "command_yaw_radps",
        "tracking_error_x_mae_mps",
        "tracking_error_y_mae_mps",
        "tracking_error_xy_mae_mps",
        "tracking_error_yaw_mae_radps",
    }
    assert float(rows[0]["tracking_error_x_mae_mps"]) == pytest.approx(1.0)
    assert float(rows[0]["tracking_error_y_mae_mps"]) == pytest.approx(2.0)
    assert float(rows[0]["tracking_error_xy_mae_mps"]) == pytest.approx(5.0**0.5, abs=1.0e-6)
    assert float(rows[0]["tracking_error_yaw_mae_radps"]) == pytest.approx(0.5)

