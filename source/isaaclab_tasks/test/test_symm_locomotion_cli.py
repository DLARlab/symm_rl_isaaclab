# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared symmetric locomotion convenience CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_symm_cli():
    repo_root = Path(__file__).resolve().parents[3]
    cli_path = repo_root / "scripts" / "symm_locomotion" / "symm_cli.py"
    return _load_module("symm_cli_under_test", cli_path)


def test_robot_aliases_resolve_to_expected_tasks():
    symm_cli = _load_symm_cli()

    assert symm_cli.get_robot("go2").train_task == "Isaac-Velocity-Flat-Unitree-Go2-Symm-v0"
    assert symm_cli.get_robot("dobot").train_task == "Isaac-Velocity-Flat-Dobot-X1-Symm-v0"
    assert symm_cli.get_robot("dobot_x1").play_task == "Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0"


def test_train_dry_run_uses_selected_robot_task(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["train", "--robot", "x1", "--smoke", "--dry-run", "--no-conda-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[symm_locomotion:x1]" in captured.out
    assert "Isaac-Velocity-Flat-Dobot-X1-Symm-v0" in captured.out
    assert "--max_iterations" in captured.out
    assert " 1" in captured.out


def test_compare_handles_missing_log_directories(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["compare", "--robots", "go2", "x1", "--limit", "1", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "robot" in captured.out
    assert "go2" in captured.out
    assert "x1" in captured.out
