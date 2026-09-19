# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the shell launcher and batched recording command construction."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import symm_cli

    return symm_cli


def test_diff_cmds_builds_six_groups_and_gait_specific_output(cli, tmp_path):
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    for gait, theta in cli.DIFF_CMDS_GAITS.items():
        args = cli.build_parser().parse_args(
            ["diff_cmds", "--robot", "x1", "--gait", gait, "--checkpoint", str(checkpoint)]
        )
        args.robot_spec = cli.get_robot(args.robot)
        command = cli.diff_cmds_lab_args(args, ["env.policy_observation_history.history_length=20"])
        assert command[command.index("--num_envs") + 1] == "600"
        assert command[command.index("--symm_command_batch_duration") + 1] == "20.0"
        assert command[command.index("--symm_command_batch_envs_per_command") + 1] == "100"
        assert Path(command[command.index("--evaluation_output_dir") + 1]).name == gait
        assert f"env.commands.base_velocity.init_foot_thetas=[[{','.join(map(str, theta))}]]" in command
        assert "--tracking_error_direction_test" not in command
        assert "--video" not in command and "--symm_rollout_plots" not in command
        assert command[-1] == "env.policy_observation_history.history_length=20"


def test_shell_launches_all_six_gaits_and_accepts_checkpoint_override(tmp_path):
    root = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path / "alternative checkpoint.pt"
    checkpoint.touch()
    result = subprocess.run(
        ["bash", str(root / "test_diff_cmds.sh"), "--checkpoint", str(checkpoint), "--dry-run", "--no-conda-run"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "--symm_command_batch " in line]
    assert len(commands) == 6
    for line in commands:
        assert "--num_envs 600" in line
        assert "--symm_command_batch_duration 20.0" in line
        assert str(checkpoint) in line
    for gait in ("trot", "bound", "half-bound-left", "half-bound-right", "rotary-gallop", "transverse-gallop"):
        assert sum(f"/{gait}'" in line or f"/{gait} " in line for line in commands) == 1
