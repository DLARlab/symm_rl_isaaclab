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


def test_train_defaults_apply_shared_scale_and_trs_settings():
    symm_cli = _load_symm_cli()
    parser = symm_cli.build_parser()
    args = parser.parse_args(["train", "--robot", "go2", "--no-conda-run"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert args.num_envs == 512
    assert args.iterations == 20000
    assert command[command.index("--num_envs") + 1] == "512"
    assert command[command.index("--max_iterations") + 1] == "20000"
    assert args.tr_min_abs_cmd_vel == 0.0
    assert "agent.algorithm.symmetry_cfg.min_abs_command_velocity=0.0" in command


def test_ablation_uses_shared_training_scale_defaults():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["ablation", "--robot", "go2", "--no-conda-run"])

    assert args.num_envs == 512
    assert args.iterations == 20000


def test_no_trs_disables_every_auxiliary_symmetry_training_path():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2", "--no-conda-run", "--no-trs"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert "agent.algorithm.symmetry_cfg.use_data_augmentation=False" in command
    assert "agent.algorithm.symmetry_cfg.use_mirror_loss=False" in command
    assert "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0" in command
    assert "agent.algorithm.symmetry_cfg.value_loss_coeff=0.0" in command


def test_record_defaults_to_thirty_seconds(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()

    for robot in ("go2", "x1"):
        args = parser.parse_args(["record", "--robot", robot, "--no-conda-run"])
        args.robot_spec = symm_cli.get_robot(args.robot)
        command, resolved_checkpoint = symm_cli.record_lab_args(args, [])

        video_length_index = command.index("--video_length") + 1
        assert command[video_length_index] == "1500"
        assert resolved_checkpoint == checkpoint


def test_record_video_length_override_is_preserved(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()
    args = parser.parse_args(["record", "--robot", "go2", "--video-length", "400", "--no-conda-run"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command, _ = symm_cli.record_lab_args(args, [])

    video_length_index = command.index("--video_length") + 1
    assert command[video_length_index] == "400"


def test_gif_conversion_rejects_stale_play_video(monkeypatch, tmp_path, capsys):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    video_dir = tmp_path / "videos" / "play"
    video_dir.mkdir(parents=True)
    (video_dir / "rl-video-step-0.mp4").touch()
    previous_videos = symm_cli.play_video_snapshot(tmp_path)
    parser = symm_cli.build_parser()
    args = parser.parse_args(["record", "--robot", "go2", "--gif", "--no-conda-run"])
    args.robot_spec = symm_cli.get_robot(args.robot)
    monkeypatch.setattr(symm_cli.subprocess, "run", lambda *args, **kwargs: None)

    result = symm_cli.convert_latest_video(args, checkpoint, previous_videos)

    captured = capsys.readouterr()
    assert result == 1
    assert "No new or updated MP4" in captured.out


def test_record_rejects_success_code_without_new_video(monkeypatch, tmp_path, capsys):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    video_dir = tmp_path / "videos" / "play"
    video_dir.mkdir(parents=True)
    (video_dir / "rl-video-step-0.mp4").touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    monkeypatch.setattr(symm_cli, "run_isaaclab", lambda args, lab_args: 0)

    result = symm_cli.main(["record", "--robot", "go2", "--no-conda-run"])

    captured = capsys.readouterr()
    assert result == 1
    assert "recording finished without a new or updated MP4" in captured.err


def test_play_and_record_enable_rollout_plots_by_default(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()

    for subcommand in ("play", "record"):
        args = parser.parse_args([subcommand, "--robot", "go2", "--no-conda-run"])
        args.robot_spec = symm_cli.get_robot(args.robot)
        command = symm_cli.play_lab_args(args, []) if subcommand == "play" else symm_cli.record_lab_args(args, [])[0]

        assert "--symm_rollout_plots" in command
        plot_env_index = command.index("--symm_rollout_plot_env_index") + 1
        assert command[plot_env_index] == "0"
        plot_max_steps = command.index("--symm_rollout_plot_max_steps") + 1
        assert command[plot_max_steps] == "1500"


def test_rollout_plots_can_be_disabled_or_redirected(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()

    disabled_args = parser.parse_args(["record", "--robot", "x1", "--no-plots", "--no-conda-run"])
    disabled_args.robot_spec = symm_cli.get_robot(disabled_args.robot)
    disabled_command, _ = symm_cli.record_lab_args(disabled_args, [])
    assert "--symm_rollout_plots" not in disabled_command

    plots_dir = tmp_path / "plots"
    redirected_args = parser.parse_args(
        ["play", "--robot", "x1", "--plots_dir", str(plots_dir), "--plot_env_index", "2", "--no-conda-run"]
    )
    redirected_args.robot_spec = symm_cli.get_robot(redirected_args.robot)
    redirected_command = symm_cli.play_lab_args(redirected_args, [])
    assert redirected_command[redirected_command.index("--symm_rollout_plots_dir") + 1] == str(plots_dir)
    assert redirected_command[redirected_command.index("--symm_rollout_plot_env_index") + 1] == "2"


def test_compare_handles_missing_log_directories(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["compare", "--robots", "go2", "x1", "--limit", "1", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "robot" in captured.out
    assert "go2" in captured.out
    assert "x1" in captured.out


def test_tensorboard_dry_run_uses_python_module(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["tensorboard", "--robots", "go2", "x1", "--dry-run", "--no-conda-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "-m tensorboard.main" in captured.out
    assert "logs" in captured.out
    assert "rsl_rl" in captured.out
