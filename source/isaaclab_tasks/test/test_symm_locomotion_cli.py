# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared symmetric locomotion convenience CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_main_rejects_unsupported_launcher_option_before_delimiter(capsys):
    symm_cli = _load_symm_cli()

    with pytest.raises(SystemExit) as exc_info:
        symm_cli.main(["train", "--robot", "go2", "--dry-run", "--unsupported-launcher-option"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --unsupported-launcher-option" in captured.err


def test_main_removes_delimiter_and_preserves_forwarded_suffix(monkeypatch):
    symm_cli = _load_symm_cli()
    captured = {}

    def fake_run_isaaclab(args, isaaclab_args):
        captured["args"] = args
        captured["isaaclab_args"] = isaaclab_args
        return 0

    monkeypatch.setattr(symm_cli, "run_isaaclab", fake_run_isaaclab)

    result = symm_cli.main(
        [
            "train",
            "--robot",
            "go2",
            "--no-conda-run",
            "--",
            "--headless",
            "env.commands.base_velocity.ranges.lin_vel_x=(-1.0,2.0)",
        ]
    )

    assert result == 0
    assert captured["isaaclab_args"][-2:] == [
        "--headless",
        "env.commands.base_velocity.ranges.lin_vel_x=(-1.0,2.0)",
    ]
    assert "--" not in captured["isaaclab_args"]


def test_main_preserves_delimiter_free_hydra_overrides(monkeypatch):
    symm_cli = _load_symm_cli()
    captured = {}

    def fake_run_isaaclab(args, isaaclab_args):
        captured["isaaclab_args"] = isaaclab_args
        return 0

    monkeypatch.setattr(symm_cli, "run_isaaclab", fake_run_isaaclab)

    result = symm_cli.main(
        [
            "train",
            "--robot",
            "go2",
            "--no-conda-run",
            "env.commands.base_velocity.ranges.lin_vel_x=(-1.0,2.0)",
        ]
    )

    assert result == 0
    assert captured["isaaclab_args"][-1] == "env.commands.base_velocity.ranges.lin_vel_x=(-1.0,2.0)"


def test_train_ramp_and_expected_branch_options(monkeypatch):
    symm_cli = _load_symm_cli()
    captured = {}

    monkeypatch.setattr(symm_cli, "git_branch", lambda: "jding/symm-72d-trs-milestone")

    def fake_run_isaaclab(args, isaaclab_args):
        captured["args"] = args
        captured["isaaclab_args"] = isaaclab_args
        return 0

    monkeypatch.setattr(symm_cli, "run_isaaclab", fake_run_isaaclab)

    result = symm_cli.main(
        [
            "train",
            "--robot",
            "go2",
            "--seed",
            "42",
            "--expected-branch",
            "jding/symm-72d-trs-milestone",
            "--tr-rampup-iterations",
            "2000",
            "--tr-ramp-shape",
            "half_cosine",
            "--no-conda-run",
        ]
    )

    assert result == 0
    assert captured["args"].expected_branch == "jding/symm-72d-trs-milestone"
    assert captured["args"].tr_rampup_iterations == 2000
    assert captured["args"].tr_ramp_shape == "half_cosine"
    assert "agent.algorithm.symmetry_cfg.rampup_iterations=2000" in captured["isaaclab_args"]
    assert "agent.algorithm.symmetry_cfg.ramp_shape=half_cosine" in captured["isaaclab_args"]


def test_repo_subprocess_environment_prefers_launcher_checkout(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    repo_root = tmp_path / "launcher_checkout"
    source_root = repo_root / "source"
    local_projects = [source_root / "isaaclab_rl", source_root / "isaaclab_tasks"]
    for project in local_projects:
        package = project / project.name
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
    stale_checkout = tmp_path / "stale_checkout"
    monkeypatch.setattr(symm_cli, "repo_root", lambda: repo_root)
    monkeypatch.setenv("PYTHONPATH", str(stale_checkout))

    environment = symm_cli.repo_subprocess_environment()

    python_paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python_paths[:2] == [str(path.resolve()) for path in local_projects]
    assert python_paths[2:] == [str(stale_checkout)]


def test_run_isaaclab_forwards_checkout_environment(monkeypatch):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2", "--no-conda-run"])
    args.robot_spec = symm_cli.get_robot(args.robot)
    expected_environment = {"PYTHONPATH": "launcher-checkout"}
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(symm_cli, "repo_subprocess_environment", lambda: expected_environment)
    monkeypatch.setattr(symm_cli.subprocess, "run", fake_run)

    result = symm_cli.run_isaaclab(args, ["train"])

    assert result == 0
    assert captured["env"] is expected_environment
    assert captured["cwd"] == symm_cli.repo_root()


def test_expected_branch_rejects_wrong_checkout(monkeypatch):
    symm_cli = _load_symm_cli()
    parser = symm_cli.build_parser()
    args = parser.parse_args(["train", "--robot", "go2", "--expected-branch", "expected"])
    args.robot_spec = symm_cli.get_robot(args.robot)
    monkeypatch.setattr(symm_cli, "git_branch", lambda: "main")

    with pytest.raises(ValueError, match="Expected branch 'expected'.*current branch is 'main'"):
        symm_cli.validate_expected_branch(args)


@pytest.mark.skipif(os.name != "nt", reason="PowerShell wrapper regression is Windows-specific")
def test_powershell_train_wrapper_forwards_python_style_options():
    repo_root = Path(__file__).resolve().parents[3]
    wrapper = repo_root / "scripts" / "symm_locomotion" / "train.ps1"
    environment = os.environ.copy()
    environment["CONDA_PREFIX"] = str(repo_root / "fake_active_base_environment")
    environment["CONDA_DEFAULT_ENV"] = "base"

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "--robot",
            "x1",
            "--smoke",
            "--dry-run",
            "--no-conda-run",
            "--",
            "--headless",
            "env.scene.num_envs=1",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[symm_locomotion:x1]" in result.stdout
    assert "Isaac-Velocity-Flat-Dobot-X1-Symm-v0" in result.stdout
    assert "--headless env.scene.num_envs=1" in result.stdout


def test_train_defaults_apply_shared_scale_and_trs_settings():
    symm_cli = _load_symm_cli()
    parser = symm_cli.build_parser()
    args = parser.parse_args(["train", "--robot", "go2", "--no-conda-run"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert args.num_envs == 512
    assert args.iterations == 20000
    assert args.tr_rampup_iterations == 0
    assert args.tr_ramp_shape == "linear"
    assert command[command.index("--num_envs") + 1] == "512"
    assert command[command.index("--max_iterations") + 1] == "20000"
    assert args.tr_min_abs_cmd_vel == 0.0
    assert "agent.algorithm.symmetry_cfg.use_data_augmentation=False" in command
    assert "agent.algorithm.symmetry_cfg.rampup_iterations=0" in command
    assert "agent.algorithm.symmetry_cfg.ramp_shape=linear" in command
    assert "agent.algorithm.symmetry_cfg.min_abs_command_velocity=0.0" in command


def test_ramped_train_run_name_records_complete_schedule():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(
        [
            "train",
            "--robot",
            "go2",
            "--seed",
            "42",
            "--mirror",
            "0.20",
            "--tr-value-coef",
            "0.10",
            "--tr-warmup-iterations",
            "0",
            "--tr-rampup-iterations",
            "2000",
            "--tr-ramp-shape",
            "linear",
            "--no-conda-run",
        ]
    )
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert command[command.index("--run_name") + 1] == "go2_trs_linear_m0p2_v0p1_w0_r2000_seed42"


def test_ablation_uses_shared_training_scale_defaults():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["ablation", "--robot", "go2", "--no-conda-run"])

    assert args.num_envs == 512
    assert args.iterations == 20000
    assert args.mirror_loss_coeff == 0.2
    assert args.tr_value_coeff == 0.05


def test_schedule_ablation_requires_explicit_coefficients(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(
        ["ablation", "--robot", "go2", "--schedule-variants", "linear", "--dry-run", "--no-conda-run"]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "require explicit --mirror and --tr-value-coef" in captured.err


def test_schedule_ablation_dry_run_resolves_ramp_variants(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(
        [
            "ablation",
            "--robot",
            "go2",
            "--seeds",
            "42",
            "--mirror",
            "0.20",
            "--tr-value-coef",
            "0.10",
            "--schedule-variants",
            "no_trs",
            "hard",
            "linear",
            "delayed_linear",
            "half_cosine",
            "--dry-run",
            "--no-conda-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    command_lines = [line for line in captured.out.splitlines() if " --run_name " in line]
    assert len(command_lines) == 5
    assert any("_w0_r2000_linear_seed42" in line for line in command_lines)
    assert any("_w500_r1500_linear_seed42" in line for line in command_lines)
    assert any("_w0_r2000_half_cosine_seed42" in line for line in command_lines)


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


def test_play_and_record_enable_fixed_five_second_gait_sequence_by_default(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()

    for subcommand in ("play", "record"):
        args = parser.parse_args([subcommand, "--robot", "go2", "--no-conda-run"])
        args.robot_spec = symm_cli.get_robot(args.robot)
        command = symm_cli.play_lab_args(args, []) if subcommand == "play" else symm_cli.record_lab_args(args, [])[0]

        assert "env.commands.base_velocity.gait_sequence_enabled=true" in command
        assert "env.commands.base_velocity.gait_sequence_duration_s=5.0" in command
        assert "env.episode_length_s=30.02" in command


def test_play_and_record_can_disable_or_retime_fixed_gait_sequence(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "model_9999.pt"
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    parser = symm_cli.build_parser()

    disabled_args = parser.parse_args(["play", "--robot", "go2", "--no-gait-sequence", "--no-conda-run"])
    disabled_args.robot_spec = symm_cli.get_robot(disabled_args.robot)
    disabled_command = symm_cli.play_lab_args(disabled_args, [])
    assert "env.commands.base_velocity.gait_sequence_enabled=false" in disabled_command
    assert "env.episode_length_s=30.0" in disabled_command

    retimed_args = parser.parse_args(["record", "--robot", "x1", "--gait-sequence-duration", "2.5", "--no-conda-run"])
    retimed_args.robot_spec = symm_cli.get_robot(retimed_args.robot)
    retimed_command, _ = symm_cli.record_lab_args(retimed_args, [])
    assert "env.commands.base_velocity.gait_sequence_enabled=true" in retimed_command
    assert "env.commands.base_velocity.gait_sequence_duration_s=2.5" in retimed_command
    assert "env.episode_length_s=15.02" in retimed_command
    video_length_index = retimed_command.index("--video_length") + 1
    assert retimed_command[video_length_index] == "750"

    too_short_args = parser.parse_args(
        ["record", "--robot", "x1", "--gait-sequence-duration", "0.001", "--no-conda-run"]
    )
    too_short_args.robot_spec = symm_cli.get_robot(too_short_args.robot)
    with pytest.raises(ValueError, match="at least one environment step"):
        symm_cli.record_lab_args(too_short_args, [])


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
