# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared symmetric locomotion convenience CLI."""

from __future__ import annotations

import importlib.util
import json
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
    assert args.mirror_loss_coeff == pytest.approx(0.1)
    assert args.tr_value_coeff == 0.0
    assert args.tr_rampup_iterations == 0
    assert args.tr_ramp_shape == "linear"
    assert command[command.index("--num_envs") + 1] == "512"
    assert command[command.index("--max_iterations") + 1] == "20000"
    assert args.tr_min_abs_cmd_vel == 0.0
    assert "agent.algorithm.symmetry_cfg.use_data_augmentation=False" in command
    assert "agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false" in command
    assert "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1" in command
    assert "agent.algorithm.symmetry_cfg.value_loss_coeff=0.0" in command
    assert "agent.algorithm.symmetry_cfg.rampup_iterations=0" in command
    assert "agent.algorithm.symmetry_cfg.ramp_shape=linear" in command
    assert "agent.algorithm.symmetry_cfg.min_abs_command_velocity=0.0" in command
    assert "env.observations.policy.history_length=30" in command
    assert "env.observations.policy.flatten_history_dim=true" in command
    assert "agent.algorithm.symmetry_cfg.history_enabled=true" in command
    assert "agent.algorithm.symmetry_cfg.history_length=30" in command
    assert "agent.algorithm.symmetry_cfg.tr_consistency_mode='transition_aligned_sequence'" in command


@pytest.mark.parametrize("robot", ["go2", "x1"])
def test_train_explicit_positive_tr_value_coefficient_is_forwarded(robot):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", robot, "--tr-value-coef", "0.05"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert args.tr_value_coeff == pytest.approx(0.05)
    assert "agent.algorithm.symmetry_cfg.value_loss_coeff=0.05" in command


def test_train_tr_value_help_distinguishes_optional_ablation_from_ppo_critic():
    symm_cli = _load_symm_cli()
    parser = symm_cli.build_parser()
    subcommands = next(action for action in parser._actions if action.dest == "command")
    train_parser = subcommands.choices["train"]
    help_text = next(action.help for action in train_parser._actions if action.dest == "tr_value_coeff")

    assert "Optional coefficient for the time-reversal critic-consistency ablation" in help_text
    assert "Default: 0.0" in help_text
    assert "not the standard PPO value-loss coefficient" in help_text
    assert "positive value explicitly enables" in help_text


@pytest.mark.parametrize("command", ["train", "play", "record", "evaluation", "ablation"])
def test_history_defaults_are_shared_by_launcher_workflows(command):
    symm_cli = _load_symm_cli()

    args = symm_cli.build_parser().parse_args([command])

    assert args.history_enabled is True
    assert args.history_length == 30
    assert args.tr_consistency_mode == "transition_aligned_sequence"
    assert symm_cli.resolve_policy_history(args) == (True, 30)


@pytest.mark.parametrize(
    ("launcher_args", "expected_mode", "expected_label"),
    [
        ([], "transition_aligned_sequence", "trseq"),
        (["--tr-consistency-mode", "framewise_feature_approx"], "framewise_feature_approx", "trff"),
        (["--no-trs"], "none", "notr"),
    ],
)
def test_tr_consistency_mode_controls_resolved_config_and_run_label(launcher_args, expected_mode, expected_label):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2", *launcher_args])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert f"agent.algorithm.symmetry_cfg.tr_consistency_mode='{expected_mode}'" in command
    assert f"agent.algorithm.symmetry_cfg.tr_consistency_mode={expected_mode}" not in command
    assert command[command.index("--run_name") + 1].startswith(f"go2_{expected_label}_")


@pytest.mark.parametrize("no_history_option", ["--no_history", "--no-history"])
@pytest.mark.parametrize("length_option", ["--history_length", "--history-length"])
def test_history_aliases_disable_history_and_force_zero(no_history_option, length_option):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", no_history_option, length_option, "7", "--robot", "go2"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert symm_cli.resolve_policy_history(args) == (False, 0)
    assert "env.observations.policy.history_length=0" in command
    assert "agent.algorithm.symmetry_cfg.history_enabled=false" in command
    assert "agent.algorithm.symmetry_cfg.history_length=0" in command
    assert command[command.index("--run_name") + 1] == "go2_trseq_m0p1_v0_h0_cur0_seeddefault"


@pytest.mark.parametrize("history_length", [0, -1])
def test_enabled_history_rejects_nonpositive_lengths(history_length, capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["train", "--history-length", str(history_length), "--dry-run", "--no-conda-run"])

    assert result == 2
    assert "must be a positive integer when history is enabled" in capsys.readouterr().err


def test_history_dry_run_prints_exact_hydra_overrides(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["train", "--no-history", "--smoke", "--dry-run", "--no-conda-run"])

    output = capsys.readouterr().out
    assert result == 0
    assert "env.observations.policy.history_length=0" in output
    assert "agent.algorithm.symmetry_cfg.history_enabled=false" in output
    assert "agent.algorithm.symmetry_cfg.history_length=0" in output
    assert "agent.algorithm.symmetry_cfg.tr_consistency_mode='transition_aligned_sequence'" in output


def test_history_overrides_are_identical_for_play_record_and_evaluation(monkeypatch, tmp_path):
    symm_cli = _load_symm_cli()
    checkpoint = tmp_path / "run" / "model_99.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.setattr(symm_cli, "resolve_checkpoint", lambda args: checkpoint)
    expected = symm_cli.policy_history_lab_args(symm_cli.build_parser().parse_args(["play", "--no-history"]))

    commands = []
    for launcher in ("play", "record", "evaluation"):
        args = symm_cli.build_parser().parse_args([launcher, "--robot", "go2", "--no-history"])
        args.robot_spec = symm_cli.get_robot(args.robot)
        if launcher == "play":
            commands.append(symm_cli.play_lab_args(args, []))
        elif launcher == "record":
            commands.append(symm_cli.record_lab_args(args, [])[0])
        else:
            commands.append(symm_cli.evaluation_lab_args(args, checkpoint, tmp_path / "study.json", []))

    for command in commands:
        assert all(override in command for override in expected)


def test_train_direct_context_records_policy_contract_metadata():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "x1", "--history-length", "5", "--seed", "42"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])
    context_index = command.index("--symm_direct_launch_context")
    context = json.loads(command[context_index + 1])

    assert context["policy_contract"] == {
        "gait_library_version": "time_reversal_closed_v2",
        "gait_phase_mapping_version": "same_gait_backward_duty_aware_integrated_reward_boundary_v4",
        "history_enabled": True,
        "history_length": 5,
        "history_packing": "term_major_oldest_to_newest_flattened",
        "action_history_length": 2,
        "actor_alignment": "edge_t_to_reverse_state_t_plus_1",
        "allowed_policy_version_span": 1,
        "candidate_max_age_updates": 1,
        "required_sequence_records": 7,
        "sequence_history_length": 5,
        "tr_consistency_mapping_version": "transition_aligned_causal_sequence_v1",
        "tr_consistency_mode": "transition_aligned_sequence",
        "value_alignment": "state_t_plus_1",
        "instantaneous_frame_dim": 64,
        "observation_contract_version": "hardware_proprio_history_64d_v1",
    }
    assert context["time_reversal_treatment"] == {
        "mirror_loss_coeff": 0.1,
        "trajectory_augmentation_enabled": False,
        "use_data_augmentation": False,
        "value_loss_coeff": 0.0,
    }


def test_command_curriculum_flags_build_validated_overrides_and_run_name():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(
        [
            "train",
            "--robot",
            "go2",
            "--seed",
            "42",
            "--command-curriculum",
            "--curriculum-velocity-bins",
            "13",
            "--curriculum-ewma",
            "0.2",
            "--curriculum-unlock-threshold",
            "0.75",
            "--curriculum-initial-max-abs-speed",
            "0.4",
            "--curriculum-min-visits",
            "24",
            "--curriculum-current-cell-increment",
            "0.8",
            "--curriculum-neighbor-increment",
            "0.3",
            "--curriculum-exploration-floor",
            "0.04",
            "--curriculum-maximum-weight",
            "8.0",
            "--curriculum-locked-cell-weight",
            "0.0",
            "--curriculum-seed",
            "123",
        ]
    )
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert "env.commands.base_velocity.command_curriculum_mode='tr_orbit_reward_threshold_v1'" in command
    assert "env.commands.base_velocity.curriculum_velocity_bin_count=13" in command
    assert "env.commands.base_velocity.curriculum_ewma_coefficient=0.2" in command
    assert "env.commands.base_velocity.curriculum_unlock_threshold=0.75" in command
    assert "env.commands.base_velocity.curriculum_initial_max_abs_speed=0.4" in command
    assert "env.commands.base_velocity.curriculum_min_visits=24" in command
    assert "env.commands.base_velocity.curriculum_current_cell_increment=0.8" in command
    assert "env.commands.base_velocity.curriculum_neighbor_increment=0.3" in command
    assert "env.commands.base_velocity.curriculum_exploration_floor=0.04" in command
    assert "env.commands.base_velocity.curriculum_maximum_weight=8.0" in command
    assert "env.commands.base_velocity.curriculum_locked_cell_weight=0.0" in command
    assert "env.commands.base_velocity.curriculum_seed=123" in command
    assert command[command.index("--run_name") + 1] == "go2_trseq_m0p1_v0_h30_cur1_seed42"


def test_disabled_command_curriculum_override_is_quoted_for_hydra():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2", "--seed", "42"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert "env.commands.base_velocity.command_curriculum_mode='none'" in command
    assert "env.commands.base_velocity.command_curriculum_mode=none" not in command


@pytest.mark.parametrize(
    "extra",
    [
        ["--symm_direct_launch_context", "{}"],
        ["--symm_direct_launch_context={}"],
        ["--symm-direct-launch-context", "{}"],
    ],
)
def test_train_rejects_passthrough_direct_context_override(extra):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    with pytest.raises(ValueError, match="does not allow overriding"):
        symm_cli.train_lab_args(args, extra)


@pytest.mark.parametrize("option", ["--command_curriculum", "--command-curriculum"])
def test_command_curriculum_underscore_and_hyphen_aliases(option):
    symm_cli = _load_symm_cli()

    args = symm_cli.build_parser().parse_args(["ablation", option])

    assert args.command_curriculum_mode == "tr_orbit_reward_threshold_v1"


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--curriculum-velocity-bins", "1", "greater than or equal to two"),
        ("--curriculum-ewma", "0", "curriculum_ewma must be finite"),
        ("--curriculum-unlock-threshold", "1.1", "curriculum_unlock_threshold must be finite"),
        ("--curriculum-initial-max-abs-speed", "-0.1", "initial_max_abs_speed must be finite"),
        ("--curriculum-min-visits", "0", "min_visits must be a positive integer"),
        ("--curriculum-current-cell-increment", "-0.1", "current_cell_increment must be finite"),
        ("--curriculum-neighbor-increment", "-0.1", "curriculum_neighbor_increment must be finite"),
        ("--curriculum-exploration-floor", "-0.1", "exploration_floor must be finite"),
        ("--curriculum-maximum-weight", "0", "maximum_weight must be finite"),
        ("--curriculum-locked-cell-weight", "-0.1", "locked_cell_weight must be finite"),
        ("--curriculum-seed", "-1", "curriculum_seed must be a nonnegative integer"),
    ],
)
def test_command_curriculum_rejects_invalid_values(option, value, message):
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", option, value])
    args.robot_spec = symm_cli.get_robot(args.robot)

    with pytest.raises(ValueError, match=message):
        symm_cli.train_lab_args(args, [])


def test_train_run_name_records_primary_history_and_curriculum_treatment():
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

    assert command[command.index("--run_name") + 1] == "go2_trseq_m0p2_v0p1_h30_cur0_seed42"


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
    assert any("_h30_cur0_seed42_linear_w0_r2000_linear" in line for line in command_lines)
    assert any("_h30_cur0_seed42_delayed_linear_w500_r1500_linear" in line for line in command_lines)
    assert any("_h30_cur0_seed42_half_cosine_w0_r2000_half_cosine" in line for line in command_lines)


def test_no_trs_disables_every_auxiliary_symmetry_training_path():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(["train", "--robot", "go2", "--no-conda-run", "--no-trs"])
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert "agent.algorithm.symmetry_cfg.use_data_augmentation=False" in command
    assert "agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false" in command
    assert "agent.algorithm.symmetry_cfg.use_mirror_loss=False" in command
    assert "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0" in command
    assert "agent.algorithm.symmetry_cfg.value_loss_coeff=0.0" in command


def test_train_forwards_reward_action_geometry_and_gait_options_directly():
    symm_cli = _load_symm_cli()
    args = symm_cli.build_parser().parse_args(
        [
            "train",
            "--robot",
            "go2",
            "--foot-phase-weight",
            "0.4",
            "--foot-phase-reduction",
            "mean",
            "--joint-target-limit-mode",
            "requested_overflow",
            "--joint-target-limit-weight",
            "0.075",
            "--actor-mean-bound-mode",
            "per_joint_feasible",
            "--tr-policy-output-space",
            "normalized_requested_joint_target",
            "--gait-sampling-profile",
            "trclosed_v2_halfbound_anneal",
            "--gait-curriculum-iterations",
            "4000",
            "--no-conda-run",
        ]
    )
    args.robot_spec = symm_cli.get_robot(args.robot)

    command = symm_cli.train_lab_args(args, [])

    assert "env.rewards.foot_phase.weight=0.4" in command
    assert "env.rewards.foot_phase.params.reduction=mean" in command
    assert "env.rewards.joint_target_limits.params.mode=requested_overflow" in command
    assert "env.rewards.joint_target_limits.weight=0.075" in command
    assert "agent.algorithm.symmetry_cfg.actor_mean_bound_mode=per_joint_feasible" in command
    assert "agent.algorithm.symmetry_cfg.tr_policy_output_space=normalized_requested_joint_target" in command
    assert "env.commands.base_velocity.gait_sampling_profile=trclosed_v2_halfbound_anneal" in command
    assert "env.commands.base_velocity.gait_curriculum_iterations=4000" in command


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


def test_compare_platform_launchers_are_deprecated():
    repo_root = Path(__file__).resolve().parents[3]
    script_dir = repo_root / "scripts" / "symm_locomotion"
    result = subprocess.run(
        [sys.executable, str(script_dir / "compare.py"), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "compare.py is deprecated" in result.stderr
    assert "symm_cli.py compare" in result.stderr
    assert "--robots" in result.stdout
    assert "Maximum runs shown per robot." in result.stdout
    bash = (script_dir / "compare.sh").read_text(encoding="utf-8")
    powershell = (script_dir / "compare.ps1").read_text(encoding="utf-8")
    assert "deprecated; use symm_locomotion.sh compare" in bash
    assert "deprecated; use symm_locomotion.ps1 compare" in powershell
    assert 'symm_cli.py compare "$@"' in bash
    assert '"symm_cli.py" "compare" @RemainingArgs' in powershell


def test_tensorboard_dry_run_uses_python_module(capsys):
    symm_cli = _load_symm_cli()

    result = symm_cli.main(["tensorboard", "--robots", "go2", "x1", "--dry-run", "--no-conda-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "-m tensorboard.main" in captured.out
    assert "logs" in captured.out
    assert "rsl_rl" in captured.out
