# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify recordings and modern metrics use only the saved training gaits."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _module(name, filename):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path, rows=((0.0, 0.0, 0.5, 0.5),)):
    checkpoint = tmp_path / "run" / "model_1499.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    params = checkpoint.parent / "params"
    params.mkdir(exist_ok=True)
    (params / "env.yaml").write_text(
        "commands:\n  base_velocity:\n    init_foot_thetas: !!python/tuple\n"
        + "".join("    - " + json.dumps(list(row)) + "\n" for row in rows)
        + "    gait_period: 0.6\n    duty_factor: 0.56\n    calculate_from_sampling_curve: false\n"
        + "policy_observation_history:\n  history_length: 20\n",
        encoding="utf-8",
    )
    return checkpoint


@pytest.mark.parametrize("group", ["bound", "trot", "halfbound", "rotarygallop", "traversegallop", "all"])
def test_trained_profile_contains_exactly_the_saved_rows(tmp_path, group):
    cli = _module("trained_cli_rows", "symm_cli.py")
    rows = (
        tuple(row for values in cli.TRAIN_GAIT_GROUPS.values() for row in values)
        if group == "all"
        else cli.TRAIN_GAIT_GROUPS[group]
    )
    checkpoint = _checkpoint(tmp_path, rows)
    analysis = _module("trained_analysis_rows", "evaluation.py")
    study = analysis.build_study(
        repo_root=Path(__file__).resolve().parents[3],
        checkpoint=checkpoint,
        robot="x1_go2bend",
        task=cli.get_robot("x1_go2bend").play_task,
        step_dt=0.02,
        protocol="trained",
        velocities_mps=(-1.0, 1.0),
    )
    assert {tuple(gait["phases"]) for gait in study["gaits"]} == set(rows)
    assert len(study["cells"]) == 2 * len(rows)
    assert study["evaluation_config"]
    assert study["method_version"] == "leg_usage_grid_trained_v1"
    assert study["protocol"] == "trained"


def test_trained_profile_rejects_untrained_canonical_index(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    analysis = _module("trained_analysis_invalid_index", "evaluation.py")
    with pytest.raises(ValueError, match="not.*train|untrained"):
        analysis.build_study(
            repo_root=tmp_path,
            checkpoint=checkpoint,
            robot="x1_go2bend",
            task="task",
            step_dt=0.02,
            protocol="trained",
            gait_indices=(0,),
        )


@pytest.mark.parametrize("mode, expected_video, expected_metrics", [("both", 1, 1), ("video", 1, 0), ("metrics", 0, 1)])
def test_unified_launcher_dispatches_bound_only_and_saved_timing(
    tmp_path, monkeypatch, mode, expected_video, expected_metrics
):
    checkpoint = _checkpoint(tmp_path)
    cli = _module("trained_cli_dispatch", "symm_cli.py")
    launches = []
    evaluations = []
    monkeypatch.setattr(cli, "run_isaaclab", lambda args, command: launches.append(command) or 0)
    monkeypatch.setattr(cli, "run_evaluation", lambda args, extra: evaluations.append((args, extra)) or 0)
    assert (
        cli.main(
            [
                "evaluate-trained",
                "--robot",
                "x1_go2bend",
                "--checkpoint",
                str(checkpoint),
                "--mode",
                mode,
                "--dry-run",
                "--no-conda-run",
            ]
        )
        == 0
    )
    assert len(launches) == expected_video
    assert len(evaluations) == expected_metrics
    for command in launches:
        assert command[command.index("--num_envs") + 1] == "1"
        assert "Isaac-Velocity-Flat-Dobot-X1-Go2Bend-Symm-Play-v0" in command
        assert "env.commands.base_velocity.init_foot_thetas=[[0.0,0.0,0.5,0.5]]" in command
        assert "env.commands.base_velocity.gait_period=0.6" in command
        assert "env.commands.base_velocity.duty_factor=0.56" in command
        assert "env.commands.base_velocity.add_noise_theta=False" in command
        assert "--video" in command
        assert "--symm_rollout_plots" in command
    for args, extra in evaluations:
        assert args.protocol == "trained"
        assert args.gait_indices == [1]
        assert args.num_envs == 100
        assert args.measure_s == 20.0
        assert args.settle_s == 5.0
        launch = cli.evaluation_lab_args(args, checkpoint, Path("study.json"), extra)
        assert launch[launch.index("--num_envs") + 1] == "100"
        assert "env.commands.base_velocity.gait_period=0.6" in extra
        assert "env.commands.base_velocity.duty_factor=0.56" in extra


def test_parallel_trained_study_keeps_100_trials_per_command_and_old_results_separate(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    analysis = _module("trained_analysis_parallel", "evaluation.py")
    kwargs = dict(
        repo_root=Path(__file__).resolve().parents[3],
        checkpoint=checkpoint,
        robot="x1",
        task="task",
        step_dt=0.02,
        protocol="trained",
        velocities_mps=(-1.0, 1.0),
        measure_s=20.0,
    )
    study = analysis.build_study(**kwargs, num_envs=100)
    old_study = analysis.build_study(**kwargs)
    assert study["num_envs"] == 100
    assert study["settle_steps"] == 250
    assert study["measure_steps"] == 1000
    assert study["total_steps"] == 1250
    assert study["episode_length_s"] == pytest.approx(25.04)
    assert len(study["cells"]) == 200
    assert len({cell["id"] for cell in study["cells"]}) == 200
    assert len({cell["relative_output_dir"] for cell in study["cells"]}) == 200
    for speed in (-1, 1):
        trials = [cell for cell in study["cells"] if cell["velocity_mps"] == speed]
        assert {cell["trial_index"] for cell in trials} == set(range(100))
        assert {cell["seed"] for cell in trials} == {42}
    assert "envs100_steps1000" in study["output_root"]
    assert old_study["output_root"] != study["output_root"]
    assert analysis._study_identity(study)["num_envs"] == 100
    assert "num_envs" not in analysis._study_identity(old_study)
    manifest_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    assert analysis.load_study_manifest(manifest_path)["num_envs"] == 100
    assert (
        analysis.existing_study_manifest_path(checkpoint.parent, "trained", num_envs=100, measure_steps=1000)
        == manifest_path
    )


def test_directional_metrics_match_each_replica_instead_of_overwriting_rows():
    analysis = _module("trained_analysis_parallel_pairs", "evaluation.py")
    rows = [
        dict(
            cell_id=f"{trial}_{speed}",
            gait_index=1,
            velocity_mps=speed,
            seed=42,
            trial_index=trial,
            family="bound",
            velocity_metric_valid=True,
            velocity_vx_bias_mps=trial * 0.01,
            velocity_vx_measured_mean_mps=speed + trial * 0.01,
        )
        for speed in (-1, 1)
        for trial in range(100)
    ]
    study = dict(
        gaits=[dict(index=1, time_reversal_partner=1)],
        evaluation_config=analysis._metrics.evaluation_config(),
    )
    pairs = analysis._directional_pair_metrics(study, rows)
    assert len(pairs) == 100
    for pair in pairs:
        assert pair["positive_cell_id"].split("_")[0] == pair["negative_cell_id"].split("_")[0]
        assert pair["positive_vx_bias_mps"] == pair["negative_vx_bias_mps"]


@pytest.mark.parametrize("gait", ["trot", "halfbound", "rotarygallop", "traversegallop"])
def test_unified_launcher_rejects_untrained_selection_before_launch(tmp_path, monkeypatch, gait):
    checkpoint = _checkpoint(tmp_path)
    cli = _module("trained_cli_reject", "symm_cli.py")
    launches = []
    monkeypatch.setattr(cli, "run_isaaclab", lambda *args: launches.append(args) or 0)
    assert (
        cli.main(
            [
                "evaluate-trained",
                "--robot",
                "x1_go2bend",
                "--checkpoint",
                str(checkpoint),
                "--gait",
                gait,
            ]
        )
        == 2
    )
    assert not launches


def test_missing_training_config_stops_before_launch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_1499.pt"
    checkpoint.write_bytes(b"checkpoint")
    cli = _module("trained_cli_missing_config", "symm_cli.py")
    launches = []
    monkeypatch.setattr(cli, "run_isaaclab", lambda *args: launches.append(args) or 0)
    assert cli.main(["evaluate-trained", "--checkpoint", str(checkpoint), "--dry-run"]) == 2
    assert not launches


def test_unified_launcher_rejects_gait_override_bypass(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    cli = _module("trained_cli_override", "symm_cli.py")
    assert (
        cli.main(
            [
                "evaluate-trained",
                "--checkpoint",
                str(checkpoint),
                "--dry-run",
                "env.commands.base_velocity.init_foot_thetas=[[0.0,0.5,0.5,0.0]]",
            ]
        )
        == 2
    )


@pytest.mark.parametrize("script", ["evaluation.sh", "test.sh"])
def test_root_bash_launchers_accept_variant_and_use_saved_gaits(tmp_path, script):
    checkpoint = _checkpoint(tmp_path)
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "bash",
            str(root / script),
            "--robot",
            "x1_go2bend",
            "--checkpoint",
            str(checkpoint),
            "--mode",
            "video",
            "--dry-run",
            "--no-conda-run",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Go2Bend-Symm-Play" in result.stdout
    assert "init_foot_thetas=[[0.0,0.0,0.5,0.5]]" in result.stdout
    assert "--video" in result.stdout
