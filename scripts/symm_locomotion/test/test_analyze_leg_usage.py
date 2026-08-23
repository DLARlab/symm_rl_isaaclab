# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fixed-grid leg-usage utility and pure analysis layer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / "analyze_leg_usage.py"
    return _load_module("analyze_leg_usage_under_test", path)


def _cli_module():
    path = Path(__file__).resolve().parents[1] / "symm_cli.py"
    return _load_module("symm_cli_leg_usage_under_test", path)


def _build_study(module, tmp_path: Path, **overrides):
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-a")
    arguments = {
        "repo_root": tmp_path,
        "checkpoint": checkpoint,
        "robot": "go2",
        "task": "Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0",
        "step_dt": 0.02,
    }
    arguments.update(overrides)
    return module.build_study(**arguments), checkpoint


def test_default_study_uses_exact_run_root_and_complete_training_grid(tmp_path):
    analysis = _analysis_module()

    study, checkpoint = _build_study(analysis, tmp_path)

    output_root = checkpoint.parent / "evaluations" / "leg_usage_grid"
    assert Path(study["output_root"]) == output_root
    assert len(study["gaits"]) == 10
    assert study["velocities_mps"] == [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    assert len(study["cells"]) == 60
    assert study["settle_steps"] == 250
    assert study["measure_steps"] == 500
    assert study["total_steps"] == 750
    assert study["nominal_profile"] is True
    assert "model_19999" not in str(output_root.relative_to(checkpoint.parent))
    assert study["cells"][2]["relative_output_dir"].startswith("cells/gait_00_trot/vx_neg_0p5/")
    assert [gait["time_reversal_partner"] for gait in study["gaits"]] == [0, 1, 3, 2, 5, 4, 8, 9, 6, 7]


def test_study_gaits_are_derived_from_canonical_task_metadata():
    analysis = _analysis_module()
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

    version, gaits = analysis.canonical_training_gaits()

    assert version == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION
    assert tuple(tuple(gait["phases"]) for gait in gaits) == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS
    assert tuple(gait["name"] for gait in gaits) == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES
    assert tuple(gait["family"] for gait in gaits) == symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES
    assert tuple(gait["time_reversal_partner"] for gait in gaits) == (
        symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS
    )


def test_study_rejects_inexact_control_step_duration(tmp_path):
    analysis = _analysis_module()

    with pytest.raises(ValueError, match="exact multiple"):
        _build_study(analysis, tmp_path, settle_s=5.01)


def test_sparse_gait_selection_preserves_stable_library_indices(tmp_path):
    analysis = _analysis_module()

    study, _ = _build_study(analysis, tmp_path, gait_indices=[6, 7], velocities_mps=[-0.5, 0.5])

    assert [gait["index"] for gait in study["gaits"]] == [6, 7]
    assert [gait["name"] for gait in study["gaits"]] == ["gallop_a", "gallop_b"]
    assert {cell["gait_index"] for cell in study["cells"]} == {6, 7}
    assert len(study["cells"]) == 4


def test_existing_root_rejects_checkpoint_mismatch_even_with_resume(tmp_path):
    analysis = _analysis_module()
    study, checkpoint = _build_study(analysis, tmp_path)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    original_bytes = study_path.read_bytes()
    checkpoint.write_bytes(b"checkpoint-b")
    mismatched = analysis.build_study(
        repo_root=tmp_path,
        checkpoint=checkpoint,
        robot="go2",
        task="Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0",
        step_dt=0.02,
    )

    with pytest.raises(ValueError, match="different checkpoint or protocol"):
        analysis.prepare_study(mismatched, resume=True, analyze_only=False, dry_run=False)

    assert study_path.read_bytes() == original_bytes


def test_existing_root_rejects_runtime_override_mismatch(tmp_path):
    analysis = _analysis_module()
    study, checkpoint = _build_study(analysis, tmp_path)
    analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    mismatched = analysis.build_study(
        repo_root=tmp_path,
        checkpoint=checkpoint,
        robot="go2",
        task="Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0",
        step_dt=0.02,
        runtime_overrides=["env.sim.physx.bounce_threshold_velocity=0.1"],
    )

    with pytest.raises(ValueError, match="different checkpoint or protocol"):
        analysis.prepare_study(mismatched, resume=True, analyze_only=False, dry_run=False)

    assert study["runtime_overrides"] == []
    assert mismatched["runtime_overrides"] == ["env.sim.physx.bounce_threshold_velocity=0.1"]


def test_resume_keeps_runtime_plan_bytes_stable(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    original_bytes = study_path.read_bytes()

    resumed_path = analysis.prepare_study(study, resume=True, analyze_only=False, dry_run=False)

    assert resumed_path == study_path
    assert study_path.read_bytes() == original_bytes


def test_prepare_study_refuses_active_recording_lock(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    output_root = Path(study["output_root"])
    output_root.mkdir(parents=True)
    (output_root / ".recording.lock").write_text("active", encoding="utf-8")

    with pytest.raises(ValueError, match="already locked"):
        analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)


def test_family_balanced_summary_does_not_cancel_signed_imbalance():
    analysis = _analysis_module()
    cells = []
    rows = []
    for family_index, family in enumerate(analysis.FAMILY_ORDER):
        for sign_index, signed in enumerate((40.0, -40.0)):
            cell = {
                "id": f"{family}-{sign_index}",
                "family": family,
                "velocity_mps": float(-1 if sign_index == 0 else 1),
            }
            row = {"status": "valid", "tracking_success": True}
            for metric in analysis.ALL_METRICS:
                row[f"{metric}_signed_imbalance_percent"] = signed
                row[f"{metric}_abs_imbalance_percent"] = abs(signed)
                row[f"{metric}_total_integral"] = 10.0 + family_index
                row[f"{metric}_total_per_s"] = 2.0 + family_index
                row[f"{metric}_total_per_directed_m"] = 3.0 + family_index
            cells.append(cell)
            rows.append(row)
    study = {
        "cells": cells,
        "velocities_mps": [-1.0, 1.0],
        "aggregation": {"overall": "equal family mean"},
    }

    family_rows, overall = analysis.aggregate_results(study, rows)

    trot = next(row for row in family_rows if row["family"] == "trot" and row["scope"] == "family")
    assert trot["normalized_torque_utilization_mean_signed_imbalance_percent"] == pytest.approx(0.0)
    assert trot["normalized_torque_utilization_mean_abs_imbalance_percent"] == pytest.approx(40.0)
    assert overall["metrics"]["normalized_torque_utilization"][
        "family_balanced_mean_abs_imbalance_percent"
    ] == pytest.approx(40.0)
    assert overall["metrics"]["normalized_torque_utilization"][
        "tracking_qualified_family_balanced_mean_abs_imbalance_percent"
    ] == pytest.approx(40.0)
    assert overall["metrics"]["normalized_torque_utilization"][
        "tracking_qualified_family_balanced_mean_total_per_s"
    ] == pytest.approx(3.5)

    rows[0]["tracking_success"] = False
    _, tracking_incomplete = analysis.aggregate_results(study, rows)
    assert tracking_incomplete["coverage"]["complete"] is True
    assert (
        tracking_incomplete["metrics"]["normalized_torque_utilization"][
            "tracking_qualified_family_balanced_mean_abs_imbalance_percent"
        ]
        is None
    )


def test_synthetic_cell_uses_steady_window_and_rejects_negative_progress(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(
        analysis,
        tmp_path,
        velocities_mps=[-0.5],
        gait_indices=[0],
        settle_s=0.04,
        measure_s=0.08,
    )
    cell = study["cells"][0]
    cell_dir = Path(study["output_root"]) / cell["relative_output_dir"]
    cell_dir.mkdir(parents=True)
    sample_count = cell["total_steps"]
    torques = np.ones((sample_count, 12), dtype=np.float64)
    torques[:, :6] *= 2.0
    commands = np.zeros((sample_count, 3), dtype=np.float64)
    commands[:, 0] = -0.5
    velocities = commands.copy()
    velocities[:, 2] = 0.10
    positions = np.zeros((sample_count, 2), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.1, sample_count)  # Opposite the negative command.
    np.savez_compressed(
        cell_dir / "sim_data.npz",
        time_steps=np.arange(sample_count) * study["step_dt"],
        desired_lin_vel=commands,
        true_lin_vel=velocities,
        base_positions=positions,
        joint_torques=torques,
        joint_powers=torques,
        joint_effort_limits=np.full(12, 40.0),
        foot_ground_reaction_forces_w=np.ones((sample_count, 4, 3)),
        episode_done=np.zeros(sample_count, dtype=bool),
    )

    row = analysis.analyze_cell(study, cell, Path(study["output_root"]))

    assert row["sample_count"] == cell["measure_steps"]
    assert row["status"] == "nonpositive_progress"
    assert row["signed_directed_progress_m"] < 0.0
    assert row["integrated_body_x_progress_m"] > 0.0
    assert row["absolute_work_total_per_directed_m"] is None
    assert row["torque_squared_abs_imbalance_percent"] > 0.0
    assert row["planar_tracking_success"] is True
    assert row["yaw_tracking_success"] is False
    assert row["tracking_success"] is False
    assert row["tracking_success_threshold_mps"] == pytest.approx(0.175)
    assert row["yaw_tracking_success_threshold_radps"] == pytest.approx(0.05)


def test_complete_runtime_status_still_exposes_termination_outcome(tmp_path):
    analysis = _analysis_module()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "outcome": "terminated",
                "termination_terms": ["base_contact", "time_out"],
                "recorded_steps": 125,
            }
        ),
        encoding="utf-8",
    )
    (cell_dir / "metadata.json").write_text(json.dumps({"step_dt": 0.02}), encoding="utf-8")

    reason = analysis._runtime_failure(cell_dir)

    assert reason == "episode terminated (base_contact, time_out) at t=2.5 s"


def test_analyze_study_writes_tables_report_and_family_figures(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(
        analysis,
        tmp_path,
        velocities_mps=[0.5],
        gait_indices=[0, 1, 2, 6],
        settle_s=0.04,
        measure_s=0.08,
    )
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    for cell in study["cells"]:
        cell_dir = study_path.parent / cell["relative_output_dir"]
        cell_dir.mkdir(parents=True)
        sample_count = cell["total_steps"]
        commands = np.zeros((sample_count, 3))
        commands[:, 0] = 0.5
        positions = np.zeros((sample_count, 2))
        positions[:, 0] = np.arange(sample_count) * 0.01
        np.savez_compressed(
            cell_dir / "sim_data.npz",
            time_steps=np.arange(sample_count) * study["step_dt"],
            desired_lin_vel=commands,
            true_lin_vel=commands,
            base_positions=positions,
            joint_torques=np.ones((sample_count, 12)),
            joint_powers=np.ones((sample_count, 12)),
            joint_effort_limits=np.full(12, 40.0),
            foot_ground_reaction_forces_w=np.ones((sample_count, 4, 3)),
            episode_done=np.zeros(sample_count, dtype=bool),
            foot_thetas=np.tile(np.asarray(cell["phases"]), (sample_count, 1)),
        )

    overall = analysis.analyze_study(study_path)

    assert overall["coverage"]["complete"] is True
    assert overall["tracking_qualification"]["yaw"] == "yaw_tracking_rmse_radps <= 0.05"
    assert overall["metrics"]["normalized_torque_utilization"][
        "family_balanced_mean_abs_imbalance_percent"
    ] == pytest.approx(0.0)
    for relative_path in (
        "metrics/cell_metrics.csv",
        "metrics/family_metrics.csv",
        "metrics/overall_metrics.json",
        "metrics/analysis_provenance.json",
        "metrics/coverage.csv",
        "metrics/REPORT.md",
        "figures/trot.svg",
        "figures/bound.svg",
        "figures/half_bound.svg",
        "figures/gallop.svg",
        "figures/overall.svg",
        "figures/coverage.svg",
    ):
        assert (study_path.parent / relative_path).is_file()


def test_cli_dry_run_builds_one_env_plan_without_writing_output(monkeypatch, tmp_path, capsys):
    cli = _cli_module()
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)

    result = cli.main(
        [
            "analyze_leg_usage",
            "--robot",
            "x1",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
            "--no-conda-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "60 cells" in captured.out
    assert "--num_envs 1" in captured.out
    assert "--symm_leg_usage_plan" in captured.out
    assert not (checkpoint.parent / "evaluations" / "leg_usage_grid").exists()


@pytest.mark.parametrize(
    "protected_token",
    (
        "--symm_leg_usage_plan=other.json",
        "--task",
        "--checkpoint=other.pt",
        "--video",
        "--num-envs=64",
        "--num_envs",
        "--seed=7",
        "--rl-library=skrl",
        "--rl_library",
    ),
)
def test_cli_rejects_forwarded_options_that_escape_the_grid_plan(protected_token):
    cli = _cli_module()

    with pytest.raises(ValueError, match="does not allow overriding"):
        cli.validate_leg_usage_runtime_overrides([protected_token])

    cli.validate_leg_usage_runtime_overrides(["--device", "cuda:0", "env.sim.dt=0.005"])


def test_child_error_still_runs_partial_analysis(monkeypatch, tmp_path):
    cli = _cli_module()
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    study_path = checkpoint.parent / "evaluations" / "leg_usage_grid" / "study.json"
    analyzed = []

    class FakeAnalysis:
        @staticmethod
        def build_study(**kwargs):
            return {"gaits": [{}], "velocities_mps": [0.5], "cells": [{}]}

        @staticmethod
        def prepare_study(*args, **kwargs):
            return study_path

        @staticmethod
        def analyze_study(path):
            analyzed.append(path)
            return {"coverage": {"valid_cells": 0, "expected_cells": 1}}

    monkeypatch.setattr(cli, "resolve_checkpoint", lambda args: checkpoint)
    monkeypatch.setattr(cli, "_load_leg_usage_module", lambda: FakeAnalysis)
    monkeypatch.setattr(cli, "run_isaaclab", lambda args, command: 7)
    args = cli.build_parser().parse_args(["analyze_leg_usage", "--robot", "go2", "--no-conda-run"])
    args.robot_spec = cli.get_robot(args.robot)

    result = cli.run_analyze_leg_usage(args, [])

    assert result == 7
    assert analyzed == [study_path]


def test_analyze_only_accepts_historical_source_provenance(monkeypatch, tmp_path):
    analysis = _analysis_module()
    cli = _cli_module()
    study, checkpoint = _build_study(analysis, tmp_path)
    study["source_provenance"]["sha256"] = "historical-source-sha256"
    study["study_identity_sha256"] = analysis._canonical_sha256(analysis._study_identity(study))
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    analyzed = []
    monkeypatch.setattr(
        analysis,
        "analyze_study",
        lambda path: analyzed.append(path) or {"coverage": {"valid_cells": 0, "expected_cells": 60}},
    )
    monkeypatch.setattr(cli, "_load_leg_usage_module", lambda: analysis)

    result = cli.main(
        [
            "analyze_leg_usage",
            "--robot",
            "go2",
            "--checkpoint",
            str(checkpoint),
            "--analyze_only",
            "--no-conda-run",
        ]
    )

    assert result == 0
    assert analyzed == [study_path]
