# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fixed-grid policy evaluation utility and pure analysis layer."""

from __future__ import annotations

import csv
import hashlib
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
    path = Path(__file__).resolve().parents[1] / "evaluation.py"
    return _load_module("evaluation_under_test", path)


def _cli_module():
    path = Path(__file__).resolve().parents[1] / "symm_cli.py"
    return _load_module("symm_cli_leg_usage_under_test", path)


def test_deprecated_python_entry_point_reexports_evaluation():
    path = Path(__file__).resolve().parents[1] / "analyze_leg_usage.py"

    with pytest.warns(DeprecationWarning, match="use evaluation.py"):
        compatibility_module = _load_module("analyze_leg_usage_compatibility_test", path)

    assert callable(compatibility_module.build_study)
    assert callable(compatibility_module.analyze_study)


def test_deprecated_shell_wrappers_select_legacy_profile():
    script_root = Path(__file__).resolve().parents[1]

    powershell = (script_root / "analyze_leg_usage.ps1").read_text(encoding="utf-8")
    bash = (script_root / "analyze_leg_usage.sh").read_text(encoding="utf-8")

    assert 'evaluation.ps1" --protocol legacy @args' in powershell
    assert 'evaluation.sh" --protocol legacy "$@"' in bash


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
    study = module.build_study(**arguments)
    if "effort_limit_provenance" in study:
        study["effort_limit_provenance"]["source"] = {
            "sha256": "a" * 64,
            "files": {"synthetic_robot_config.py": "b" * 64},
            "missing_files": [],
        }
    study["study_identity_sha256"] = module._canonical_sha256(module._study_identity(study))
    return study, checkpoint


def _rewrite_study_manifest(module, study_path: Path, study: dict) -> None:
    """Rewrite a synthetic manifest after refreshing its self-consistency digest."""
    study["study_identity_sha256"] = module._canonical_sha256(module._study_identity(study))
    study_path.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_identity_fields(
    study: dict,
    cell: dict,
    sample_count: int,
    plan_sha256: str,
) -> dict[str, np.ndarray]:
    """Return strict archive identity/provenance fields for synthetic cells."""
    return {
        "protocol_version": np.asarray(study["method_version"]),
        "plan_sha256": np.asarray(plan_sha256),
        "cell_id": np.asarray(cell["id"]),
        "gait_index": np.asarray(cell["gait_index"]),
        "gait_name": np.asarray(cell["gait_name"]),
        "gait_family": np.asarray(cell["family"]),
        "evaluation_seed": np.asarray(cell["seed"]),
        "velocity_mps": np.asarray(cell["velocity_mps"]),
        "step_dt": np.asarray(study["step_dt"]),
        "expected_steps": np.asarray(cell["total_steps"]),
        "recorded_steps": np.asarray(sample_count),
        "configured_foot_thetas": np.asarray(cell["phases"]),
        "base_headings": np.zeros(sample_count),
        "desired_headings": np.zeros(sample_count),
        "heading_sample_valid": np.ones(sample_count, dtype=bool),
        "joint_names": np.asarray([f"joint_{index}" for index in range(12)]),
        "effort_limit_provenance_json": np.asarray(json.dumps(study["effort_limit_provenance"], sort_keys=True)),
        "ground_filter_paths": np.asarray(["/World/ground/terrain/mesh"] * 4),
    }


def _write_recording_manifest(study_path: Path, study: dict, cell: dict, archive_path: Path) -> None:
    """Write the deterministic runtime contact manifest used by publication analysis."""
    with np.load(archive_path, allow_pickle=False) as archive:
        manifest = {
            "schema_version": 1,
            "method_version": study["method_version"],
            "plan_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
            "cell_id": cell["id"],
            "gait_index": cell["gait_index"],
            "evaluation_seed": cell["seed"],
            "recorded_steps": len(archive["time_steps"]),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "robot_mass_kg": float(archive["robot_mass_kg"].item()),
            "contact_threshold_mode": study["evaluation_config"]["contact"]["mode"],
            "contact_threshold_on_n": float(archive["contact_threshold_on_n"].item()),
            "contact_threshold_off_n": float(archive["contact_threshold_off_n"].item()),
            "ground_filter_paths": [str(value) for value in archive["ground_filter_paths"].tolist()],
            "ground_filtered_samples": int(np.count_nonzero(archive["foot_normal_force_is_ground_filtered"])),
        }
    manifest["record_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    path = archive_path.with_name("recording_manifest.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_default_study_uses_exact_run_root_and_complete_training_grid(tmp_path):
    analysis = _analysis_module()

    study, checkpoint = _build_study(analysis, tmp_path)

    output_root = checkpoint.parent / "evaluations" / "leg_usage_grid_full_v3"
    assert Path(study["output_root"]) == output_root
    assert study["method_version"] == "leg_usage_grid_full_v3"
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


def test_full_protocol_rejects_custom_gait_or_velocity_subsets(tmp_path):
    analysis = _analysis_module()

    with pytest.raises(ValueError, match="all ten canonical gait rows"):
        _build_study(analysis, tmp_path, gait_indices=[6, 7])
    with pytest.raises(ValueError, match="declared six-velocity grid"):
        _build_study(analysis, tmp_path, velocities_mps=[-0.5, 0.5])


def test_legacy_profile_preserves_custom_v1_grid(tmp_path):
    analysis = _analysis_module()

    study, checkpoint = _build_study(
        analysis,
        tmp_path,
        protocol="legacy",
        velocities_mps=[-0.5, 1.0],
        gait_indices=[6, 7],
    )

    assert study["method_version"] == "leg_usage_grid_v1"
    assert study["protocol"] == "full"
    assert Path(study["output_root"]) == checkpoint.parent / "evaluations" / "leg_usage_grid"
    assert study["velocities_mps"] == [-0.5, 1.0]
    assert [gait["index"] for gait in study["gaits"]] == [6, 7]
    assert len(study["cells"]) == 4
    assert "evaluation_config" not in study


def test_full_v3_and_historical_v1_roots_coexist_and_v1_resumes(tmp_path):
    analysis = _analysis_module()
    legacy, checkpoint = _build_study(
        analysis,
        tmp_path,
        protocol="legacy",
        velocities_mps=[-0.5, 0.5],
        gait_indices=[6, 7],
    )
    legacy_path = analysis.prepare_study(legacy, resume=False, analyze_only=False, dry_run=False)
    historical = json.loads(legacy_path.read_text(encoding="utf-8"))
    historical["source_provenance"]["sha256"] = "historical-source"
    historical["study_identity_sha256"] = analysis._canonical_sha256(analysis._study_identity(historical))
    legacy_path.write_text(json.dumps(historical, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    full, _ = _build_study(analysis, tmp_path)
    full_path = analysis.prepare_study(full, resume=False, analyze_only=False, dry_run=False)
    resumed = analysis.validate_existing_legacy_study_for_resume(
        legacy_path,
        checkpoint=checkpoint,
        robot="go2",
        task="Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0",
        step_dt=0.02,
        settle_s=5.0,
        measure_s=10.0,
        evaluation_seed=42,
        velocities_mps=[-0.5, 0.5],
        gait_indices=[6, 7],
        render_cell_plots=False,
    )

    assert legacy_path.parent.name == "leg_usage_grid"
    assert full_path.parent.name == "leg_usage_grid_full_v3"
    assert legacy_path.is_file() and full_path.is_file()
    assert resumed["source_provenance"]["sha256"] == "historical-source"


def test_pre_split_full_v3_root_remains_discoverable_and_resumable(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    historical_root = Path(study["output_root"]).parent / "leg_usage_grid"
    study["output_root"] = str(historical_root)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)

    resolved = analysis.existing_study_manifest_path(Path(study["checkpoint"]["path"]).parent, "full")
    candidate, _ = _build_study(analysis, tmp_path)
    candidate["output_root"] = str(resolved.parent)
    resumed_path = analysis.prepare_study(candidate, resume=True, analyze_only=False, dry_run=False)

    assert resolved == study_path
    assert resumed_path == study_path
    assert not (historical_root.parent / "leg_usage_grid_full_v3").exists()


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


def test_manifest_loader_rejects_absolute_and_parent_cell_paths(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)

    for unsafe_path, message in (
        (str((tmp_path / "absolute-cell").resolve()), "must be relative"),
        ("../outside-cell", "escapes the study root"),
    ):
        study["cells"][0]["relative_output_dir"] = unsafe_path
        _rewrite_study_manifest(analysis, study_path, study)
        with pytest.raises(ValueError, match=message):
            analysis.load_study_manifest(study_path)


def test_manifest_loader_rejects_duplicate_resolved_cell_paths(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    study["cells"][1]["relative_output_dir"] = study["cells"][0]["relative_output_dir"]
    _rewrite_study_manifest(analysis, study_path, study)

    with pytest.raises(ValueError, match="must resolve uniquely"):
        analysis.load_study_manifest(study_path)


def test_manifest_loader_rejects_cell_path_through_escaping_symlink(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(analysis, tmp_path)
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    outside = tmp_path / "outside-study"
    outside.mkdir()
    link = study_path.parent / "linked-cells"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable on this platform: {exc}")
    study["cells"][0]["relative_output_dir"] = "linked-cells/cell"
    _rewrite_study_manifest(analysis, study_path, study)

    with pytest.raises(ValueError, match="escapes the study root"):
        analysis.load_study_manifest(study_path)


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


def test_configured_tracking_and_ground_filter_controls_are_effective():
    analysis = _analysis_module()
    config = analysis._metrics.evaluation_config(
        {
            "contact": {"ground_filtered_required": False},
            "tracking": {"vx_relative_error_limit": 0.1, "yaw_rmse_limit_radps": 0.2},
        }
    )

    assert (
        analysis._ground_filter_requirement_reason(
            config,
            ground_filter_declared=False,
            selected_ground_filtered=np.zeros(4, dtype=bool),
        )
        == ""
    )
    tracking = analysis._tracking_qualification(
        config,
        expected_velocity_mps=1.0,
        planar_rmse_mps=0.1,
        vx_relative_rmse=0.15,
        yaw_rmse_radps=0.1,
    )
    assert tracking["planar_tracking_success"] is True
    assert tracking["yaw_tracking_success"] is True
    assert tracking["vx_relative_tracking_success"] is False
    assert tracking["yaw_tracking_success_threshold_radps"] == pytest.approx(0.2)


def test_publication_aggregation_keeps_raw_grf_and_normalized_domains_independent():
    analysis = _analysis_module()
    cells = []
    rows = []
    for family in analysis.FAMILY_ORDER:
        cells.append({"id": family, "family": family, "velocity_mps": 1.0})
        row = {
            "status": "valid",
            "tracking_success": True,
            "raw_load_metric_valid": True,
            "grf_load_metric_valid": True,
            "normalized_load_metric_valid": False,
            "load_metric_valid": False,
            "velocity_metric_valid": True,
            "gait_metric_valid": True,
        }
        for metric in analysis.ALL_METRICS:
            row[f"{metric}_signed_imbalance_percent"] = 10.0
            row[f"{metric}_abs_imbalance_percent"] = 10.0
            row[f"{metric}_total_integral"] = 2.0
            row[f"{metric}_total_per_s"] = 1.0
            row[f"{metric}_total_per_directed_m"] = 1.0
        rows.append(row)
    study = {
        "method_version": analysis.FULL_METHOD_VERSION,
        "cells": cells,
        "velocities_mps": [1.0],
        "aggregation": {"overall": "equal family mean"},
    }

    family_rows, overall = analysis.aggregate_results(study, rows)

    trot = next(row for row in family_rows if row["family"] == "trot" and row["scope"] == "family")
    assert trot["absolute_work_valid_cells"] == 1
    assert trot["vertical_grf_impulse_valid_cells"] == 1
    assert trot["normalized_torque_utilization_valid_cells"] == 0
    assert overall["metrics"]["absolute_work"]["family_balanced_mean_abs_imbalance_percent"] == pytest.approx(10.0)
    assert overall["metrics"]["vertical_grf_impulse"]["family_balanced_mean_abs_imbalance_percent"] == pytest.approx(
        10.0
    )
    assert overall["metrics"]["normalized_torque_utilization"]["family_balanced_mean_abs_imbalance_percent"] is None


def test_synthetic_cell_uses_steady_window_and_rejects_negative_progress(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(
        analysis,
        tmp_path,
        settle_s=0.04,
        measure_s=0.08,
    )
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    plan_sha256 = hashlib.sha256(study_path.read_bytes()).hexdigest()
    cell = next(cell for cell in study["cells"] if cell["gait_index"] == 0 and cell["velocity_mps"] == -0.5)
    cell_dir = study_path.parent / cell["relative_output_dir"]
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
    contact_on_n, contact_off_n = analysis._metrics.resolve_contact_thresholds(study["evaluation_config"], 10.0)
    np.savez_compressed(
        cell_dir / "sim_data.npz",
        time_steps=np.arange(sample_count) * study["step_dt"],
        desired_lin_vel=commands,
        true_lin_vel=velocities,
        base_positions=positions,
        joint_torques=torques,
        joint_powers=torques,
        joint_effort_limits=np.full(12, 40.0),
        configured_joint_effort_limits=np.full(12, 40.0),
        configured_effort_limit_source_by_joint=np.asarray(["synthetic actuator config"] * 12),
        configured_effort_limit_fallback=np.asarray(False),
        configured_effort_limits_valid=np.asarray(True),
        foot_ground_reaction_forces_w=np.ones((sample_count, 4, 3)),
        foot_normal_forces_w=np.tile([0.0, 0.0, 100.0], (sample_count, 4, 1)),
        foot_normal_force_is_ground_filtered=np.ones(sample_count, dtype=bool),
        contact_threshold_on_n=np.asarray(contact_on_n),
        contact_threshold_off_n=np.asarray(contact_off_n),
        robot_mass_kg=np.asarray(10.0),
        episode_done=np.zeros(sample_count, dtype=bool),
        foot_thetas=np.tile(np.asarray(cell["phases"]), (sample_count, 1)),
        duty_factors=np.full(sample_count, 0.5),
        common_gait_phases=np.arange(sample_count) * 0.25,
        **_publication_identity_fields(study, cell, sample_count, plan_sha256),
    )
    _write_recording_manifest(study_path, study, cell, cell_dir / "sim_data.npz")

    row = analysis.analyze_cell(study, cell, study_path.parent)

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

    assert reason is None


def test_reanalysis_removes_stale_optional_metric_tables(tmp_path, monkeypatch):
    analysis = _analysis_module()
    study, _ = _build_study(
        analysis,
        tmp_path,
        protocol="legacy",
        velocities_mps=[0.5],
        gait_indices=[0],
    )
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    cell = study["cells"][0]
    optional_state = "valid"

    def fake_analyze_cell(_study, _cell, _study_root):
        row = {
            "cell_id": cell["id"],
            "gait_index": cell["gait_index"],
            "gait_name": cell["gait_name"],
            "family": cell["family"],
            "velocity_mps": cell["velocity_mps"],
            "seed": cell["seed"],
            "status": "analyzed",
            "reason": "",
            "relative_output_dir": cell["relative_output_dir"],
        }
        if optional_state == "valid":
            row.update(
                {
                    "_joint_metric_rows": [{"cell_id": cell["id"], "joint": "joint_0"}],
                    "_foot_metric_rows": [{"cell_id": cell["id"], "foot": "FL"}],
                    "_confusion_row": {
                        "classification_domain_valid": True,
                        "planned_row": "trot",
                        "classified_row": "trot",
                        "planned_family": "trot",
                        "classified_family": "trot",
                    },
                    "_synchronization_rows": [{"cell_id": cell["id"], "event": "touchdown"}],
                }
            )
        elif optional_state == "invalid_classification":
            row["_confusion_row"] = {
                "classification_domain_valid": False,
                "planned_row": "trot",
                "classified_row": "unavailable",
                "planned_family": "trot",
                "classified_family": "unavailable",
            }
        return row

    monkeypatch.setattr(analysis, "analyze_cell", fake_analyze_cell)
    monkeypatch.setattr(analysis, "aggregate_results", lambda _study, _rows: ([], {"coverage": {}}))
    monkeypatch.setattr(analysis, "_stratified_fidelity_metrics", lambda _study, _rows: [])
    monkeypatch.setattr(
        analysis,
        "_directional_pair_metrics",
        lambda _study, _rows: [{"pair": "synthetic"}] if optional_state == "valid" else [],
    )
    for name in ("_save_family_figure", "_save_overall_figure", "_save_coverage_figure", "_write_report"):
        monkeypatch.setattr(analysis, name, lambda *_args, **_kwargs: None)

    optional_paths = [
        study_path.parent / "metrics" / name
        for name in (
            "joint_metrics.csv",
            "foot_metrics.csv",
            "gait_classifications.csv",
            "gait_confusion_matrix.csv",
            "same_phase_pair_metrics.csv",
            "directional_pair_metrics.csv",
        )
    ]
    analysis.analyze_study(study_path)
    assert all(path.is_file() for path in optional_paths)

    optional_state = "invalid_classification"
    analysis.analyze_study(study_path)
    assert optional_paths[2].is_file()
    with optional_paths[2].open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "classification_domain_valid": "False",
                "planned_row": "trot",
                "classified_row": "unavailable",
                "planned_family": "trot",
                "classified_family": "unavailable",
            }
        ]
    assert not optional_paths[3].exists()
    assert all(not path.exists() for path in (*optional_paths[:2], *optional_paths[4:]))

    optional_state = "empty"
    analysis.analyze_study(study_path)
    assert all(not path.exists() for path in optional_paths)


@pytest.mark.parametrize("method_version", ["leg_usage_grid_full_v3", "leg_usage_grid_light_v2"])
def test_publication_reanalysis_refreshes_paired_diagnostics(tmp_path, monkeypatch, method_version):
    analysis = _analysis_module()
    study_path = tmp_path / "study.json"
    study_path.write_text("{}\n", encoding="utf-8")
    study = {
        "method_version": method_version,
        "study_identity_sha256": "synthetic-study-identity",
        "cells": [{"id": "cell_0"}],
    }
    row = {
        "cell_id": "cell_0",
        "gait_index": 0,
        "gait_name": "trot",
        "family": "trot",
        "velocity_mps": 0.5,
        "seed": 42,
        "status": "analyzed",
        "reason": "",
        "relative_output_dir": "cells/cell_0",
    }
    generation = 0

    class FakePairedAnalysis:
        @staticmethod
        def analyze_study_file(path):
            nonlocal generation
            generation += 1
            metrics_dir = path.parent / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            output_names = {
                "rows": "paired_tr_consistency.csv",
                "summaries": "paired_tr_consistency_summary.csv",
                "record": "paired_tr_consistency.json",
            }
            for name in output_names.values():
                (metrics_dir / name).write_text(f"generation-{generation}\n", encoding="utf-8")
            record_sha256 = hashlib.sha256((metrics_dir / output_names["record"]).read_bytes()).hexdigest()
            return {
                "method_version": "paired_tr_consistency_phase_v1",
                "rows": [{"status": "analyzed"}],
                "outputs": {**output_names, "record_sha256": record_sha256},
            }

    monkeypatch.setattr(analysis, "load_study_manifest", lambda _path: study)
    monkeypatch.setattr(analysis, "analyze_cell", lambda *_args: dict(row))
    monkeypatch.setattr(analysis, "aggregate_results", lambda _study, _rows: ([], {"coverage": {}}))
    monkeypatch.setattr(analysis, "_stratified_fidelity_metrics", lambda _study, _rows: [])
    monkeypatch.setattr(analysis, "_directional_pair_metrics", lambda _study, _rows: [])
    monkeypatch.setattr(analysis, "_load_paired_consistency_module", lambda: FakePairedAnalysis)
    for name in (
        "_save_family_figure",
        "_save_overall_figure",
        "_save_coverage_figure",
        "_save_screening_figure",
        "_write_screening_report",
        "_write_report",
    ):
        monkeypatch.setattr(analysis, name, lambda *_args, **_kwargs: None)

    analysis.analyze_study(study_path)
    metrics_dir = study_path.parent / "metrics"
    for name in (
        "paired_tr_consistency.csv",
        "paired_tr_consistency_summary.csv",
        "paired_tr_consistency.json",
    ):
        (metrics_dir / name).write_text("stale\n", encoding="utf-8")

    overall = analysis.analyze_study(study_path)

    assert generation == 2
    assert all(
        (metrics_dir / name).read_text(encoding="utf-8") == "generation-2\n"
        for name in (
            "paired_tr_consistency.csv",
            "paired_tr_consistency_summary.csv",
            "paired_tr_consistency.json",
        )
    )
    paired = overall["paired_tr_consistency"]
    assert paired["method_version"] == "paired_tr_consistency_phase_v1"
    assert paired["rows"] == 1
    assert paired["analyzed_rows"] == 1
    assert paired["outputs"]["record"] == "metrics/paired_tr_consistency.json"
    assert (
        paired["record_sha256"] == hashlib.sha256((metrics_dir / "paired_tr_consistency.json").read_bytes()).hexdigest()
    )


def test_analyze_study_writes_tables_report_and_family_figures(tmp_path):
    analysis = _analysis_module()
    study, _ = _build_study(
        analysis,
        tmp_path,
        settle_s=0.04,
        measure_s=2.4,
    )
    study_path = analysis.prepare_study(study, resume=False, analyze_only=False, dry_run=False)
    plan_sha256 = hashlib.sha256(study_path.read_bytes()).hexdigest()
    for cell in study["cells"]:
        cell_dir = study_path.parent / cell["relative_output_dir"]
        cell_dir.mkdir(parents=True)
        sample_count = cell["total_steps"]
        commands = np.zeros((sample_count, 3))
        commands[:, 0] = cell["velocity_mps"]
        positions = np.zeros((sample_count, 2))
        positions[:, 0] = np.arange(sample_count) * 0.01 * np.sign(cell["velocity_mps"])
        common_phase = np.arange(sample_count) * study["step_dt"] / 0.5
        foot_phase = np.remainder(common_phase[:, None] + np.asarray(cell["phases"])[None, :], 1.0)
        stance = foot_phase >= 0.5
        normal_forces = np.zeros((sample_count, 4, 3))
        normal_forces[:, :, 2] = np.where(stance, 100.0, 0.0)
        contact_on_n, contact_off_n = analysis._metrics.resolve_contact_thresholds(study["evaluation_config"], 10.0)
        np.savez_compressed(
            cell_dir / "sim_data.npz",
            time_steps=np.arange(sample_count) * study["step_dt"],
            desired_lin_vel=commands,
            true_lin_vel=commands,
            base_positions=positions,
            joint_torques=np.ones((sample_count, 12)),
            joint_powers=np.ones((sample_count, 12)),
            joint_effort_limits=np.full(12, 40.0),
            configured_joint_effort_limits=np.full(12, 40.0),
            configured_effort_limit_source_by_joint=np.asarray(["synthetic actuator config"] * 12),
            configured_effort_limit_fallback=np.asarray(False),
            configured_effort_limits_valid=np.asarray(True),
            foot_ground_reaction_forces_w=np.ones((sample_count, 4, 3)),
            foot_normal_forces_w=normal_forces,
            foot_normal_force_is_ground_filtered=np.ones(sample_count, dtype=bool),
            contact_threshold_on_n=np.asarray(contact_on_n),
            contact_threshold_off_n=np.asarray(contact_off_n),
            robot_mass_kg=np.asarray(10.0),
            episode_done=np.zeros(sample_count, dtype=bool),
            foot_thetas=np.tile(np.asarray(cell["phases"]), (sample_count, 1)),
            duty_factors=np.full(sample_count, 0.5),
            common_gait_phases=common_phase,
            **_publication_identity_fields(study, cell, sample_count, plan_sha256),
        )
        _write_recording_manifest(study_path, study, cell, cell_dir / "sim_data.npz")

    overall = analysis.analyze_study(study_path)

    assert overall["coverage"]["complete"] is True
    assert overall["coverage"]["velocity_only_success_domain_cells"] == 60
    assert overall["coverage"]["gait_only_success_domain_cells"] == 60
    assert overall["coverage"]["joint_velocity_and_gait_success_domain_cells"] == 60
    assert overall["coverage"]["velocity_only_success_coverage_fraction"] == pytest.approx(1.0)
    assert overall["coverage"]["gait_only_success_coverage_fraction"] == pytest.approx(1.0)
    assert overall["coverage"]["joint_velocity_and_gait_success_coverage_fraction"] == pytest.approx(1.0)
    assert overall["gait_classification"]["row_accuracy"] == pytest.approx(1.0)
    assert overall["gait_classification"]["family_accuracy"] == pytest.approx(1.0)
    assert overall["tracking_qualification"]["yaw"] == "yaw_tracking_rmse_radps <= 0.05"
    assert overall["metrics"]["normalized_torque_utilization"][
        "family_balanced_mean_abs_imbalance_percent"
    ] == pytest.approx(0.0)
    for relative_path in (
        "metrics/cell_metrics.csv",
        "metrics/family_metrics.csv",
        "metrics/overall_metrics.json",
        "metrics/stratified_fidelity.csv",
        "metrics/stratified_fidelity.json",
        "metrics/analysis_provenance.json",
        "metrics/coverage.csv",
        "metrics/REPORT.md",
        "metrics/SCREENING_REPORT.md",
        "metrics/joint_metrics.csv",
        "metrics/foot_metrics.csv",
        "metrics/gait_classifications.csv",
        "metrics/gait_confusion_matrix.csv",
        "metrics/same_phase_pair_metrics.csv",
        "metrics/paired_tr_consistency.csv",
        "metrics/paired_tr_consistency_summary.csv",
        "metrics/paired_tr_consistency.json",
        "figures/trot.svg",
        "figures/bound.svg",
        "figures/half_bound.svg",
        "figures/gallop.svg",
        "figures/overall.svg",
        "figures/coverage.svg",
        "figures/screening_report.svg",
    ):
        assert (study_path.parent / relative_path).is_file()

    provenance = json.loads((study_path.parent / "metrics" / "analysis_provenance.json").read_text(encoding="utf-8"))
    metrics_path = Path(analysis._metrics.__file__).resolve()
    assert provenance["metrics_path"] == str(metrics_path)
    assert provenance["metrics_sha256"] == hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert overall["analysis_provenance"]["metrics_sha256"] == provenance["metrics_sha256"]

    with (study_path.parent / "metrics" / "foot_metrics.csv").open(encoding="utf-8", newline="") as stream:
        foot_rows = list(csv.DictReader(stream))
    assert len(foot_rows) == 240
    assert {
        "agreement_boundary_excluded",
        "contact_f1",
        "measured_duty_factor",
        "duty_factor_abs_error",
        "touchdown_abs_error_p95_cycles",
        "liftoff_abs_error_p95_cycles",
        "touchdown_unmatched_fraction",
        "liftoff_unmatched_fraction",
    }.issubset(foot_rows[0])

    with (study_path.parent / "metrics" / "gait_confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
        confusion_rows = list(csv.DictReader(stream))
    assert sum(int(row["count"]) for row in confusion_rows if row["level"] == "row") == 60
    assert sum(int(row["count"]) for row in confusion_rows if row["level"] == "family") == 60
    assert all("cell_id" not in row for row in confusion_rows)

    with (study_path.parent / "metrics" / "same_phase_pair_metrics.csv").open(encoding="utf-8", newline="") as stream:
        pair_rows = list(csv.DictReader(stream))
    assert {row["event"] for row in pair_rows} == {"touchdown", "liftoff"}
    assert {
        "first_foot",
        "second_foot",
        "eligible_complete_cycles",
        "matched_complete_cycles",
        "unmatched_complete_cycles",
        "coverage_fraction",
        "mean_s",
        "mean_cycles",
        "p95_s",
        "p95_cycles",
    }.issubset(pair_rows[0])

    with (study_path.parent / "metrics" / "stratified_fidelity.csv").open(encoding="utf-8", newline="") as stream:
        stratified_rows = list(csv.DictReader(stream))
    assert {row["scope"] for row in stratified_rows} == {"row", "family", "velocity", "direction", "policy"}
    assert all(int(row["planned_cells"]) > 0 for row in stratified_rows)
    with (study_path.parent / "metrics" / "cell_metrics.csv").open(encoding="utf-8", newline="") as stream:
        cell_rows = list(csv.DictReader(stream))
    assert {
        "load_raw_torque_squared_FL_n2m2s",
        "load_absolute_work_FR_j",
        "load_normalized_torque_squared_RL_s",
        "load_vertical_grf_impulse_RR_ns",
    }.issubset(cell_rows[0])
    report = (study_path.parent / "metrics" / "REPORT.md").read_text(encoding="utf-8")
    screening = (study_path.parent / "metrics" / "SCREENING_REPORT.md").read_text(encoding="utf-8")
    assert "Grid: 60 planned protocol cells" in report
    assert "| FL |" in screening
    assert overall["paired_tr_consistency"]["method_version"] == "paired_tr_consistency_phase_v1"
    assert overall["paired_tr_consistency"]["rows"] == 60
    assert overall["paired_tr_consistency"]["analyzed_rows"] == 0

    first_cell = study["cells"][0]
    first_archive_path = study_path.parent / first_cell["relative_output_dir"] / "sim_data.npz"
    with np.load(first_archive_path, allow_pickle=False) as archive:
        original_archive = {name: archive[name] for name in archive.files}

    def rewrite_archive(fields, *, manifest=True):
        np.savez_compressed(first_archive_path, **fields)
        manifest_path = first_archive_path.with_name("recording_manifest.json")
        if manifest:
            _write_recording_manifest(study_path, study, first_cell, first_archive_path)
        elif manifest_path.exists():
            manifest_path.unlink()

    no_events = dict(original_archive)
    no_events["foot_normal_forces_w"] = np.zeros_like(no_events["foot_normal_forces_w"])
    rewrite_archive(no_events)
    insufficient_gait_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert insufficient_gait_row["velocity_metric_valid"] is True
    assert insufficient_gait_row["gait_metric_valid"] is False
    assert insufficient_gait_row["gait_success_domain_valid"] is False
    assert insufficient_gait_row["gait_only_success"] is None
    assert insufficient_gait_row["joint_velocity_and_gait_success"] is None
    assert insufficient_gait_row["gait_classification_available"] is False
    assert insufficient_gait_row["gait_row_correct"] is None

    invalid_heading = dict(original_archive)
    invalid_heading["heading_sample_valid"] = np.zeros_like(invalid_heading["heading_sample_valid"])
    rewrite_archive(invalid_heading)
    invalid_heading_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert invalid_heading_row["velocity_metric_valid"] is True
    assert invalid_heading_row["heading_metric_valid"] is False
    assert "authentic pre-reset" in invalid_heading_row["heading_metric_reason"]
    assert invalid_heading_row.get("velocity_heading_rmse_rad") is None

    missing_heading = {
        name: value
        for name, value in original_archive.items()
        if name not in {"base_headings", "desired_headings", "heading_sample_valid"}
    }
    rewrite_archive(missing_heading)
    missing_heading_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert missing_heading_row["velocity_metric_valid"] is True
    assert missing_heading_row["heading_metric_valid"] is False
    assert "unexpected shapes" in missing_heading_row["heading_metric_reason"]

    missing_manifest_archive = dict(original_archive)
    rewrite_archive(missing_manifest_archive, manifest=False)
    missing_manifest_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert missing_manifest_row["recording_manifest_valid"] is False
    assert missing_manifest_row["gait_metric_valid"] is False
    assert missing_manifest_row["raw_load_metric_valid"] is True
    assert missing_manifest_row["normalized_load_metric_valid"] is True
    assert missing_manifest_row["gait_only_success"] is None

    missing_ground_archive = {
        name: value
        for name, value in original_archive.items()
        if name
        not in {
            "foot_normal_forces_w",
            "foot_normal_force_is_ground_filtered",
            "ground_filter_paths",
            "contact_threshold_on_n",
            "contact_threshold_off_n",
        }
    }
    rewrite_archive(missing_ground_archive, manifest=False)
    missing_ground_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert missing_ground_row["velocity_metric_valid"] is True
    assert missing_ground_row["raw_load_metric_valid"] is True
    assert missing_ground_row["normalized_load_metric_valid"] is True
    assert missing_ground_row["grf_load_metric_valid"] is False
    assert missing_ground_row["gait_metric_valid"] is False
    assert "normal-force array is missing" in missing_ground_row["gait_metric_reason"]

    invalid_ground_force = dict(original_archive)
    invalid_ground_force["foot_normal_forces_w"] = invalid_ground_force["foot_normal_forces_w"].copy()
    invalid_ground_force["foot_normal_forces_w"][0, 0, 2] = np.nan
    rewrite_archive(invalid_ground_force)
    invalid_ground_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert invalid_ground_row["raw_load_metric_valid"] is True
    assert invalid_ground_row["grf_load_metric_valid"] is False
    assert invalid_ground_row["gait_metric_valid"] is False
    assert "non-finite" in invalid_ground_row["gait_metric_reason"]

    malformed_c1_cases = []
    missing_joint_names = {name: value for name, value in original_archive.items() if name != "joint_names"}
    malformed_c1_cases.append((missing_joint_names, "joint-name identity/order"))
    empty_limits = dict(original_archive)
    empty_limits["configured_joint_effort_limits"] = np.asarray([], dtype=float)
    malformed_c1_cases.append((empty_limits, "configured effort-limit vector"))
    nonnumeric_limits = dict(original_archive)
    nonnumeric_limits["configured_joint_effort_limits"] = np.asarray(["not-a-limit"] * 12)
    malformed_c1_cases.append((nonnumeric_limits, "configured effort-limit vector is malformed"))
    empty_sources = dict(original_archive)
    empty_sources["configured_effort_limit_source_by_joint"] = np.asarray([], dtype=str)
    malformed_c1_cases.append((empty_sources, "per-joint configured effort-limit sources"))
    malformed_provenance = dict(original_archive)
    malformed_provenance["effort_limit_provenance_json"] = np.asarray("{not-json")
    malformed_c1_cases.append((malformed_provenance, "provenance JSON is malformed"))
    for malformed_archive, expected_reason in malformed_c1_cases:
        rewrite_archive(malformed_archive)
        malformed_row = analysis.analyze_cell(study, first_cell, study_path.parent)
        assert malformed_row["raw_load_metric_valid"] is True
        assert malformed_row["grf_load_metric_valid"] is True
        assert malformed_row["normalized_load_metric_valid"] is False
        assert malformed_row["load_metric_valid"] is False
        assert malformed_row["load_absolute_work_total_j"] > 0.0
        assert expected_reason in malformed_row["load_metric_reason"]

    wrong_identity = dict(original_archive)
    wrong_identity["gait_name"] = np.asarray("not_the_planned_row")
    rewrite_archive(wrong_identity)
    wrong_identity_row = analysis.analyze_cell(study, first_cell, study_path.parent)
    assert wrong_identity_row["status"] == "invalid"
    assert "gait-row identity mismatch" in wrong_identity_row["reason"]


def test_cli_dry_run_builds_one_env_plan_without_writing_output(monkeypatch, tmp_path, capsys):
    cli = _cli_module()
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)

    result = cli.main(
        [
            "evaluation",
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
    assert not (checkpoint.parent / "evaluations" / "leg_usage_grid_full_v3").exists()


def test_training_context_uses_actual_platform_neutral_python_entrypoint(monkeypatch, tmp_path):
    cli = _cli_module()
    commands = []
    entrypoint = tmp_path / "scripts" / "symm_locomotion" / "train.py"
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli.sys, "argv", [str(entrypoint)])
    monkeypatch.setattr(cli, "run_isaaclab", lambda args, command: commands.append(command) or 0)

    result = cli.main(["train", "--robot", "x1", "--dry-run", "--no-conda-run"])

    assert result == 0
    context_index = commands[0].index("--symm_direct_launch_context")
    context = json.loads(commands[0][context_index + 1])
    assert context["interface"] == "scripts/symm_locomotion/train.py"
    assert context["argv"] == [
        "scripts/symm_locomotion/train.py",
        "--robot",
        "x1",
        "--dry-run",
        "--no-conda-run",
    ]
    assert "train.ps1" not in json.dumps(context)


def test_new_cli_options_present_snake_case_first_and_keep_hyphen_aliases():
    cli = _cli_module()
    parser = cli.build_parser()
    subparsers = next(action for action in parser._actions if action.choices and "train" in action.choices)
    train_parser = subparsers.choices["train"]
    actions = {action.dest: action for action in train_parser._actions}
    expected = {
        "foot_phase_weight": "--foot_phase_weight",
        "foot_phase_reduction": "--foot_phase_reduction",
        "joint_target_limit_mode": "--joint_target_limit_mode",
        "joint_target_limit_weight": "--joint_target_limit_weight",
        "actor_mean_bound_mode": "--actor_mean_bound_mode",
        "tr_policy_output_space": "--tr_policy_output_space",
        "gait_sampling_profile": "--gait_sampling_profile",
        "gait_curriculum_iterations": "--gait_curriculum_iterations",
    }
    for destination, canonical in expected.items():
        assert actions[destination].option_strings[0] == canonical
        assert canonical.replace("_", "-") in actions[destination].option_strings

    parsed = parser.parse_args(
        [
            "train",
            "--foot-phase-weight",
            "0.4",
            "--joint-target-limit-mode",
            "requested_overflow",
            "--gait-curriculum-iterations",
            "50",
        ]
    )
    assert parsed.foot_phase_weight == pytest.approx(0.4)
    assert parsed.joint_target_limit_mode == "requested_overflow"
    assert parsed.gait_curriculum_iterations == 50


def test_deprecated_cli_alias_routes_to_evaluation(monkeypatch, capsys):
    cli = _cli_module()
    calls = []
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda args, extra: calls.append((args.command, args.protocol, extra)) or 0,
    )

    result = cli.main(["analyze_leg_usage", "--robot", "go2", "--no-conda-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert calls == [("analyze_leg_usage", "legacy", [])]
    assert cli.build_parser().parse_args(["evaluation"]).protocol == "full"
    assert "'analyze_leg_usage' is deprecated; use 'evaluation'" in captured.err


def test_deprecated_cli_alias_defaults_to_legacy_and_keeps_custom_grid(monkeypatch, tmp_path, capsys):
    cli = _cli_module()
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)

    result = cli.main(
        [
            "analyze_leg_usage",
            "--robot",
            "go2",
            "--checkpoint",
            str(checkpoint),
            "--velocities",
            "-0.5",
            "0.5",
            "--gait_indices",
            "6",
            "7",
            "--dry-run",
            "--no-conda-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "profile: legacy; method: leg_usage_grid_v1" in captured.out
    assert "4 cells" in captured.out
    assert "leg_usage_grid\\study.json" in captured.out or "leg_usage_grid/study.json" in captured.out
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
        cli.validate_evaluation_runtime_overrides([protected_token])

    cli.validate_evaluation_runtime_overrides(["--device", "cuda:0", "env.sim.dt=0.005"])


def test_child_error_still_runs_partial_analysis(monkeypatch, tmp_path):
    cli = _cli_module()
    checkpoint = tmp_path / "run" / "model_19999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    study_path = checkpoint.parent / "evaluations" / "leg_usage_grid_full_v3" / "study.json"
    analyzed = []

    class FakeAnalysis:
        PROTOCOL_OUTPUT_ROOT_NAMES = {
            "full": "leg_usage_grid_full_v3",
            "light": "leg_usage_grid_light",
            "legacy": "leg_usage_grid",
        }

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
    monkeypatch.setattr(cli, "_load_evaluation_module", lambda: FakeAnalysis)
    monkeypatch.setattr(cli, "run_isaaclab", lambda args, command: 7)
    args = cli.build_parser().parse_args(["evaluation", "--robot", "go2", "--no-conda-run"])
    args.robot_spec = cli.get_robot(args.robot)

    result = cli.run_evaluation(args, [])

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
    monkeypatch.setattr(cli, "_load_evaluation_module", lambda: analysis)

    result = cli.main(
        [
            "evaluation",
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
