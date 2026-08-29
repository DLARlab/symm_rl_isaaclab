# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "paired_tr_consistency.py"
    name = "_test_paired_tr_consistency"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _joint_signal(phase: np.ndarray) -> np.ndarray:
    indices = np.arange(12, dtype=np.float64)
    return 0.08 * indices + (0.2 + 0.01 * indices) * np.sin(2.0 * np.pi * phase[:, None] + 0.17 * indices)


def _foot_signal(phase: np.ndarray) -> np.ndarray:
    offsets = np.asarray((0.0, 0.13, 0.31, 0.47))
    return 30.0 + 3.0 * np.arange(4) + 5.0 * np.cos(2.0 * np.pi * phase[:, None] + offsets)


def _contacts(phase: np.ndarray, swing_ratio: float) -> np.ndarray:
    offsets = np.asarray((0.0, 0.13, 0.31, 0.47))
    return np.remainder(phase[:, None] + offsets, 1.0) >= swing_ratio


def _reflect_joints(values: np.ndarray, robot: str) -> np.ndarray:
    signs = np.asarray((1.0, -1.0, -1.0)) if robot == "x1" else np.ones(3)
    return (values.reshape(-1, 4, 3)[:, (2, 3, 0, 1), :] * signs).reshape(-1, 12)


def _write_trace(
    path: Path,
    phase: np.ndarray,
    *,
    source_phase: np.ndarray,
    robot: str,
    reflected: bool,
    decision_phase: np.ndarray | None = None,
    decision_source_phase: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    swing_ratio = 0.4
    decision_phase = phase if decision_phase is None else decision_phase
    decision_source_phase = source_phase if decision_source_phase is None else decision_source_phase
    joint_position = _joint_signal(source_phase)
    joint_velocity = -1.7 * _joint_signal(source_phase)
    actor_mean = 0.5 * _joint_signal(decision_source_phase)
    normalized_target = 0.75 * _joint_signal(decision_source_phase)
    torque = 2.0 * _joint_signal(source_phase)
    power = np.abs(3.0 * _joint_signal(source_phase))
    contacts = _contacts(source_phase, swing_ratio)
    grf_z = contacts * (100.0 + _foot_signal(source_phase))
    if reflected:
        joint_position = _reflect_joints(joint_position, robot)
        joint_velocity = _reflect_joints(joint_velocity, robot)
        actor_mean = _reflect_joints(actor_mean, robot)
        normalized_target = _reflect_joints(normalized_target, robot)
        torque = _reflect_joints(torque, robot)
        power = power.reshape(-1, 4, 3)[:, (2, 3, 0, 1), :].reshape(-1, 12)
        contacts = contacts[:, (2, 3, 0, 1)]
        grf_z = grf_z[:, (2, 3, 0, 1)]
    force = np.zeros((len(phase), 4, 3))
    force[..., 2] = grf_z
    joint_names = (
        tuple(f"{leg}_{role}_joint" for leg in ("FL", "FR", "RL", "RR") for role in ("hip", "thigh", "calf"))
        if robot == "go2"
        else tuple(
            f"joint_{leg}_{role}"
            for leg in ("front_left", "front_right", "rear_left", "rear_right")
            for role in ("abad", "thigh_pitch", "calf_pitch")
        )
    )
    foot_body_names = (
        ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        if robot == "go2"
        else (
            "link_front_left_foot",
            "link_front_right_foot",
            "link_rear_left_foot",
            "link_rear_right_foot",
        )
    )
    np.savez_compressed(
        path,
        common_gait_phases=phase,
        pre_decision_common_gait_phases=decision_phase,
        duty_factors=np.full(len(phase), 1.0 - swing_ratio),
        time_steps=np.arange(len(phase)) * 0.02,
        true_lin_vel=np.column_stack(
            (
                -(1.0 + 0.2 * np.sin(2.0 * np.pi * source_phase)),
                np.zeros(len(phase)),
                -(0.1 + 0.03 * np.cos(2.0 * np.pi * source_phase)),
            )
        ),
        joint_positions=joint_position,
        joint_velocities=joint_velocity,
        actor_means=actor_mean,
        normalized_requested_joint_targets=normalized_target,
        critic_values=(2.0 + 0.4 * np.cos(2.0 * np.pi * decision_source_phase))[:, None],
        joint_torques=torque,
        joint_powers=power,
        foot_normal_forces_w=force,
        foot_ground_reaction_forces_w=force,
        joint_names=np.asarray(joint_names),
        foot_body_names=np.asarray(foot_body_names),
        leg_names=np.asarray(("Front Left", "Front Right", "Rear Left", "Rear Right")),
        motor_role_names=np.asarray(("Hip/Abad", "Thigh", "Calf")),
        contact_threshold_on_n=np.asarray(50.0),
        contact_threshold_off_n=np.asarray(20.0),
    )


def _write_pair(tmp_path: Path, *, robot: str, reflected: bool, decision_phase_offset: float = 0.0):
    sample_count = 201
    source_phase = 0.07 + np.arange(sample_count) * 0.025
    target_phase = 0.31 + np.arange(sample_count) * 0.025
    swing_ratio = 0.4
    transformed_source_phase = np.remainder(swing_ratio - target_phase, 1.0)
    source_dir = Path("cells/source")
    target_dir = Path("cells/target")

    # The source archive is positive-direction data, so undo the sign convention
    # used by the target helper for base velocity/yaw and odd joint velocity.
    _write_trace(
        tmp_path / source_dir / "sim_data.npz",
        source_phase,
        source_phase=source_phase,
        robot=robot,
        reflected=False,
        decision_phase=source_phase - decision_phase_offset,
        decision_source_phase=source_phase - decision_phase_offset,
    )
    with np.load(tmp_path / source_dir / "sim_data.npz", allow_pickle=False) as original:
        values = {name: np.asarray(original[name]) for name in original.files}
    values["true_lin_vel"][:, (0, 2)] *= -1.0
    values["joint_velocities"] *= -1.0
    np.savez_compressed(tmp_path / source_dir / "sim_data.npz", **values)

    _write_trace(
        tmp_path / target_dir / "sim_data.npz",
        target_phase,
        source_phase=transformed_source_phase,
        robot=robot,
        reflected=reflected,
        decision_phase=target_phase - decision_phase_offset,
        decision_source_phase=np.remainder(swing_ratio - (target_phase - decision_phase_offset), 1.0),
    )
    study = {
        "method_version": "leg_usage_grid_full_v3",
        "protocol": "full",
        "robot": robot,
        "step_dt": 0.02,
        "gaits": [
            {
                "index": 0,
                "name": "test_gait",
                "family": "trot",
                "time_reversal_partner": 0,
            }
        ],
        "cells": [
            {
                "id": "positive",
                "gait_index": 0,
                "gait_name": "test_gait",
                "family": "trot",
                "velocity_mps": 1.0,
                "seed": 42,
                "settle_steps": 0,
                "measure_steps": sample_count,
                "relative_output_dir": source_dir.as_posix(),
            },
            {
                "id": "negative",
                "gait_index": 0,
                "gait_name": "test_gait",
                "family": "trot",
                "velocity_mps": -1.0,
                "seed": 42,
                "settle_steps": 0,
                "measure_steps": sample_count,
                "relative_output_dir": target_dir.as_posix(),
            },
        ],
    }
    return study


def _bind_publication_recordings(study_path: Path, study: dict) -> None:
    """Bind synthetic archives to the publication study like the rollout writer."""
    plan_sha256 = hashlib.sha256(study_path.read_bytes()).hexdigest()
    for cell in study["cells"]:
        archive_path = study_path.parent / cell["relative_output_dir"] / "sim_data.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        recorded_steps = len(values["common_gait_phases"])
        settle_steps = int(cell.get("settle_steps", study.get("settle_steps", 0)))
        measure_steps = int(cell.get("measure_steps", study.get("measure_steps", recorded_steps - settle_steps)))
        expected_steps = int(cell.get("total_steps", settle_steps + measure_steps))
        values.update(
            {
                "protocol_version": np.asarray(study["method_version"]),
                "plan_sha256": np.asarray(plan_sha256),
                "cell_id": np.asarray(cell["id"]),
                "gait_index": np.asarray(cell["gait_index"]),
                "evaluation_seed": np.asarray(cell.get("seed", 0)),
                "velocity_mps": np.asarray(cell["velocity_mps"]),
                "step_dt": np.asarray(cell.get("step_dt", study.get("step_dt", 0.02))),
                "expected_steps": np.asarray(expected_steps),
                "recorded_steps": np.asarray(recorded_steps),
                "settle_steps": np.asarray(settle_steps),
                "measure_steps": np.asarray(measure_steps),
                "measurement_start_step": np.asarray(settle_steps),
                "measurement_stop_step": np.asarray(settle_steps + measure_steps),
                "checkpoint_json": np.asarray(json.dumps(study.get("checkpoint", {}), sort_keys=True)),
            }
        )
        np.savez_compressed(archive_path, **values)
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "method_version": study["method_version"],
            "plan_sha256": plan_sha256,
            "cell_id": cell["id"],
            "gait_index": cell["gait_index"],
            "evaluation_seed": cell.get("seed", 0),
            "recorded_steps": recorded_steps,
            "archive_sha256": archive_sha256,
        }
        manifest["record_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        archive_path.with_name("recording_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def test_phase_profile_aligns_phase_shifted_wall_clock_samples():
    module = _load_module()
    phase = 0.37 + np.arange(241) * 0.025
    signal = np.sin(2.0 * np.pi * phase)
    requested = np.arange(64) / 64.0
    profile = module.phase_profile(phase, signal, requested)
    np.testing.assert_allclose(profile, np.sin(2.0 * np.pi * requested), atol=3.5e-3)


def test_paired_forward_backward_evaluation_uses_declared_phase_partner(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    payload = module.analyze_paired_study(study, tmp_path, phase_samples=64)

    assert len(payload["rows"]) == 2
    pure = next(row for row in payload["rows"] if row["operator"] == "temporal_reversal")
    for metric in module.METRICS:
        assert pure[f"{metric}_valid"] is True
        tolerance = {
            "paired_contact_pattern_error": 0.035,
            "paired_touchdown_phase_error": 0.035,
            # A discontinuous stance-only force trace is interpolated from two
            # intentionally offset wall-clock sample grids.  Keep this bound
            # tight relative to its roughly 130 N stance magnitude.
            "paired_grf_pattern_error": 12.0,
        }.get(metric, 0.01)
        assert pure[metric] < tolerance, (metric, pure[metric])
    assert {row["scope"] for row in payload["summaries"]} == {
        "overall",
        "family",
        "direction",
        "speed",
        "robot",
    }
    assert "do not prove exact" in payload["caveat"]


def test_policy_metrics_use_pre_decision_phase_not_post_step_phase(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False, decision_phase_offset=0.08)

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=64)
    corrected = next(row for row in payload["rows"] if row["operator"] == "temporal_reversal")

    for metric in (
        "paired_actor_mean_consistency_error",
        "paired_normalized_target_consistency_error",
        "paired_value_consistency_error",
    ):
        assert corrected[f"{metric}_valid"] is True
        assert corrected[metric] < 0.01
    for metric in (
        "paired_joint_position_error",
        "paired_joint_velocity_error",
        "paired_torque_pattern_error",
        "paired_work_pattern_error",
    ):
        assert corrected[metric] < 0.02

    for relative_dir in ("cells/source", "cells/target"):
        path = tmp_path / relative_dir / "sim_data.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        values["pre_decision_common_gait_phases"] = values["common_gait_phases"]
        np.savez_compressed(path, **values)

    misaligned_payload = module.analyze_paired_study(study, tmp_path, phase_samples=64)
    misaligned = next(row for row in misaligned_payload["rows"] if row["operator"] == "temporal_reversal")
    assert misaligned["paired_actor_mean_consistency_error"] > 0.02
    assert misaligned["paired_normalized_target_consistency_error"] > 0.02
    assert misaligned["paired_value_consistency_error"] > 0.02
    assert misaligned["paired_joint_position_error"] < 0.02
    assert misaligned["paired_joint_velocity_error"] < 0.02


def test_x1_front_hind_reflected_temporal_operator_is_reported_separately(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="x1", reflected=True)
    payload = module.analyze_paired_study(study, tmp_path, phase_samples=64)
    by_operator = {row["operator"]: row for row in payload["rows"]}
    pure = by_operator["temporal_reversal"]
    composed = by_operator["front_hind_reflected_temporal_reversal"]

    assert pure["paired_joint_position_error"] > 0.1
    assert pure["paired_contact_pattern_error"] > 0.1
    assert composed["paired_joint_position_error"] < 0.01
    assert composed["paired_joint_velocity_error"] < 0.02
    assert composed["paired_contact_pattern_error"] < 0.035
    assert composed["paired_touchdown_phase_error"] < 0.035


def test_reordered_archived_axes_are_rejected_before_reflection(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    path = tmp_path / "cells/target/sim_data.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    joint_permutation = np.asarray((3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11))
    for field in (
        "joint_positions",
        "joint_velocities",
        "actor_means",
        "normalized_requested_joint_targets",
        "joint_torques",
        "joint_powers",
    ):
        values[field] = values[field][:, joint_permutation]
    values["joint_names"] = values["joint_names"][joint_permutation]
    np.savez_compressed(path, **values)

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=32)

    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("joint_names" in row["reason"] for row in payload["rows"])
    assert all(row["paired_joint_position_error"] is None for row in payload["rows"])


def test_light_full_hook_writes_paired_rows_summaries_and_provenance(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    study["protocol"] = "light"
    study["method_version"] = "leg_usage_grid_light_v2"
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    _bind_publication_recordings(study_path, study)

    payload = module.analyze_study_file(study_path, phase_samples=32)

    assert payload["study_method_version"] == "leg_usage_grid_light_v2"
    assert payload["input_study"]["sha256"]
    assert payload["analyzer"]["sha256"] == module._sha256_file(Path(module.__file__))
    assert len(payload["input_archives"]) == 2
    assert all(item["sha256"] for item in payload["input_archives"])
    assert payload["archive_identity"] == {"required": True, "invalid_cells": {}, "valid": True}
    assert set(payload["outputs"]) == {
        "rows",
        "rows_sha256",
        "summaries",
        "summaries_sha256",
        "record",
        "record_sha256",
    }
    for filename in (
        "paired_tr_consistency.csv",
        "paired_tr_consistency_summary.csv",
        "paired_tr_consistency.json",
    ):
        assert (tmp_path / "metrics" / filename).is_file()


def test_file_analysis_rejects_an_archive_swapped_between_cells(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    _bind_publication_recordings(study_path, study)
    source_path = tmp_path / study["cells"][0]["relative_output_dir"] / "sim_data.npz"
    target_path = tmp_path / study["cells"][1]["relative_output_dir"] / "sim_data.npz"
    source_path.write_bytes(target_path.read_bytes())

    payload = module.analyze_study_file(study_path, phase_samples=32)

    assert payload["archive_identity"]["valid"] is False
    assert study["cells"][0]["id"] in payload["archive_identity"]["invalid_cells"]
    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("cell identity" in row["reason"] for row in payload["rows"])


def test_file_analysis_reports_missing_publication_archive_as_invalid(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    _bind_publication_recordings(study_path, study)
    missing_cell = study["cells"][0]
    archive_path = tmp_path / missing_cell["relative_output_dir"] / "sim_data.npz"
    archive_path.unlink()

    payload = module.analyze_study_file(study_path, phase_samples=32)

    assert payload["archive_identity"]["valid"] is False
    assert payload["archive_identity"]["invalid_cells"] == {missing_cell["id"]: "publication archive is missing"}
    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("publication archive is missing" in row["reason"] for row in payload["rows"])


def test_paired_analysis_rejects_asymmetric_measurement_windows(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    study["cells"][1]["measure_steps"] -= 1

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=32)

    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("paired rollout windows differ" in row["reason"] for row in payload["rows"])
    assert all("measure_steps" in row["reason"] for row in payload["rows"])


def test_cell_archive_path_cannot_escape_study_directory(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    study["cells"][0]["relative_output_dir"] = "../outside"

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=32)

    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("escapes the study directory" in row["reason"] for row in payload["rows"])


def test_normalized_target_metric_never_substitutes_clamped_execution_target(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    for relative_dir in ("cells/source", "cells/target"):
        path = tmp_path / relative_dir / "sim_data.npz"
        with np.load(path, allow_pickle=False) as archive:
            values = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "normalized_requested_joint_targets"
            }
        values["joint_position_targets"] = np.zeros((len(values["common_gait_phases"]), 12))
        np.savez_compressed(path, **values)

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=32)
    for row in payload["rows"]:
        assert row["paired_normalized_target_consistency_error"] is None
        assert row["paired_normalized_target_consistency_error_valid"] is False
        assert "deliberately not substituted" in row["paired_normalized_target_consistency_error_reason"]


def test_normalized_target_metric_accepts_requested_targets_with_static_limits(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    lower = np.linspace(-2.0, -0.9, 12)
    upper = np.linspace(0.8, 1.9, 12)
    for relative_dir in ("cells/source", "cells/target"):
        path = tmp_path / relative_dir / "sim_data.npz"
        with np.load(path, allow_pickle=False) as archive:
            normalized = np.asarray(archive["normalized_requested_joint_targets"])
            values = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "normalized_requested_joint_targets"
            }
        values["requested_joint_position_targets"] = 0.5 * (normalized * (upper - lower) + upper + lower)
        values["joint_position_lower_limits"] = lower
        values["joint_position_upper_limits"] = upper
        np.savez_compressed(path, **values)

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=64)
    pure = next(row for row in payload["rows"] if row["operator"] == "temporal_reversal")
    assert pure["paired_normalized_target_consistency_error_valid"] is True
    assert pure["paired_normalized_target_consistency_error"] < 0.01
    assert pure["paired_value_consistency_error_valid"] is True


def test_terminated_measurement_window_is_fail_closed(tmp_path):
    module = _load_module()
    study = _write_pair(tmp_path, robot="go2", reflected=False)
    path = tmp_path / "cells/target/sim_data.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["episode_done"] = np.zeros(len(values["common_gait_phases"]), dtype=bool)
    values["episode_done"][20] = True
    np.savez_compressed(path, **values)

    payload = module.analyze_paired_study(study, tmp_path, phase_samples=32)

    assert all(row["status"] == "unavailable" for row in payload["rows"])
    assert all("terminated during" in row["reason"] for row in payload["rows"])
