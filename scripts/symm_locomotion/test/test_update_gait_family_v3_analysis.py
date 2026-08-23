# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fixed-grid gait-family comparison updater."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "update_gait_family_v3_analysis.py"
    spec = importlib.util.spec_from_file_location("update_gait_family_v3_analysis_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol_study(run_dir: Path, velocity: float = 0.5) -> dict:
    families = ("trot", "bound", "half_bound", "gallop")
    gaits = [
        {
            "index": index,
            "name": f"gait_{index}",
            "family": family,
            "phases": [0.0, 0.5, 0.5, 0.0],
            "training_weight": 0.1,
            "weight": 0.1,
            "time_reversal_partner": index,
        }
        for index, family in enumerate(families)
    ]
    return {
        "schema_version": 1,
        "method_version": "leg_usage_grid_v1",
        "robot": "go2",
        "task": "test-task",
        "gait_library_version": "test-gaits",
        "checkpoint": {
            "path": str(run_dir / "model_19999.pt"),
            "iteration": 19999,
            "sha256": "checkpoint-sha",
        },
        "study_identity_sha256": f"identity-{run_dir.name}",
        "step_dt": 0.02,
        "settle_s": 5.0,
        "measure_s": 10.0,
        "settle_steps": 250,
        "measure_steps": 500,
        "total_steps": 750,
        "evaluation_seed": 42,
        "nominal_profile": True,
        "runtime_overrides": [],
        "velocities_mps": [velocity],
        "gaits": gaits,
        "cells": [
            {
                "id": f"cell_{index}",
                "gait_index": index,
                "gait_name": f"gait_{index}",
                "family": family,
                "phases": [0.0, 0.5, 0.5, 0.0],
                "velocity_mps": velocity,
                "seed": 42,
                "step_dt": 0.02,
                "settle_steps": 250,
                "measure_steps": 500,
                "total_steps": 750,
            }
            for index, family in enumerate(families)
        ],
        "source_provenance": {"sha256": "source-sha"},
        "aggregation": {
            "within_family": "equal",
            "overall": "family balanced",
            "primary_balance_metric": "mean absolute imbalance",
            "tracking_qualified": "reported separately",
        },
    }


def _metric_result(base: float) -> dict:
    family_values = {
        "trot": base,
        "bound": base + 1.0,
        "half_bound": base + 2.0,
        "gallop": base + 3.0,
    }
    family_rates = {family: value * 2.0 for family, value in family_values.items()}
    return {
        "complete": True,
        "family_balanced_mean_abs_imbalance_percent": base + 1.5,
        "observed_family_balanced_mean_abs_imbalance_percent": base + 1.5,
        "tracking_qualified_observed_family_balanced_mean_abs_imbalance_percent": base + 1.0,
        "tracking_qualified_observed_family_balanced_mean_total_per_s": base * 2.5,
        "tracking_qualified_observed_family_balanced_mean_total_per_directed_m": base * 3.5,
        "family_balanced_mean_total_per_s": base * 3.0,
        "family_balanced_mean_total_per_directed_m": base * 4.0,
        "family_mean_abs_imbalance_percent": family_values,
        "family_mean_total_per_directed_m": family_rates,
    }


def _write_grid_result(module, run_dir: Path, study: dict, offset: float, valid_cells: int = 4) -> None:
    grid_root = run_dir / "evaluations" / "leg_usage_grid"
    metrics_dir = grid_root / "metrics"
    metrics_dir.mkdir(parents=True)
    study_path = grid_root / "study.json"
    study_path.write_text(json.dumps({"placeholder": run_dir.name}), encoding="utf-8")
    study_sha = module._sha256_file(study_path)
    provenance = {
        "input_study_sha256": study_sha,
        "input_study_identity_sha256": study["study_identity_sha256"],
        "analyzer_sha256": "analyzer-sha",
    }
    provenance["record_sha256"] = module._canonical_sha256(provenance)
    overall = {
        "schema_version": 1,
        "method_version": "leg_usage_grid_v1",
        "coverage": {
            "expected_cells": 4,
            "valid_cells": valid_cells,
            "failed_or_missing_cells": 4 - valid_cells,
            "complete": valid_cells == 4,
            "tracking_success_cells": valid_cells,
            "tracking_success_fraction_of_valid": 1.0 if valid_cells else None,
            "status_counts": {"valid": 4} if valid_cells == 4 else {"failed": 1, "valid": 3},
        },
        "analysis_provenance": dict(provenance),
        "metrics": {
            "torque_squared": _metric_result(10.0 + offset),
            "absolute_work": _metric_result(20.0 + offset),
            "vertical_grf_impulse": _metric_result(30.0 + offset),
        },
    }
    (metrics_dir / "analysis_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (metrics_dir / "overall_metrics.json").write_text(json.dumps(overall), encoding="utf-8")
    (grid_root / "progress.json").write_text(
        json.dumps({"status": "complete", "total_cells": 4, "completed_cells": 4}), encoding="utf-8"
    )
    if valid_cells != 4:
        for result in overall["metrics"].values():
            result["complete"] = False
            result["family_balanced_mean_abs_imbalance_percent"] = None
        (metrics_dir / "overall_metrics.json").write_text(json.dumps(overall), encoding="utf-8")
    family_fields = [
        "scope",
        "family",
        "expected_cells",
        "valid_cells",
        "complete",
        "tracking_success_cells",
        "tracking_qualified_complete",
    ]
    for metric in module.PRIMARY_METRICS:
        family_fields.extend(
            [
                f"{metric}_mean_abs_imbalance_percent",
                f"{metric}_mean_total_per_directed_m",
                f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent",
                f"{metric}_tracking_qualified_observed_mean_total_per_s",
                f"{metric}_tracking_qualified_observed_mean_total_per_directed_m",
            ]
        )
    with (metrics_dir / "family_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=family_fields)
        writer.writeheader()
        for index, family in enumerate(("trot", "bound", "half_bound", "gallop")):
            valid = index < valid_cells
            row = {
                "scope": "family",
                "family": family,
                "expected_cells": 1,
                "valid_cells": int(valid),
                "complete": valid,
                "tracking_success_cells": int(valid),
                "tracking_qualified_complete": valid,
            }
            for metric_index, metric in enumerate(module.PRIMARY_METRICS):
                value = 10.0 * (metric_index + 1) + offset + index
                row[f"{metric}_mean_abs_imbalance_percent"] = value if valid else ""
                row[f"{metric}_mean_total_per_directed_m"] = value * 2.0 if valid else ""
                row[f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"] = value if valid else ""
                row[f"{metric}_tracking_qualified_observed_mean_total_per_s"] = value * 3.0 if valid else ""
                row[f"{metric}_tracking_qualified_observed_mean_total_per_directed_m"] = value * 2.0 if valid else ""
            writer.writerow(row)
    cell_fields = [
        "cell_id",
        "family",
        "status",
        "reason",
        "planar_tracking_success",
        "yaw_tracking_success",
        "tracking_success",
        "tracking_rmse_mps",
        "yaw_tracking_rmse_radps",
        "effort_limit_min_nm",
        "effort_limit_max_nm",
    ]
    for metric in module.PRIMARY_METRICS:
        cell_fields.extend(
            [f"{metric}_abs_imbalance_percent", f"{metric}_total_per_s", f"{metric}_total_per_directed_m"]
        )
    coverage_rows = []
    with (metrics_dir / "cell_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        for index, cell in enumerate(study["cells"]):
            valid = index < valid_cells
            row = {
                "cell_id": cell["id"],
                "family": cell["family"],
                "status": "valid" if valid else "failed",
                "reason": "" if valid else "episode ended at sample 10",
                "planar_tracking_success": valid,
                "yaw_tracking_success": valid,
                "tracking_success": valid,
                "tracking_rmse_mps": 0.05 if valid else "",
                "yaw_tracking_rmse_radps": 0.04 if valid else "",
                "effort_limit_min_nm": 1.0e9 if valid else "",
                "effort_limit_max_nm": 1.0e9 if valid else "",
            }
            for metric_index, metric in enumerate(module.PRIMARY_METRICS):
                value = 10.0 * (metric_index + 1) + offset + index
                row[f"{metric}_abs_imbalance_percent"] = value if valid else ""
                row[f"{metric}_total_per_s"] = value * 3.0 if valid else ""
                row[f"{metric}_total_per_directed_m"] = value * 2.0 if valid else ""
            writer.writerow(row)
            coverage_rows.append({"cell_id": cell["id"], "status": row["status"], "reason": row["reason"]})
    with (metrics_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cell_id", "status", "reason"])
        writer.writeheader()
        writer.writerows(coverage_rows)


def test_protocol_validation_rejects_velocity_mismatch(tmp_path):
    module = _load_module()
    first = _protocol_study(tmp_path / "first", velocity=0.5)
    second = _protocol_study(tmp_path / "second", velocity=1.0)

    with pytest.raises(ValueError, match="velocities_mps"):
        module.validate_compatible_protocols([first, second])


def test_completed_recording_retains_invalid_cells_as_explicit_incomplete_metrics(tmp_path, monkeypatch):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    study = _protocol_study(run_dir)
    _write_grid_result(module, run_dir, study, 0.0, valid_cells=3)
    monkeypatch.setattr(module.grid_analysis, "load_study_manifest", lambda _path: study)

    result = module.load_grid_result(run_dir, "go2")

    assert result["progress"]["status"] == "complete"
    assert result["overall"]["coverage"]["valid_cells"] == 3
    assert result["overall"]["metrics"]["absolute_work"]["family_balanced_mean_abs_imbalance_percent"] is None


def test_common_planar_tracking_intersection_is_plan_ordered_and_coverage_gated(tmp_path):
    module = _load_module()
    study = _protocol_study(tmp_path / "run")

    def cell_rows(excluded: set[str]):
        rows = {}
        for index, cell in enumerate(study["cells"]):
            rows[cell["id"]] = {
                "cell_id": cell["id"],
                "family": cell["family"],
                "status": "valid",
                "planar_tracking_success": cell["id"] not in excluded,
                **{
                    f"{metric}_{suffix}": float(index + metric_index + 1)
                    for metric_index, metric in enumerate(module.PRIMARY_METRICS)
                    for suffix in ("abs_imbalance_percent", "total_per_s", "total_per_directed_m")
                },
            }
        return rows

    grids = [
        {"study": study, "cell_metrics": cell_rows(set())},
        {"study": study, "cell_metrics": cell_rows({"cell_3"})},
    ]

    common = module.common_planar_tracking_cell_ids(grids)
    summary = module.summarize_common_planar_tracking_cells(grids[0], common)

    assert common == ["cell_0", "cell_1", "cell_2"]
    assert summary["family_counts"] == {"trot": 1, "bound": 1, "half_bound": 1, "gallop": 0}
    assert summary["all_families_represented"] is False
    assert summary["metrics"]["absolute_work"]["abs_imbalance_percent"] is None


def test_updater_combines_training_and_grid_results_without_overwriting_legacy_files(tmp_path, monkeypatch):
    module = _load_module()
    analysis_dir = tmp_path / "logs" / "robot" / "gait_famility_v3_analysis"
    analysis_dir.mkdir(parents=True)
    sentinel = analysis_dir / "REPORT.md"
    sentinel.write_text("legacy report stays byte-for-byte unchanged\n", encoding="utf-8")
    runs = [tmp_path / "logs" / "robot" / "run_a", tmp_path / "logs" / "robot" / "run_b"]
    labels = ["No TRS", "TRS"]
    for run_dir in runs:
        run_dir.mkdir(parents=True)
        (run_dir / "model_19999.pt").write_bytes(b"checkpoint")
    comparison_study = {
        "robot": "go2",
        "display_name": "Test Go2",
        "runs": [
            {"label": label, "folder": str(run_dir.relative_to(tmp_path))}
            for label, run_dir in zip(labels, runs, strict=True)
        ],
    }
    (analysis_dir / "study.json").write_text(json.dumps(comparison_study), encoding="utf-8")
    with (analysis_dir / "training_efficiency.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["run", "label", *module.TRAINING_FIELDS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (label, run_dir) in enumerate(zip(labels, runs, strict=True)):
            writer.writerow(
                {
                    "run": str(run_dir.relative_to(tmp_path)),
                    "label": label,
                    "iterations": 20000,
                    "reward_auc_first_10000": 20.0 + index,
                    "reward_auc_full": 30.0 + index,
                    "reward_last_1000_mean": 40.0 + index,
                    "reward_last_1000_std": 1.0,
                    "reward_35_iteration": 1000 + index,
                    "reward_35_transitions": 12_000_000 + index,
                    "wall_time_hours": 9.0 + index,
                }
            )
    studies = {}
    for index, run_dir in enumerate(runs):
        study = _protocol_study(run_dir)
        study_path = run_dir / "evaluations" / "leg_usage_grid" / "study.json"
        studies[str(study_path.resolve())] = study
        _write_grid_result(module, run_dir, study, float(index))
    monkeypatch.setattr(module.grid_analysis, "load_study_manifest", lambda path: studies[str(path.resolve())])

    manifest = module.update_comparison(analysis_dir, repo_root=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "legacy report stays byte-for-byte unchanged\n"
    assert len(manifest["rows"]) == 2
    assert manifest["rows"][0]["reward_last_1000_mean"] == 40.0
    assert manifest["rows"][1]["fixed_grid_absolute_work_mean_abs_imbalance_percent"] == 22.5
    assert manifest["rows"][0]["fixed_grid_torque_squared_common_planar_tracking_total_per_directed_m"] > 0.0
    assert manifest["rows"][0]["fixed_grid_normalized_torque_available"] is False
    assert "sentinel-scale" in manifest["rows"][0]["fixed_grid_normalized_torque_unavailable_reason"]
    assert manifest["rows"][0]["rank_reward_last_1000"] == 2
    assert manifest["rows"][0]["rank_absolute_work_common_planar_tracking_imbalance"] == 1
    assert manifest["rows"][0]["reward_balance_pareto"] is True
    assert manifest["rows"][1]["reward_balance_pareto"] is True
    assert manifest["inputs"]["fixed_grid_protocol_sha256"]
    for output_name in module.OUTPUT_NAMES:
        assert (analysis_dir / output_name).is_file()
    report = (analysis_dir / "FIXED_GRID_COMPARISON.md").read_text(encoding="utf-8")
    assert "deterministic fixed-grid" in report
    assert "No TRS" in report
    assert "TRS" in report
