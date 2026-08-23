# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combine matched training results with fixed-grid leg-usage evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_leg_usage as grid_analysis  # noqa: E402

COMPARISON_METHOD_VERSION = "gait_family_fixed_grid_comparison_v2"
PHYSX_EFFORT_LIMIT_SENTINEL_THRESHOLD_NM = 1.0e8
PRIMARY_METRICS = (
    "torque_squared",
    "absolute_work",
    "vertical_grf_impulse",
)
METRIC_SHORT_LABELS = {
    "torque_squared": "Raw torque squared",
    "absolute_work": "Absolute mechanical work",
    "vertical_grf_impulse": "Vertical GRF impulse",
}
TRAINING_FIELDS = (
    "iterations",
    "reward_auc_first_10000",
    "reward_auc_full",
    "reward_last_1000_mean",
    "reward_last_1000_std",
    "reward_35_iteration",
    "reward_35_transitions",
    "wall_time_hours",
)
OUTPUT_NAMES = (
    "fixed_grid_comparison.csv",
    "fixed_grid_family_comparison.csv",
    "fixed_grid_comparison.json",
    "fixed_grid_leg_usage_overall.svg",
    "fixed_grid_leg_usage_by_family.svg",
    "fixed_grid_reward_vs_leg_usage.svg",
    "FIXED_GRID_COMPARISON.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty comparison table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_from_repo(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolved_run_key(value: str | Path, repo_root: Path) -> str:
    return os.path.normcase(str(_resolve_from_repo(value, repo_root)))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required fixed-grid result does not exist: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite numeric result, received {value!r}.")
    return result


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if not parsed.is_integer():
        raise ValueError(f"Expected an integer result, received {value!r}.")
    return int(parsed)


def load_training_rows(path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    """Load training-efficiency results keyed by resolved run directory."""
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training-efficiency table does not exist: {path}") from exc
    with stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise ValueError(f"Training-efficiency table is empty: {path}")
    by_run: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if not source.get("run"):
            raise ValueError(f"Training-efficiency row has no run path: {path}")
        key = _resolved_run_key(source["run"], repo_root)
        if key in by_run:
            raise ValueError(f"Training-efficiency table contains duplicate run {source['run']!r}.")
        row: dict[str, Any] = dict(source)
        for field in TRAINING_FIELDS:
            if field not in source:
                raise ValueError(f"Training-efficiency table is missing required column {field!r}: {path}")
            row[field] = _float_or_none(source[field])
        by_run[key] = row
    return by_run


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required fixed-grid table does not exist: {path}") from exc
    if not rows:
        raise ValueError(f"Required fixed-grid table is empty: {path}")
    return rows


def _csv_bool(value: Any, field: str, path: Path) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true/false in {field!r} at {path}; received {value!r}.")


def _load_family_summaries(path: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for source in _read_csv_rows(path):
        if source.get("scope") != "family":
            continue
        family = str(source.get("family", ""))
        if family not in grid_analysis.FAMILY_ORDER or family in summaries:
            raise ValueError(f"Fixed-grid family table has an invalid or duplicate family {family!r}: {path}")
        result: dict[str, Any] = {
            "family": family,
            "expected_cells": _int_or_none(source.get("expected_cells")),
            "valid_cells": _int_or_none(source.get("valid_cells")),
            "complete": _csv_bool(source.get("complete"), "complete", path),
            "tracking_success_cells": _int_or_none(source.get("tracking_success_cells")),
            "tracking_qualified_complete": _csv_bool(
                source.get("tracking_qualified_complete"), "tracking_qualified_complete", path
            ),
        }
        for metric in PRIMARY_METRICS:
            for suffix in (
                "mean_abs_imbalance_percent",
                "mean_total_per_directed_m",
                "tracking_qualified_observed_mean_abs_imbalance_percent",
                "tracking_qualified_observed_mean_total_per_s",
                "tracking_qualified_observed_mean_total_per_directed_m",
            ):
                field = f"{metric}_{suffix}"
                if field not in source:
                    raise ValueError(f"Fixed-grid family table is missing column {field!r}: {path}")
                result[field] = _float_or_none(source[field])
        expected = result["expected_cells"]
        valid = result["valid_cells"]
        tracking = result["tracking_success_cells"]
        if expected is None or valid is None or tracking is None or not 0 <= tracking <= valid <= expected:
            raise ValueError(f"Fixed-grid family table has invalid coverage for {family!r}: {path}")
        if result["complete"] is not (valid == expected):
            raise ValueError(f"Fixed-grid family completeness is inconsistent for {family!r}: {path}")
        if result["tracking_qualified_complete"] is not (tracking == expected):
            raise ValueError(f"Fixed-grid family tracking completeness is inconsistent for {family!r}: {path}")
        summaries[family] = result
    if tuple(summaries) != grid_analysis.FAMILY_ORDER:
        raise ValueError(
            f"Fixed-grid family table must contain ordered summaries for {grid_analysis.FAMILY_ORDER}: {path}"
        )
    return summaries


def _load_coverage_summary(path: Path, overall_coverage: dict[str, Any], expected_cells: int) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    if len(rows) != expected_cells:
        raise ValueError(f"Fixed-grid coverage table has {len(rows)} rows, expected {expected_cells}: {path}")
    statuses = Counter(str(row.get("status", "")) for row in rows)
    if dict(sorted(statuses.items())) != overall_coverage.get("status_counts"):
        raise ValueError(f"Fixed-grid coverage table status counts disagree with overall metrics: {path}")
    invalid_rows = [row for row in rows if row.get("status") != "valid"]
    reasons = Counter(str(row.get("reason", "")) or "unspecified" for row in invalid_rows)
    termination_count = sum(
        str(row.get("reason", "")).startswith(("episode terminated", "episode ended at sample")) for row in invalid_rows
    )
    return {
        "rows": rows,
        "status_counts": dict(sorted(statuses.items())),
        "failure_reasons": dict(sorted(reasons.items())),
        "termination_count": termination_count,
        "other_invalid_count": len(invalid_rows) - termination_count,
    }


def _load_cell_metrics(path: Path, study: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected_cells = {str(cell["id"]): cell for cell in study["cells"]}
    source_rows = _read_csv_rows(path)
    if len(source_rows) != len(expected_cells):
        raise ValueError(f"Fixed-grid cell table has {len(source_rows)} rows, expected {len(expected_cells)}: {path}")
    parsed: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        cell_id = str(source.get("cell_id", ""))
        if cell_id not in expected_cells or cell_id in parsed:
            raise ValueError(f"Fixed-grid cell table has an unknown or duplicate cell {cell_id!r}: {path}")
        planned = expected_cells[cell_id]
        family = str(source.get("family", ""))
        if family != planned.get("family"):
            raise ValueError(f"Fixed-grid cell table has the wrong family for {cell_id!r}: {path}")
        status = str(source.get("status", ""))
        planar_tracking_success = False
        yaw_tracking_success = False
        tracking_success = False
        if status == "valid":
            planar_tracking_success = _csv_bool(source.get("planar_tracking_success"), "planar_tracking_success", path)
            yaw_tracking_success = _csv_bool(source.get("yaw_tracking_success"), "yaw_tracking_success", path)
            tracking_success = _csv_bool(source.get("tracking_success"), "tracking_success", path)
            if tracking_success is not (planar_tracking_success and yaw_tracking_success):
                raise ValueError(f"Fixed-grid cell has inconsistent tracking flags for {cell_id!r}: {path}")
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "family": family,
            "status": status,
            "reason": str(source.get("reason", "")),
            "planar_tracking_success": planar_tracking_success,
            "yaw_tracking_success": yaw_tracking_success,
            "tracking_success": tracking_success,
            "tracking_rmse_mps": _float_or_none(source.get("tracking_rmse_mps")),
            "yaw_tracking_rmse_radps": _float_or_none(source.get("yaw_tracking_rmse_radps")),
            "effort_limit_min_nm": _float_or_none(source.get("effort_limit_min_nm")),
            "effort_limit_max_nm": _float_or_none(source.get("effort_limit_max_nm")),
        }
        if status == "valid" and any(
            row[field] is None
            for field in (
                "tracking_rmse_mps",
                "yaw_tracking_rmse_radps",
                "effort_limit_min_nm",
                "effort_limit_max_nm",
            )
        ):
            raise ValueError(f"Valid fixed-grid cell has missing tracking or effort-limit audit data: {cell_id!r}")
        for metric in PRIMARY_METRICS:
            for suffix in ("abs_imbalance_percent", "total_per_s", "total_per_directed_m"):
                field = f"{metric}_{suffix}"
                if field not in source:
                    raise ValueError(f"Fixed-grid cell table is missing column {field!r}: {path}")
                row[field] = _float_or_none(source[field])
        parsed[cell_id] = row
    if set(parsed) != set(expected_cells):
        raise ValueError(f"Fixed-grid cell table does not exactly cover its plan: {path}")
    return parsed


def common_planar_tracking_cell_ids(grids: list[dict[str, Any]]) -> list[str]:
    """Return plan-ordered cells that are valid and planar-tracking-qualified in every run."""
    if not grids:
        raise ValueError("At least one fixed-grid result is required.")
    qualifying_sets = [
        {
            cell_id
            for cell_id, row in grid["cell_metrics"].items()
            if row["status"] == "valid" and row["planar_tracking_success"]
        }
        for grid in grids
    ]
    common = set.intersection(*qualifying_sets)
    return [str(cell["id"]) for cell in grids[0]["study"]["cells"] if str(cell["id"]) in common]


def summarize_common_planar_tracking_cells(grid: dict[str, Any], cell_ids: list[str]) -> dict[str, Any]:
    """Aggregate one run over the shared cross-run planar-tracking-qualified cell set."""
    selected = [grid["cell_metrics"][cell_id] for cell_id in cell_ids]
    family_rows: dict[str, dict[str, Any]] = {}
    for family in grid_analysis.FAMILY_ORDER:
        family_cells = [row for row in selected if row["family"] == family]
        family_result: dict[str, Any] = {"cell_count": len(family_cells)}
        for metric in PRIMARY_METRICS:
            for suffix in ("abs_imbalance_percent", "total_per_s", "total_per_directed_m"):
                raw_values = [row[f"{metric}_{suffix}"] for row in family_cells]
                values = [float(value) for value in raw_values if value is not None]
                family_result[f"{metric}_{suffix}"] = (
                    sum(values) / len(values) if values and len(values) == len(raw_values) else None
                )
        family_rows[family] = family_result
    all_families_represented = all(result["cell_count"] > 0 for result in family_rows.values())
    metrics: dict[str, dict[str, Any]] = {}
    for metric in PRIMARY_METRICS:
        result: dict[str, Any] = {}
        for suffix in ("abs_imbalance_percent", "total_per_s", "total_per_directed_m"):
            family_values = [family_rows[family][f"{metric}_{suffix}"] for family in grid_analysis.FAMILY_ORDER]
            result[suffix] = (
                sum(float(value) for value in family_values) / len(family_values)
                if all_families_represented and all(value is not None for value in family_values)
                else None
            )
        metrics[metric] = result
    return {
        "cell_count": len(selected),
        "family_counts": {family: result["cell_count"] for family, result in family_rows.items()},
        "all_families_represented": all_families_represented,
        "families": family_rows,
        "metrics": metrics,
    }


def _protocol_identity(study: dict[str, Any]) -> dict[str, Any]:
    gait_rows = [
        {
            "index": gait.get("index"),
            "name": gait.get("name"),
            "family": gait.get("family"),
            "phases": gait.get("phases"),
            "training_weight": gait.get("training_weight"),
            "weight": gait.get("weight"),
            "time_reversal_partner": gait.get("time_reversal_partner"),
        }
        for gait in study.get("gaits", [])
    ]
    cells = [
        {
            "id": cell.get("id"),
            "gait_index": cell.get("gait_index"),
            "gait_name": cell.get("gait_name"),
            "family": cell.get("family"),
            "phases": cell.get("phases"),
            "velocity_mps": cell.get("velocity_mps"),
            "seed": cell.get("seed"),
            "step_dt": cell.get("step_dt"),
            "settle_steps": cell.get("settle_steps"),
            "measure_steps": cell.get("measure_steps"),
            "total_steps": cell.get("total_steps"),
        }
        for cell in study.get("cells", [])
    ]
    return {
        "schema_version": study.get("schema_version"),
        "method_version": study.get("method_version"),
        "robot": study.get("robot"),
        "task": study.get("task"),
        "gait_library_version": study.get("gait_library_version"),
        "source_provenance_sha256": study.get("source_provenance", {}).get("sha256"),
        "step_dt": study.get("step_dt"),
        "settle_s": study.get("settle_s"),
        "measure_s": study.get("measure_s"),
        "settle_steps": study.get("settle_steps"),
        "measure_steps": study.get("measure_steps"),
        "total_steps": study.get("total_steps"),
        "evaluation_seed": study.get("evaluation_seed"),
        "nominal_profile": study.get("nominal_profile"),
        "runtime_overrides": study.get("runtime_overrides"),
        "velocities_mps": study.get("velocities_mps"),
        "gaits": gait_rows,
        "cells": cells,
        "aggregation": study.get("aggregation"),
    }


def validate_compatible_protocols(studies: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Validate that all fixed-grid studies used the same scientific protocol."""
    if not studies:
        raise ValueError("At least one fixed-grid study is required.")
    reference = _protocol_identity(studies[0])
    if reference["schema_version"] != grid_analysis.SCHEMA_VERSION:
        raise ValueError(f"Unsupported fixed-grid schema: {reference['schema_version']!r}.")
    if reference["method_version"] != grid_analysis.METHOD_VERSION:
        raise ValueError(f"Unsupported fixed-grid method: {reference['method_version']!r}.")
    if not reference["source_provenance_sha256"]:
        raise ValueError("Fixed-grid study does not record source provenance.")
    for index, study in enumerate(studies[1:], start=2):
        candidate = _protocol_identity(study)
        if candidate != reference:
            differing = sorted(key for key in reference if reference[key] != candidate.get(key))
            raise ValueError(
                f"Fixed-grid study {index} is incompatible with study 1; differing protocol fields: "
                + ", ".join(differing)
            )
    return reference, _canonical_sha256(reference)


def _verify_analysis_provenance(
    grid_root: Path, study_path: Path, study: dict[str, Any], overall: dict[str, Any]
) -> dict[str, Any]:
    provenance_path = grid_root / "metrics" / "analysis_provenance.json"
    provenance = _read_json_object(provenance_path)
    expected_study_sha = _sha256_file(study_path)
    if provenance.get("input_study_sha256") != expected_study_sha:
        raise ValueError(f"Fixed-grid analysis provenance is stale for {grid_root}.")
    if provenance.get("input_study_identity_sha256") != study.get("study_identity_sha256"):
        raise ValueError(f"Fixed-grid analysis provenance has the wrong study identity: {grid_root}")
    recorded_provenance_sha = provenance.get("record_sha256")
    provenance_without_record = {key: value for key, value in provenance.items() if key != "record_sha256"}
    if recorded_provenance_sha != _canonical_sha256(provenance_without_record):
        raise ValueError(f"Fixed-grid analysis provenance has an invalid record digest: {grid_root}")
    summary_provenance = overall.get("analysis_provenance", {})
    for key in ("record_sha256", "analyzer_sha256", "input_study_sha256"):
        if summary_provenance.get(key) != provenance.get(key):
            raise ValueError(f"Fixed-grid overall metrics and analysis provenance disagree on {key!r}: {grid_root}")
    return provenance


def load_grid_result(run_dir: Path, expected_robot: str) -> dict[str, Any]:
    """Load and validate one completed fixed-grid evaluation."""
    grid_root = run_dir / "evaluations" / "leg_usage_grid"
    study_path = grid_root / "study.json"
    study = grid_analysis.load_study_manifest(study_path)
    if study.get("robot") != expected_robot:
        raise ValueError(
            f"Fixed-grid robot {study.get('robot')!r} does not match comparison robot {expected_robot!r}: {run_dir}"
        )
    checkpoint_parent = Path(str(study.get("checkpoint", {}).get("path", ""))).expanduser().resolve().parent
    if checkpoint_parent != run_dir.resolve():
        raise ValueError(f"Fixed-grid checkpoint does not belong to its declared run directory: {run_dir}")
    if (grid_root / ".recording.lock").exists():
        raise ValueError(f"Fixed-grid recording is still active or locked: {grid_root}")
    progress_path = grid_root / "progress.json"
    progress = _read_json_object(progress_path)
    expected_cells = len(study.get("cells", []))
    if (
        progress.get("status") != "complete"
        or progress.get("total_cells") != expected_cells
        or progress.get("completed_cells") != expected_cells
    ):
        raise ValueError(
            f"Fixed-grid recording has not completed all planned cells "
            f"({progress.get('completed_cells')}/{expected_cells}, status={progress.get('status')!r}): {grid_root}"
        )
    overall_path = grid_root / "metrics" / "overall_metrics.json"
    overall = _read_json_object(overall_path)
    if overall.get("schema_version") != study.get("schema_version"):
        raise ValueError(f"Fixed-grid result schema does not match its study: {grid_root}")
    if overall.get("method_version") != study.get("method_version"):
        raise ValueError(f"Fixed-grid result method does not match its study: {grid_root}")
    coverage = overall.get("coverage", {})
    if coverage.get("expected_cells") != expected_cells:
        raise ValueError(f"Fixed-grid result has inconsistent expected-cell coverage: {grid_root}")
    valid_cells = coverage.get("valid_cells")
    if not isinstance(valid_cells, int) or not 0 <= valid_cells <= expected_cells:
        raise ValueError(f"Fixed-grid result has invalid valid-cell coverage: {grid_root}")
    if coverage.get("complete") is not (valid_cells == expected_cells):
        raise ValueError(f"Fixed-grid result has internally inconsistent completeness: {grid_root}")
    if coverage.get("failed_or_missing_cells") != expected_cells - valid_cells:
        raise ValueError(f"Fixed-grid result has internally inconsistent failed-cell coverage: {grid_root}")
    provenance = _verify_analysis_provenance(grid_root, study_path, study, overall)
    for metric in PRIMARY_METRICS:
        result = overall.get("metrics", {}).get(metric)
        if not isinstance(result, dict):
            raise ValueError(f"Fixed-grid metric {metric!r} is missing: {grid_root}")
        value = result.get("family_balanced_mean_abs_imbalance_percent")
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Fixed-grid metric {metric!r} has a non-finite family-balanced value: {grid_root}")
    family_path = grid_root / "metrics" / "family_metrics.csv"
    family_summaries = _load_family_summaries(family_path)
    if sum(summary["expected_cells"] for summary in family_summaries.values()) != expected_cells:
        raise ValueError(f"Fixed-grid family expected-cell coverage does not sum to the plan: {grid_root}")
    if sum(summary["valid_cells"] for summary in family_summaries.values()) != valid_cells:
        raise ValueError(f"Fixed-grid family valid-cell coverage does not match overall metrics: {grid_root}")
    coverage_path = grid_root / "metrics" / "coverage.csv"
    coverage_summary = _load_coverage_summary(coverage_path, coverage, expected_cells)
    cell_metrics_path = grid_root / "metrics" / "cell_metrics.csv"
    cell_metrics = _load_cell_metrics(cell_metrics_path, study)
    cell_status_counts = dict(sorted(Counter(row["status"] for row in cell_metrics.values()).items()))
    if cell_status_counts != coverage_summary["status_counts"]:
        raise ValueError(f"Fixed-grid cell metrics and coverage table have different status counts: {grid_root}")
    valid_tracking_rows = [row for row in cell_metrics.values() if row["status"] == "valid"]
    planar_rmse = [float(row["tracking_rmse_mps"]) for row in valid_tracking_rows]
    yaw_rmse = [float(row["yaw_tracking_rmse_radps"]) for row in valid_tracking_rows]
    effort_limit_minima = [float(row["effort_limit_min_nm"]) for row in valid_tracking_rows]
    effort_limit_maxima = [float(row["effort_limit_max_nm"]) for row in valid_tracking_rows]
    recorded_effort_limit_min = min(effort_limit_minima, default=None)
    recorded_effort_limit_max = max(effort_limit_maxima, default=None)
    normalized_torque_available = (
        recorded_effort_limit_max is not None and recorded_effort_limit_max < PHYSX_EFFORT_LIMIT_SENTINEL_THRESHOLD_NM
    )
    tracking_audit = {
        "planar_success_cells": sum(row["planar_tracking_success"] for row in valid_tracking_rows),
        "yaw_success_cells": sum(row["yaw_tracking_success"] for row in valid_tracking_rows),
        "combined_success_cells": sum(row["tracking_success"] for row in valid_tracking_rows),
        "planar_rmse_mean_mps": sum(planar_rmse) / len(planar_rmse) if planar_rmse else None,
        "planar_rmse_max_mps": max(planar_rmse, default=None),
        "yaw_rmse_mean_radps": sum(yaw_rmse) / len(yaw_rmse) if yaw_rmse else None,
        "yaw_rmse_max_radps": max(yaw_rmse, default=None),
        "recorded_effort_limit_min_nm": recorded_effort_limit_min,
        "recorded_effort_limit_max_nm": recorded_effort_limit_max,
        "normalized_torque_available": normalized_torque_available,
        "normalized_torque_unavailable_reason": (
            None
            if normalized_torque_available
            else "recorded joint_effort_limits are PhysX sentinel-scale, not physical actuator limits"
        ),
        "families": {
            family: {
                "planar_success_cells": sum(
                    row["planar_tracking_success"] for row in valid_tracking_rows if row["family"] == family
                ),
                "yaw_success_cells": sum(
                    row["yaw_tracking_success"] for row in valid_tracking_rows if row["family"] == family
                ),
                "combined_success_cells": sum(
                    row["tracking_success"] for row in valid_tracking_rows if row["family"] == family
                ),
            }
            for family in grid_analysis.FAMILY_ORDER
        },
    }
    if tracking_audit["combined_success_cells"] != coverage.get("tracking_success_cells"):
        raise ValueError(f"Fixed-grid cell metrics and overall metrics have different tracking counts: {grid_root}")
    return {
        "run_dir": run_dir,
        "grid_root": grid_root,
        "study_path": study_path,
        "study": study,
        "overall_path": overall_path,
        "overall": overall,
        "provenance": provenance,
        "progress_path": progress_path,
        "progress": progress,
        "family_path": family_path,
        "family_summaries": family_summaries,
        "coverage_path": coverage_path,
        "coverage_summary": coverage_summary,
        "cell_metrics_path": cell_metrics_path,
        "cell_metrics": cell_metrics,
        "tracking_audit": tracking_audit,
    }


def _load_analysis_study(analysis_dir: Path) -> dict[str, Any]:
    study = _read_json_object(analysis_dir / "study.json")
    if not isinstance(study.get("runs"), list) or not study["runs"]:
        raise ValueError(f"Comparison study has no runs: {analysis_dir / 'study.json'}")
    if study.get("robot") not in grid_analysis.SUPPORTED_ROBOTS:
        raise ValueError(f"Comparison study has unsupported robot {study.get('robot')!r}.")
    return study


def resolve_cohort(analysis_study: dict[str, Any], explicit_runs: list[str], repo_root: Path) -> list[tuple[str, Path]]:
    """Resolve the default study cohort or an explicitly ordered run cohort."""
    configured: dict[str, dict[str, Any]] = {}
    for entry in analysis_study["runs"]:
        configured[_resolved_run_key(entry["folder"], repo_root)] = entry
    if not explicit_runs:
        cohort = [
            (str(entry["label"]), _resolve_from_repo(entry["folder"], repo_root)) for entry in analysis_study["runs"]
        ]
    else:
        cohort = []
        for specification in explicit_runs:
            if "=" in specification:
                label, raw_path = specification.split("=", maxsplit=1)
                if not label.strip() or not raw_path.strip():
                    raise ValueError(f"Invalid --run specification {specification!r}; use LABEL=RUN_PATH.")
                run_path = _resolve_from_repo(raw_path.strip(), repo_root)
                cohort.append((label.strip(), run_path))
            else:
                run_path = _resolve_from_repo(specification, repo_root)
                configured_entry = configured.get(os.path.normcase(str(run_path)))
                label = str(configured_entry["label"]) if configured_entry else run_path.name
                cohort.append((label, run_path))
    if len({os.path.normcase(str(path)) for _, path in cohort}) != len(cohort):
        raise ValueError("The comparison cohort contains duplicate run directories.")
    if len({label for label, _ in cohort}) != len(cohort):
        raise ValueError("The comparison cohort contains duplicate labels.")
    return cohort


def _combined_row(
    label: str,
    run_dir: Path,
    training: dict[str, Any],
    grid: dict[str, Any],
    common_summary: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    study = grid["study"]
    overall = grid["overall"]
    coverage = overall["coverage"]
    family_summaries = grid["family_summaries"]
    tracking_audit = grid["tracking_audit"]
    tracking_families_represented = sum(summary["tracking_success_cells"] > 0 for summary in family_summaries.values())
    planar_tracking_families_represented = sum(
        summary["planar_success_cells"] > 0 for summary in tracking_audit["families"].values()
    )
    yaw_tracking_families_represented = sum(
        summary["yaw_success_cells"] > 0 for summary in tracking_audit["families"].values()
    )
    all_families_tracking_represented = tracking_families_represented == len(grid_analysis.FAMILY_ORDER)
    try:
        run_label = run_dir.relative_to(repo_root).as_posix()
    except ValueError:
        run_label = str(run_dir)
    row: dict[str, Any] = {
        "run": run_label,
        "label": label,
        "checkpoint_iteration": study["checkpoint"].get("iteration"),
        "checkpoint_sha256": study["checkpoint"].get("sha256"),
        "fixed_grid_method_version": study["method_version"],
        "fixed_grid_study_identity_sha256": study["study_identity_sha256"],
        "fixed_grid_valid_cells": coverage["valid_cells"],
        "fixed_grid_expected_cells": coverage["expected_cells"],
        "fixed_grid_tracking_success_cells": coverage["tracking_success_cells"],
        "fixed_grid_tracking_success_fraction": coverage["tracking_success_fraction_of_valid"],
        "fixed_grid_tracking_families_represented": tracking_families_represented,
        "fixed_grid_planar_tracking_success_cells": tracking_audit["planar_success_cells"],
        "fixed_grid_planar_tracking_families_represented": planar_tracking_families_represented,
        "fixed_grid_yaw_tracking_success_cells": tracking_audit["yaw_success_cells"],
        "fixed_grid_yaw_tracking_families_represented": yaw_tracking_families_represented,
        "fixed_grid_planar_tracking_rmse_mean_mps": tracking_audit["planar_rmse_mean_mps"],
        "fixed_grid_planar_tracking_rmse_max_mps": tracking_audit["planar_rmse_max_mps"],
        "fixed_grid_yaw_tracking_rmse_mean_radps": tracking_audit["yaw_rmse_mean_radps"],
        "fixed_grid_yaw_tracking_rmse_max_radps": tracking_audit["yaw_rmse_max_radps"],
        "fixed_grid_recorded_effort_limit_min_nm": tracking_audit["recorded_effort_limit_min_nm"],
        "fixed_grid_recorded_effort_limit_max_nm": tracking_audit["recorded_effort_limit_max_nm"],
        "fixed_grid_normalized_torque_available": tracking_audit["normalized_torque_available"],
        "fixed_grid_normalized_torque_unavailable_reason": tracking_audit["normalized_torque_unavailable_reason"],
        "fixed_grid_failed_or_missing_cells": coverage["failed_or_missing_cells"],
        "fixed_grid_termination_count": grid["coverage_summary"]["termination_count"],
        "fixed_grid_other_invalid_count": grid["coverage_summary"]["other_invalid_count"],
        "fixed_grid_status_counts_json": json.dumps(grid["coverage_summary"]["status_counts"], sort_keys=True),
        "fixed_grid_failure_reasons_json": json.dumps(grid["coverage_summary"]["failure_reasons"], sort_keys=True),
        "fixed_grid_common_planar_tracking_cells": common_summary["cell_count"],
        "fixed_grid_common_planar_tracking_all_families_represented": common_summary["all_families_represented"],
    }
    for field in TRAINING_FIELDS:
        row[field] = training[field]
    for metric in PRIMARY_METRICS:
        result = overall["metrics"][metric]
        prefix = f"fixed_grid_{metric}"
        row[f"{prefix}_mean_abs_imbalance_percent"] = result["family_balanced_mean_abs_imbalance_percent"]
        row[f"{prefix}_observed_mean_abs_imbalance_percent"] = result[
            "observed_family_balanced_mean_abs_imbalance_percent"
        ]
        row[f"{prefix}_tracking_observed_mean_abs_imbalance_percent"] = (
            result["tracking_qualified_observed_family_balanced_mean_abs_imbalance_percent"]
            if all_families_tracking_represented
            else None
        )
        row[f"{prefix}_tracking_observed_total_per_s"] = (
            result["tracking_qualified_observed_family_balanced_mean_total_per_s"]
            if all_families_tracking_represented
            else None
        )
        row[f"{prefix}_tracking_observed_total_per_directed_m"] = (
            result["tracking_qualified_observed_family_balanced_mean_total_per_directed_m"]
            if all_families_tracking_represented
            else None
        )
        row[f"{prefix}_total_per_s"] = result["family_balanced_mean_total_per_s"]
        row[f"{prefix}_total_per_directed_m"] = result["family_balanced_mean_total_per_directed_m"]
        common_metric = common_summary["metrics"][metric]
        row[f"{prefix}_common_planar_tracking_mean_abs_imbalance_percent"] = common_metric["abs_imbalance_percent"]
        row[f"{prefix}_common_planar_tracking_total_per_s"] = common_metric["total_per_s"]
        row[f"{prefix}_common_planar_tracking_total_per_directed_m"] = common_metric["total_per_directed_m"]
    return row


def _family_rows(
    label: str, run_dir: Path, grid: dict[str, Any], common_summary: dict[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    try:
        run_label = run_dir.relative_to(repo_root).as_posix()
    except ValueError:
        run_label = str(run_dir)
    rows: list[dict[str, Any]] = []
    for family in grid_analysis.FAMILY_ORDER:
        summary = grid["family_summaries"][family]
        tracking_audit = grid["tracking_audit"]["families"][family]
        row: dict[str, Any] = {
            "run": run_label,
            "label": label,
            "family": family,
            "fixed_grid_expected_cells": summary["expected_cells"],
            "fixed_grid_valid_cells": summary["valid_cells"],
            "fixed_grid_tracking_success_cells": summary["tracking_success_cells"],
            "fixed_grid_planar_tracking_success_cells": tracking_audit["planar_success_cells"],
            "fixed_grid_yaw_tracking_success_cells": tracking_audit["yaw_success_cells"],
            "fixed_grid_common_planar_tracking_cells": common_summary["families"][family]["cell_count"],
        }
        for metric in PRIMARY_METRICS:
            row[f"fixed_grid_{metric}_mean_abs_imbalance_percent"] = summary[f"{metric}_mean_abs_imbalance_percent"]
            row[f"fixed_grid_{metric}_mean_total_per_directed_m"] = summary[f"{metric}_mean_total_per_directed_m"]
            row[f"fixed_grid_{metric}_tracking_observed_mean_abs_imbalance_percent"] = summary[
                f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"
            ]
            row[f"fixed_grid_{metric}_tracking_observed_total_per_s"] = summary[
                f"{metric}_tracking_qualified_observed_mean_total_per_s"
            ]
            row[f"fixed_grid_{metric}_tracking_observed_total_per_directed_m"] = summary[
                f"{metric}_tracking_qualified_observed_mean_total_per_directed_m"
            ]
            common_metric = common_summary["families"][family]
            row[f"fixed_grid_{metric}_common_planar_tracking_mean_abs_imbalance_percent"] = common_metric[
                f"{metric}_abs_imbalance_percent"
            ]
            row[f"fixed_grid_{metric}_common_planar_tracking_total_per_s"] = common_metric[f"{metric}_total_per_s"]
            row[f"fixed_grid_{metric}_common_planar_tracking_total_per_directed_m"] = common_metric[
                f"{metric}_total_per_directed_m"
            ]
        rows.append(row)
    return rows


def _save_figure_atomic(figure: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    figure.savefig(temporary, format="svg")
    temporary.replace(path)


def _save_overall_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(14.5, 5.0), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        values = [row[f"fixed_grid_{metric}_common_planar_tracking_mean_abs_imbalance_percent"] for row in rows]
        heights = [float("nan") if value is None else value for value in values]
        positions = np.arange(len(rows))
        bars = axis.bar(positions, heights, color=[colors[index % len(colors)] for index in range(len(rows))])
        axis.bar_label(
            bars,
            labels=["N/A" if value is None else f"{value:.1f}" for value in values],
            fontsize="x-small",
            padding=2,
        )
        axis.set_xticks(positions, [row["label"] for row in rows], rotation=30, ha="right")
        axis.set_title(METRIC_SHORT_LABELS[metric])
        axis.set_ylabel("Mean absolute front/hind imbalance [%]")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Fixed-grid leg usage on the common planar-tracking-qualified cell set (lower is more even)")
    _save_figure_atomic(figure, path)
    plt.close(figure)


def _save_family_figure(path: Path, rows: list[dict[str, Any]], labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_label_family = {(row["label"], row["family"]): row for row in rows}
    metric_matrices = {
        metric: np.asarray(
            [
                [
                    by_label_family[(label, family)][
                        f"fixed_grid_{metric}_common_planar_tracking_mean_abs_imbalance_percent"
                    ]
                    for label in labels
                ]
                for family in grid_analysis.FAMILY_ORDER
            ],
            dtype=float,
        )
        for metric in PRIMARY_METRICS
    }
    finite_values = np.concatenate([values[np.isfinite(values)] for values in metric_matrices.values()])
    color_maximum = float(np.max(finite_values)) if finite_values.size else 1.0
    figure, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(15.0, 5.2), layout="constrained")
    image = None
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        values = metric_matrices[metric]
        image = axis.imshow(values, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=color_maximum)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.set_yticks(
            range(len(grid_analysis.FAMILY_ORDER)),
            [family.replace("_", " ").title() for family in grid_analysis.FAMILY_ORDER],
        )
        axis.set_title(METRIC_SHORT_LABELS[metric])
        for family_index in range(values.shape[0]):
            for label_index in range(values.shape[1]):
                value = values[family_index, label_index]
                text = "N/A" if not np.isfinite(value) else f"{value:.1f}"
                axis.text(label_index, family_index, text, ha="center", va="center", fontsize="x-small")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Mean absolute imbalance [%]", shrink=0.78)
    figure.suptitle(
        "Fixed-grid leg usage by family on the common planar-tracking-qualified cell set (lower is more even)"
    )
    _save_figure_atomic(figure, path)
    plt.close(figure)


def _save_reward_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(14.5, 4.8), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        for index, row in enumerate(rows):
            imbalance = row[f"fixed_grid_{metric}_common_planar_tracking_mean_abs_imbalance_percent"]
            reward = row["reward_last_1000_mean"]
            if imbalance is None or reward is None:
                continue
            axis.scatter(
                imbalance,
                reward,
                color=colors[index % len(colors)],
                s=42,
            )
            axis.annotate(
                row["label"],
                (imbalance, reward),
                fontsize="xx-small",
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_xlabel(f"{METRIC_SHORT_LABELS[metric]} imbalance [%]\n(lower is more even)")
        axis.set_ylabel("Training reward, last 1,000 iterations")
        axis.grid(alpha=0.25)
    figure.suptitle("Training reward versus fixed-grid leg usage (descriptive, one seed per run)")
    _save_figure_atomic(figure, path)
    plt.close(figure)


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_percent(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}%"


def _assign_dense_rank(
    rows: list[dict[str, Any]],
    value_field: str,
    rank_field: str,
    *,
    higher_better: bool,
    eligible_only: bool = False,
) -> None:
    values = sorted(
        {
            float(row[value_field])
            for row in rows
            if row.get(value_field) is not None and (not eligible_only or row["comparison_eligible"])
        },
        reverse=higher_better,
    )
    ranks = {value: index + 1 for index, value in enumerate(values)}
    for row in rows:
        value = row.get(value_field)
        row[rank_field] = (
            None if value is None or (eligible_only and not row["comparison_eligible"]) else ranks[float(value)]
        )


def add_domain_ranks(rows: list[dict[str, Any]]) -> None:
    """Add transparent per-domain ranks and a reward/balance Pareto flag."""
    for row in rows:
        row["comparison_eligible"] = (
            row["fixed_grid_valid_cells"] == row["fixed_grid_expected_cells"]
            and row["fixed_grid_termination_count"] == 0
        )
        row["fixed_grid_planar_tracking_success_fraction_of_expected"] = (
            row["fixed_grid_planar_tracking_success_cells"] / row["fixed_grid_expected_cells"]
            if row["fixed_grid_expected_cells"]
            else None
        )
        row["fixed_grid_yaw_tracking_success_fraction_of_expected"] = (
            row["fixed_grid_yaw_tracking_success_cells"] / row["fixed_grid_expected_cells"]
            if row["fixed_grid_expected_cells"]
            else None
        )
        row["fixed_grid_combined_tracking_success_fraction_of_expected"] = (
            row["fixed_grid_tracking_success_cells"] / row["fixed_grid_expected_cells"]
            if row["fixed_grid_expected_cells"]
            else None
        )
    _assign_dense_rank(
        rows,
        "fixed_grid_planar_tracking_success_fraction_of_expected",
        "rank_planar_tracking_coverage",
        higher_better=True,
    )
    _assign_dense_rank(rows, "reward_last_1000_mean", "rank_reward_last_1000", higher_better=True, eligible_only=True)
    _assign_dense_rank(
        rows,
        "reward_auc_first_10000",
        "rank_reward_auc_first_10000",
        higher_better=True,
        eligible_only=True,
    )
    _assign_dense_rank(rows, "reward_auc_full", "rank_reward_auc_full", higher_better=True, eligible_only=True)
    _assign_dense_rank(
        rows,
        "reward_35_transitions",
        "rank_reward_35_transitions",
        higher_better=False,
        eligible_only=True,
    )
    balance_fields: list[str] = []
    for metric in PRIMARY_METRICS:
        balance_field = f"fixed_grid_{metric}_common_planar_tracking_mean_abs_imbalance_percent"
        balance_fields.append(balance_field)
        _assign_dense_rank(
            rows,
            balance_field,
            f"rank_{metric}_common_planar_tracking_imbalance",
            higher_better=False,
            eligible_only=True,
        )
        _assign_dense_rank(
            rows,
            f"fixed_grid_{metric}_common_planar_tracking_total_per_directed_m",
            f"rank_{metric}_common_planar_tracking_total_per_directed_m",
            higher_better=False,
            eligible_only=True,
        )
    pareto_fields = ["reward_last_1000_mean", *balance_fields]
    for candidate in rows:
        if not candidate["comparison_eligible"] or any(candidate.get(field) is None for field in pareto_fields):
            candidate["reward_balance_pareto"] = None
            continue
        dominated = False
        for challenger in rows:
            if (
                challenger is candidate
                or not challenger["comparison_eligible"]
                or any(challenger.get(field) is None for field in pareto_fields)
            ):
                continue
            no_worse = challenger["reward_last_1000_mean"] >= candidate["reward_last_1000_mean"] and all(
                challenger[field] <= candidate[field] for field in balance_fields
            )
            strictly_better = challenger["reward_last_1000_mean"] > candidate["reward_last_1000_mean"] or any(
                challenger[field] < candidate[field] for field in balance_fields
            )
            if no_worse and strictly_better:
                dominated = True
                break
        candidate["reward_balance_pareto"] = not dominated


def _report_text(
    analysis_study: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_sha: str,
    common_family_counts: dict[str, int],
) -> str:
    robot_name = analysis_study.get("display_name", analysis_study["robot"])
    common_cells = rows[0]["fixed_grid_common_planar_tracking_cells"]
    expected_cells = rows[0]["fixed_grid_expected_cells"]
    family_count_text = ", ".join(
        f"{family.replace('_', ' ')}={common_family_counts[family]}" for family in grid_analysis.FAMILY_ORDER
    )
    normalized_available_count = sum(row["fixed_grid_normalized_torque_available"] for row in rows)
    if normalized_available_count == 0:
        normalization_note = (
            "Normalized torque-squared is intentionally unavailable because every recording contains "
            "PhysX sentinel-scale `joint_effort_limits` (at least 1e8 N m), not physical actuator limits. "
            "The raw per-run grid files retain those invalid normalized values only for audit."
        )
    else:
        normalization_note = (
            "Raw torque squared remains the primary cross-run metric. Normalized-torque availability and recorded "
            "effort-limit ranges are retained per run in the machine-readable audit fields."
        )
    lines = [
        f"# {robot_name} fixed-grid leg-usage and training-efficiency comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This comparison uses the deterministic fixed-grid leg-usage protocol, not the older transition-based "
        "recording windows in the original report. Existing diagnosis artifacts are preserved unchanged.",
        "",
        f"Protocol: `{protocol['method_version']}`; SHA-256 `{protocol_sha}`; "
        f"{len(protocol['gaits'])} gait rows x {len(protocol['velocities_mps'])} velocities; "
        f"{protocol['settle_s']:g} s settle + {protocol['measure_s']:g} s measured per cell.",
        "",
        f"Headline leg-usage results use the exact cross-run intersection of cells that were valid and passed the "
        f"declared planar velocity-tracking threshold for every policy: **{common_cells}/{expected_cells} cells** "
        f"({family_count_text}). This keeps the gait/velocity sample identical across runs. If any family has zero "
        "common cells, family-balanced headline metrics and their ranks are N/A. Yaw tracking and the official "
        "combined planar-plus-yaw flag remain separate audit results; the yaw threshold is not relaxed post hoc.",
        "",
        "## Training efficiency",
        "",
        "Higher reward and reward AUC are better; fewer transitions to reward 35 are better. Wall time is context "
        "only because concurrent jobs and host load can differ.",
        "",
        "| Run | AUC first 10k | AUC full | Reward last 1k | Reward 35 transitions | Wall time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reward_35 = "N/A" if row["reward_35_transitions"] is None else f"{int(row['reward_35_transitions']):,}"
        reward_last = (
            "N/A"
            if row["reward_last_1000_mean"] is None or row["reward_last_1000_std"] is None
            else f"{row['reward_last_1000_mean']:.3f} +/- {row['reward_last_1000_std']:.3f}"
        )
        lines.append(
            f"| {row['label']} | {_format_number(row['reward_auc_first_10000'])} | "
            f"{_format_number(row['reward_auc_full'])} | {reward_last} | {reward_35} | "
            f"{_format_number(row['wall_time_hours'], 2)} h |"
        )
    lines.extend(
        [
            "",
            "## Recording integrity and coverage",
            "",
            "All planned cells were attempted. Terminations are separated from other invalid outcomes; neither is "
            "silently converted to a zero metric.",
            "",
            "| Run | Valid / planned | Terminated | Other invalid | Eligible |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        valid = f"{row['fixed_grid_valid_cells']}/{row['fixed_grid_expected_cells']}"
        lines.append(
            f"| {row['label']} | {valid} | {row['fixed_grid_termination_count']} | "
            f"{row['fixed_grid_other_invalid_count']} | {'yes' if row['comparison_eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Tracking audit",
            "",
            "Planar qualification drives the common-cell headline because this is a fixed-vx study. Yaw RMSE and "
            "the unchanged strict combined flag are retained to expose heading drift.",
            "",
            "| Run | Planar pass / valid | Planar RMSE mean / max [m/s] | Yaw pass / valid | "
            "Yaw RMSE mean / max [rad/s] | Strict both / valid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        valid_cells = row["fixed_grid_valid_cells"]
        lines.append(
            f"| {row['label']} | {row['fixed_grid_planar_tracking_success_cells']}/{valid_cells} | "
            f"{_format_number(row['fixed_grid_planar_tracking_rmse_mean_mps'], 4)} / "
            f"{_format_number(row['fixed_grid_planar_tracking_rmse_max_mps'], 4)} | "
            f"{row['fixed_grid_yaw_tracking_success_cells']}/{valid_cells} | "
            f"{_format_number(row['fixed_grid_yaw_tracking_rmse_mean_radps'], 4)} / "
            f"{_format_number(row['fixed_grid_yaw_tracking_rmse_max_radps'], 4)} | "
            f"{row['fixed_grid_tracking_success_cells']}/{valid_cells} |"
        )
    lines.extend(
        [
            "",
            "## Torque normalization audit",
            "",
            "Raw torque squared is the comparison metric. Sentinel-scale effort limits make normalized torque "
            "numerically tiny without adding physical meaning.",
            "",
            "| Run | Recorded effort-limit range [N m] | Normalized torque usable |",
            "|---|---:|:---:|",
        ]
    )
    for row in rows:
        limit_range = (
            f"{_format_number(row['fixed_grid_recorded_effort_limit_min_nm'], 3)} to "
            f"{_format_number(row['fixed_grid_recorded_effort_limit_max_nm'], 3)}"
        )
        lines.append(
            f"| {row['label']} | {limit_range} | {'yes' if row['fixed_grid_normalized_torque_available'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Common-cell planar-tracking-qualified leg balance",
            "",
            "Lower mean absolute front/hind imbalance is more even. Each family contributes equally; opposite "
            "signed imbalances cannot cancel.",
            "",
            "| Run | Common cells | Raw torque^2 | Absolute work | Vertical GRF impulse |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        torque_balance = row["fixed_grid_torque_squared_common_planar_tracking_mean_abs_imbalance_percent"]
        work_balance = row["fixed_grid_absolute_work_common_planar_tracking_mean_abs_imbalance_percent"]
        grf_balance = row["fixed_grid_vertical_grf_impulse_common_planar_tracking_mean_abs_imbalance_percent"]
        lines.append(
            f"| {row['label']} | {row['fixed_grid_common_planar_tracking_cells']} | "
            f"{_format_percent(torque_balance)} | {_format_percent(work_balance)} | {_format_percent(grf_balance)} |"
        )
    lines.extend(
        [
            "",
            "## Common-cell planar-tracking-qualified usage per directed meter",
            "",
            "Lower values mean less measured usage per commanded-direction distance on the same common cell set.",
            "",
            "| Run | Raw torque^2 [N^2 m^2 s/m] | Absolute work [J/m] | Vertical GRF impulse [N s/m] |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        torque_per_m = row["fixed_grid_torque_squared_common_planar_tracking_total_per_directed_m"]
        work_per_m = row["fixed_grid_absolute_work_common_planar_tracking_total_per_directed_m"]
        grf_per_m = row["fixed_grid_vertical_grf_impulse_common_planar_tracking_total_per_directed_m"]
        lines.append(
            f"| {row['label']} | {_format_number(torque_per_m, 6)} | "
            f"{_format_number(work_per_m, 6)} | {_format_number(grf_per_m, 6)} |"
        )
    lines.extend(
        [
            "",
            "## Domain ranks and Pareto status",
            "",
            "Ranks are reported per domain rather than collapsed into an opaque grand score. Comparison-domain "
            "ranks require full valid coverage and zero terminations; planar tracking coverage is ranked "
            "separately for every run. The Pareto flag uses reward last 1,000 (higher) and the three common-cell "
            "imbalance metrics (lower).",
            "",
            "| Run | Eligible | Planar TQ rank | Reward rank | AUC10k rank | Torque balance rank | "
            "Work balance rank | GRF balance rank | Pareto |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        pareto = "N/A" if row["reward_balance_pareto"] is None else ("yes" if row["reward_balance_pareto"] else "no")
        lines.append(
            f"| {row['label']} | {'yes' if row['comparison_eligible'] else 'no'} | "
            f"{_format_number(row['rank_planar_tracking_coverage'], 0)} | "
            f"{_format_number(row['rank_reward_last_1000'], 0)} | "
            f"{_format_number(row['rank_reward_auc_first_10000'], 0)} | "
            f"{_format_number(row['rank_torque_squared_common_planar_tracking_imbalance'], 0)} | "
            f"{_format_number(row['rank_absolute_work_common_planar_tracking_imbalance'], 0)} | "
            f"{_format_number(row['rank_vertical_grf_impulse_common_planar_tracking_imbalance'], 0)} | {pareto} |"
        )
    lines.extend(
        [
            "",
            "![Fixed-grid overall leg usage](fixed_grid_leg_usage_overall.svg)",
            "",
            "![Fixed-grid leg usage by family](fixed_grid_leg_usage_by_family.svg)",
            "",
            "![Training reward versus fixed-grid leg usage](fixed_grid_reward_vs_leg_usage.svg)",
            "",
            "## Validation and interpretation",
            "",
            "All included grid recordings completed every planned cell, used byte-validated study manifests, had "
            "current analysis provenance, and matched on robot, task, gait rows, velocities, timing, seed, nominal "
            "profile, runtime overrides, source provenance, analyzer hash, terminal training horizon, and aggregation "
            "rules. Checkpoint identity is intentionally different between runs.",
            "",
            "Common-cell results average per-cell absolute front/hind imbalance first within each gait family and "
            "then equally across trot, bound, half bound, and gallop. The CSV retains strict all-planned-cell audit "
            "values, each run's own observed strict planar-plus-yaw subset, and the common planar-qualified results.",
            "A terminated, malformed, or nonpositive-progress cell remains explicit in grid validity. Strict "
            "family-balanced metrics are N/A unless every planned cell is valid; observed partial-cohort values "
            "are retained only in the machine-readable CSV and are not substituted into the comparison figure.",
            "Absolute work per directed meter is an energetic quantity. Raw torque squared per meter and vertical "
            "GRF impulse per meter are actuator/load proxies, not direct energy or lifetime estimates.",
            normalization_note,
            "",
            "Training wall time is observational and may reflect concurrent host load. These single-seed results "
            "do not establish statistical significance or hardware lifetime.",
            "",
            "Machine-readable results are in `fixed_grid_comparison.csv`, "
            "`fixed_grid_family_comparison.csv`, and `fixed_grid_comparison.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def update_comparison(
    analysis_dir: Path,
    *,
    repo_root: Path,
    explicit_runs: list[str] | None = None,
) -> dict[str, Any]:
    """Regenerate a fixed-grid comparison without changing legacy analysis artifacts."""
    analysis_dir = analysis_dir.resolve()
    repo_root = repo_root.resolve()
    analysis_study_path = analysis_dir / "study.json"
    training_path = analysis_dir / "training_efficiency.csv"
    analysis_study = _load_analysis_study(analysis_dir)
    cohort = resolve_cohort(analysis_study, explicit_runs or [], repo_root)
    training_by_run = load_training_rows(training_path, repo_root)
    grid_results: list[dict[str, Any]] = []
    cohort_records: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for label, run_dir in cohort:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        key = os.path.normcase(str(run_dir.resolve()))
        if key not in training_by_run:
            raise ValueError(f"No training-efficiency row was found for run {run_dir}.")
        grid = load_grid_result(run_dir, str(analysis_study["robot"]))
        training_iterations = training_by_run[key]["iterations"]
        checkpoint_iteration = grid["study"]["checkpoint"].get("iteration")
        if (
            training_iterations is None
            or not float(training_iterations).is_integer()
            or checkpoint_iteration != int(training_iterations) - 1
        ):
            raise ValueError(
                f"Fixed-grid checkpoint iteration {checkpoint_iteration!r} does not match the terminal "
                f"training horizon {training_iterations!r} for {run_dir}."
            )
        grid_results.append(grid)
        cohort_records.append((label, run_dir, training_by_run[key], grid))
    protocol, protocol_sha = validate_compatible_protocols([grid["study"] for grid in grid_results])
    analyzer_shas = {grid["provenance"].get("analyzer_sha256") for grid in grid_results}
    if len(analyzer_shas) != 1 or None in analyzer_shas:
        raise ValueError("Fixed-grid results were generated by different or unknown analyzer versions.")
    training_horizons = {int(training["iterations"]) for _, _, training, _ in cohort_records}
    if len(training_horizons) != 1:
        raise ValueError(f"Comparison runs have different training horizons: {sorted(training_horizons)}")
    common_cell_ids = common_planar_tracking_cell_ids(grid_results)
    common_summaries = [summarize_common_planar_tracking_cells(grid, common_cell_ids) for grid in grid_results]
    combined_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for (label, run_dir, training, grid), common_summary in zip(cohort_records, common_summaries, strict=True):
        combined_rows.append(_combined_row(label, run_dir, training, grid, common_summary, repo_root))
        family_rows.extend(_family_rows(label, run_dir, grid, common_summary, repo_root))
    add_domain_ranks(combined_rows)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(analysis_dir / "fixed_grid_comparison.csv", combined_rows)
    _write_csv_atomic(analysis_dir / "fixed_grid_family_comparison.csv", family_rows)
    _save_overall_figure(analysis_dir / "fixed_grid_leg_usage_overall.svg", combined_rows)
    _save_family_figure(
        analysis_dir / "fixed_grid_leg_usage_by_family.svg", family_rows, [row["label"] for row in combined_rows]
    )
    _save_reward_figure(analysis_dir / "fixed_grid_reward_vs_leg_usage.svg", combined_rows)
    report = _report_text(
        analysis_study,
        combined_rows,
        protocol,
        protocol_sha,
        common_summaries[0]["family_counts"],
    )
    _write_text_atomic(analysis_dir / "FIXED_GRID_COMPARISON.md", report)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method_version": COMPARISON_METHOD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot": analysis_study["robot"],
        "analysis_directory": str(analysis_dir),
        "comparison_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "analysis_study": {"path": str(analysis_study_path), "sha256": _sha256_file(analysis_study_path)},
            "training_efficiency": {"path": str(training_path), "sha256": _sha256_file(training_path)},
            "fixed_grid_protocol": protocol,
            "fixed_grid_protocol_sha256": protocol_sha,
            "fixed_grid_analyzer_sha256": next(iter(analyzer_shas)),
            "common_planar_tracking_cell_ids": common_cell_ids,
            "common_planar_tracking_cell_ids_sha256": _canonical_sha256(common_cell_ids),
            "common_planar_tracking_family_counts": common_summaries[0]["family_counts"],
            "runs": [
                {
                    "label": row["label"],
                    "run": row["run"],
                    "fixed_grid_study_path": str(grid["study_path"]),
                    "fixed_grid_study_sha256": _sha256_file(grid["study_path"]),
                    "fixed_grid_progress_path": str(grid["progress_path"]),
                    "fixed_grid_progress_sha256": _sha256_file(grid["progress_path"]),
                    "fixed_grid_overall_path": str(grid["overall_path"]),
                    "fixed_grid_overall_sha256": _sha256_file(grid["overall_path"]),
                    "fixed_grid_family_metrics_path": str(grid["family_path"]),
                    "fixed_grid_family_metrics_sha256": _sha256_file(grid["family_path"]),
                    "fixed_grid_cell_metrics_path": str(grid["cell_metrics_path"]),
                    "fixed_grid_cell_metrics_sha256": _sha256_file(grid["cell_metrics_path"]),
                    "fixed_grid_coverage_path": str(grid["coverage_path"]),
                    "fixed_grid_coverage_sha256": _sha256_file(grid["coverage_path"]),
                    "fixed_grid_analysis_record_sha256": grid["provenance"]["record_sha256"],
                }
                for row, grid in zip(combined_rows, grid_results, strict=True)
            ],
        },
        "outputs": list(OUTPUT_NAMES),
        "rows": combined_rows,
        "family_rows": family_rows,
    }
    manifest["record_sha256"] = _canonical_sha256(manifest)
    _write_json_atomic(analysis_dir / "fixed_grid_comparison.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine existing training-efficiency results with completed fixed-grid leg-usage studies."
    )
    parser.add_argument(
        "analysis_dir",
        type=Path,
        help="Existing gait_famility_v3_analysis directory containing study.json and training_efficiency.csv.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to resolve run paths stored in the analysis study.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional ordered cohort entry as RUN_PATH or LABEL=RUN_PATH; repeat for multiple runs.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run the fixed-grid comparison updater."""
    args = _build_parser().parse_args(arguments)
    manifest = update_comparison(args.analysis_dir, repo_root=args.repo_root, explicit_runs=args.run)
    print(
        f"Updated {manifest['robot']} fixed-grid comparison for {len(manifest['rows'])} runs in "
        f"{manifest['analysis_directory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
