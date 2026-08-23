# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run and summarize the symmetric locomotion leg-usage evaluation grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD_VERSION = "leg_usage_grid_v1"
DEFAULT_VELOCITIES_MPS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
DEFAULT_SETTLE_S = 5.0
DEFAULT_MEASURE_S = 10.0
DEFAULT_EVALUATION_SEED = 42
YAW_TRACKING_SUCCESS_THRESHOLD_RADPS = 0.05
FAMILY_ORDER = ("trot", "bound", "half_bound", "gallop")
PRIMARY_METRICS = ("normalized_torque_utilization", "absolute_work", "vertical_grf_impulse")
ALL_METRICS = ("torque_squared", *PRIMARY_METRICS)
SUPPORTED_ROBOTS = ("go2", "x1")
METRIC_LABELS = {
    "torque_squared": "Raw torque squared",
    "normalized_torque_utilization": "Normalized torque squared",
    "absolute_work": "Absolute mechanical work",
    "vertical_grf_impulse": "Vertical GRF impulse",
}
METRIC_UNITS = {
    "torque_squared": "N^2 m^2 s",
    "normalized_torque_utilization": "s",
    "absolute_work": "J",
    "vertical_grf_impulse": "N s",
}


def canonical_training_gaits() -> tuple[str, tuple[dict[str, Any], ...]]:
    """Return the gait protocol derived from the task's canonical constants."""
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

    rows = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROWS
    weights = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_WEIGHTS
    names = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_ROW_NAMES
    families = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_FAMILIES
    partners = symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_TIME_REVERSAL_PARTNERS
    lengths = {len(rows), len(weights), len(names), len(families), len(partners)}
    if len(lengths) != 1 or not rows:
        raise ValueError("Canonical symmetric training gait metadata has inconsistent lengths.")
    gaits = tuple(
        {
            "index": index,
            "name": str(name),
            "family": str(family),
            "phases": [float(value) for value in phases],
            "training_weight": float(weight),
            "weight": float(weight),
            "time_reversal_partner": int(partner),
        }
        for index, (phases, weight, name, family, partner) in enumerate(
            zip(rows, weights, names, families, partners, strict=True)
        )
    )
    return str(symm_quadruped.SYMM_QUADRUPED_GAIT_LIBRARY_TRAIN_VERSION), gaits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _combined_file_provenance(repo_root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    missing: list[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            label = str(resolved)
        if resolved.is_file():
            files[label] = _sha256_file(resolved)
        else:
            missing.append(label)
    return {
        "sha256": _canonical_sha256(files),
        "files": files,
        "missing_files": sorted(missing),
    }


def _source_provenance(repo_root: Path, robot: str) -> dict[str, Any]:
    config_folder = "go2_symm" if robot == "go2" else "dobot_x1_symm"
    velocity_root = (
        repo_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "locomotion" / "velocity"
    )
    source_paths = (
        repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play_rsl_rl.py",
        repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "symm_rollout_plotter.py",
        repo_root / "scripts" / "symm_locomotion" / "analyze_leg_usage.py",
        repo_root / "scripts" / "symm_locomotion" / "symm_cli.py",
        velocity_root / "mdp" / "symm_quadruped.py",
        velocity_root / "config" / config_folder / "flat_env_cfg.py",
    )
    result = _combined_file_provenance(repo_root, source_paths)
    result.update(
        {
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_branch": _git_value(repo_root, "branch", "--show-current"),
        }
    )
    return result


def _config_provenance(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    return _combined_file_provenance(
        repo_root,
        (run_dir / "params" / "agent.yaml", run_dir / "params" / "env.yaml"),
    )


def _steps_for_duration(duration_s: float, step_dt: float, label: str) -> int:
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive; received {step_dt!r} s.")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"{label} must be finite and positive; received {duration_s!r} s.")
    steps = round(duration_s / step_dt)
    if steps < 1 or not math.isclose(duration_s, steps * step_dt, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"{label} must be an exact multiple of step_dt={step_dt:g} s; received {duration_s!r} s.")
    return steps


def velocity_slug(velocity_mps: float) -> str:
    """Return a stable directory label for a signed x velocity [m/s]."""
    sign = "pos" if velocity_mps > 0.0 else "neg"
    magnitude = f"{abs(velocity_mps):.6f}".rstrip("0").rstrip(".")
    if "." not in magnitude:
        magnitude += ".0"
    return f"{sign}_{magnitude.replace('.', 'p')}"


def _study_identity(study: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "method_version",
        "robot",
        "task",
        "gait_library_version",
        "checkpoint",
        "source_provenance",
        "config_provenance",
        "step_dt",
        "settle_s",
        "measure_s",
        "settle_steps",
        "measure_steps",
        "total_steps",
        "episode_length_s",
        "evaluation_seed",
        "nominal_profile",
        "render_cell_plots",
        "runtime_overrides",
        "velocities_mps",
        "gaits",
        "cells",
        "aggregation",
    )
    return {key: study[key] for key in keys}


def build_study(
    *,
    repo_root: Path,
    checkpoint: Path,
    robot: str,
    task: str,
    step_dt: float,
    settle_s: float = DEFAULT_SETTLE_S,
    measure_s: float = DEFAULT_MEASURE_S,
    evaluation_seed: int = DEFAULT_EVALUATION_SEED,
    velocities_mps: Sequence[float] = DEFAULT_VELOCITIES_MPS,
    gait_indices: Sequence[int] = tuple(range(10)),
    render_cell_plots: bool = False,
    runtime_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the immutable protocol and runtime plan for one checkpoint."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if robot not in SUPPORTED_ROBOTS:
        raise ValueError(f"Unsupported leg-usage robot: {robot}")
    if evaluation_seed < 0:
        raise ValueError("Evaluation seed must be non-negative.")
    settle_steps = _steps_for_duration(settle_s, step_dt, "Settle duration")
    measure_steps = _steps_for_duration(measure_s, step_dt, "Measurement duration")
    selected_velocities = [float(value) for value in velocities_mps]
    if not selected_velocities:
        raise ValueError("At least one x velocity is required.")
    if any(not math.isfinite(value) or value == 0.0 for value in selected_velocities):
        raise ValueError("Every x velocity must be finite and non-zero.")
    if len(set(selected_velocities)) != len(selected_velocities):
        raise ValueError("Duplicate x velocities are not allowed.")
    if len({velocity_slug(value) for value in selected_velocities}) != len(selected_velocities):
        raise ValueError("Selected x velocities must map to distinct output-directory labels.")
    gait_library_version, canonical_gaits = canonical_training_gaits()
    gait_by_index = {int(gait["index"]): gait for gait in canonical_gaits}
    if not gait_indices or len(set(gait_indices)) != len(gait_indices):
        raise ValueError("Gait indices must be a non-empty list without duplicates.")
    try:
        selected_gaits = [dict(gait_by_index[int(index)]) for index in gait_indices]
    except KeyError as exc:
        raise ValueError(f"Unknown gait index {exc.args[0]}; choose indices 0 through 9.") from exc

    run_dir = checkpoint.parent.resolve()
    output_root = run_dir / "evaluations" / "leg_usage_grid"
    total_steps = settle_steps + measure_steps
    cells: list[dict[str, Any]] = []
    for gait in selected_gaits:
        gait_index = int(gait["index"])
        family = str(gait["family"])
        for velocity_mps in selected_velocities:
            slug = velocity_slug(velocity_mps)
            relative_output_dir = (
                Path("cells") / f"gait_{gait_index:02d}_{family}" / f"vx_{slug}" / f"seed_{evaluation_seed:04d}"
            )
            cells.append(
                {
                    "id": f"gait_{gait_index:02d}_{family}__vx_{slug}__seed_{evaluation_seed:04d}",
                    "gait_index": gait_index,
                    "gait_name": gait["name"],
                    "family": family,
                    "phases": gait["phases"],
                    "velocity_mps": velocity_mps,
                    "seed": evaluation_seed,
                    "step_dt": step_dt,
                    "settle_steps": settle_steps,
                    "measure_steps": measure_steps,
                    "total_steps": total_steps,
                    "relative_output_dir": relative_output_dir.as_posix(),
                }
            )
    checkpoint_record = {
        "path": str(checkpoint),
        "iteration": _model_iteration(checkpoint),
        "sha256": _sha256_file(checkpoint),
    }
    study: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "robot": robot,
        "task": task,
        "gait_library_version": gait_library_version,
        "checkpoint": checkpoint_record,
        "source_provenance": _source_provenance(repo_root, robot),
        "config_provenance": _config_provenance(repo_root, run_dir),
        "step_dt": step_dt,
        "settle_s": settle_s,
        "measure_s": measure_s,
        "settle_steps": settle_steps,
        "measure_steps": measure_steps,
        "total_steps": total_steps,
        "episode_length_s": (total_steps + 2) * step_dt,
        "evaluation_seed": evaluation_seed,
        "nominal_profile": True,
        "render_cell_plots": render_cell_plots,
        "runtime_overrides": [str(value) for value in runtime_overrides],
        "velocities_mps": selected_velocities,
        "gaits": selected_gaits,
        "cells": cells,
        "aggregation": {
            "within_family": "equal mean across selected gait rows and velocities",
            "overall": "equal mean of trot, bound, half_bound, and gallop family means",
            "primary_balance_metric": "mean absolute per-cell front/hind imbalance; signed values never cancel",
            "tracking_qualified": "reported alongside all valid cells using the declared per-cell tracking threshold",
        },
    }
    study["study_identity_sha256"] = _canonical_sha256(_study_identity(study))
    return study


def _model_iteration(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("model_"):
        return -1
    try:
        return int(stem.removeprefix("model_"))
    except ValueError:
        return -1


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_study_manifest(study_path: Path) -> dict[str, Any]:
    """Load a study and verify its recorded immutable identity."""
    study_path = study_path.resolve()
    try:
        study = json.loads(study_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Leg-usage study manifest does not exist: {study_path}") from exc
    if not isinstance(study, dict):
        raise ValueError(f"Leg-usage study manifest must contain one JSON object: {study_path}")
    try:
        expected_digest = _canonical_sha256(_study_identity(study))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Study manifest is incomplete or incompatible: {study_path}") from exc
    if study.get("study_identity_sha256") != expected_digest:
        raise ValueError(f"Study identity digest does not match its protocol: {study_path}")
    declared_root = Path(str(study.get("output_root", ""))).expanduser().resolve()
    if declared_root != study_path.parent:
        raise ValueError(f"Study output_root does not match its manifest directory: {study_path}")
    return study


def validate_existing_study_for_analysis(
    study_path: Path,
    *,
    checkpoint: Path,
    robot: str,
    task: str,
    supplied_runtime_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate recorded inputs for analysis without comparing current source."""
    study_path = study_path.resolve()
    recording_lock = study_path.parent / ".recording.lock"
    if recording_lock.exists():
        raise ValueError(
            f"Leg-usage recording is already locked by another launcher: {recording_lock}. "
            "Wait for it to finish before analyzing."
        )
    study = load_study_manifest(study_path)
    checkpoint = checkpoint.resolve()
    recorded_checkpoint = study.get("checkpoint", {})
    if Path(str(recorded_checkpoint.get("path", ""))).expanduser().resolve() != checkpoint:
        raise ValueError("Existing leg-usage study belongs to a different checkpoint path.")
    if recorded_checkpoint.get("sha256") != _sha256_file(checkpoint):
        raise ValueError("Existing leg-usage study checkpoint SHA-256 does not match the selected checkpoint.")
    if study.get("robot") != robot or study.get("task") != task:
        raise ValueError("Existing leg-usage study robot/task does not match the selected utility robot.")
    supplied_overrides = [str(value) for value in supplied_runtime_overrides]
    if supplied_overrides and supplied_overrides != study.get("runtime_overrides", []):
        raise ValueError("Supplied analyze-only runtime overrides do not match the recorded study protocol.")
    return study


def prepare_study(
    study: dict[str, Any],
    *,
    resume: bool,
    analyze_only: bool,
    dry_run: bool,
) -> Path:
    """Validate or create ``study.json`` without ever mixing protocols."""
    output_root = Path(study["output_root"])
    study_path = output_root / "study.json"
    recording_lock = output_root / ".recording.lock"
    if recording_lock.exists() and not dry_run:
        raise ValueError(
            f"Leg-usage recording is already locked by another launcher: {recording_lock}. "
            "Wait for it to finish; stale locks require deliberate removal after verifying no evaluator is running."
        )
    if study_path.exists():
        existing = json.loads(study_path.read_text(encoding="utf-8"))
        recorded_digest = existing.get("study_identity_sha256")
        try:
            actual_digest = _canonical_sha256(_study_identity(existing))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Existing study manifest is incomplete or incompatible: {study_path}") from exc
        if recorded_digest != actual_digest:
            raise ValueError(f"Existing study manifest has an invalid identity digest: {study_path}")
        if actual_digest != study["study_identity_sha256"]:
            raise ValueError(
                "The existing leg-usage output belongs to a different checkpoint or protocol. "
                f"Refusing to mix results in {output_root}."
            )
        if not (resume or analyze_only):
            has_cell_artifacts = (output_root / "cells").exists() and any((output_root / "cells").rglob("*"))
            has_runtime_artifacts = has_cell_artifacts or any(
                (output_root / name).exists() for name in ("events.jsonl", "progress.json", "status.json")
            )
            if has_runtime_artifacts:
                raise ValueError(f"Leg-usage results already exist in {output_root}; pass --resume to continue them.")
        study = existing
    elif analyze_only:
        raise FileNotFoundError(f"Analyze-only mode requires an existing study manifest: {study_path}")
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Refusing to initialize a study in non-empty directory without study.json: {output_root}")

    if not dry_run and not study_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(study_path, study)
    return study_path


def pair_imbalance(front: float, hind: float) -> float | None:
    """Return signed front/hind imbalance in percent, or ``None`` for no usage."""
    denominator = front + hind
    return None if denominator <= 0.0 else 100.0 * (front - hind) / denominator


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _runtime_failure(cell_dir: Path) -> str | None:
    status = _read_json_if_present(cell_dir / "status.json")
    metadata = _read_json_if_present(cell_dir / "metadata.json")
    state = str(status.get("state", status.get("status", ""))).lower()
    if state in {"failed", "failure", "terminated", "invalid", "error", "interrupted", "running"}:
        return str(status.get("reason", status.get("message", status.get("error", state))))
    if status.get("terminated_early") is True or status.get("failed") is True:
        return str(status.get("reason", "runtime reported early termination"))
    for payload in (status, metadata):
        outcome = str(payload.get("outcome", "")).lower()
        if outcome == "terminated":
            terms = payload.get("termination_terms") or []
            term_text = ", ".join(str(term) for term in terms) if terms else "unknown term"
            termination_time_s = payload.get("termination_time_s")
            if termination_time_s is None:
                recorded_steps = payload.get("recorded_steps")
                step_dt = payload.get("step_dt", metadata.get("step_dt"))
                if recorded_steps is not None and step_dt is not None:
                    termination_time_s = float(recorded_steps) * float(step_dt)
            time_text = "unknown time" if termination_time_s is None else f"t={float(termination_time_s):g} s"
            return f"episode terminated ({term_text}) at {time_text}"
        if outcome in {"error", "interrupted", "failed", "failure"}:
            return str(payload.get("reason", payload.get("error", f"runtime outcome: {outcome}")))
    return None


def _period_s(data: Any, cell_dir: Path) -> float | None:
    for key in ("gait_period_s", "gait_period", "gait_periods", "command_gait_periods"):
        if key in data:
            values = np.asarray(data[key], dtype=float).reshape(-1)
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size:
                return float(np.median(values))
    metadata = _read_json_if_present(cell_dir / "metadata.json")
    for key in ("gait_period_s", "gait_period"):
        value = metadata.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return None


def _metric_summary(values: np.ndarray, step_dt: float) -> dict[str, float | None]:
    front = float(values[:, :2].sum() * step_dt)
    hind = float(values[:, 2:].sum() * step_dt)
    signed = pair_imbalance(front, hind)
    return {
        "front_integral": front,
        "hind_integral": hind,
        "total_integral": front + hind,
        "signed_imbalance_percent": signed,
        "abs_imbalance_percent": None if signed is None else abs(signed),
    }


def _invalid_cell_row(cell: dict[str, Any], reason: str, status: str = "invalid") -> dict[str, Any]:
    return {
        "cell_id": cell["id"],
        "gait_index": cell["gait_index"],
        "gait_name": cell["gait_name"],
        "family": cell["family"],
        "velocity_mps": cell["velocity_mps"],
        "velocity_sign": "forward" if cell["velocity_mps"] > 0.0 else "backward",
        "seed": cell["seed"],
        "status": status,
        "reason": reason,
        "relative_output_dir": cell["relative_output_dir"],
    }


def analyze_cell(study: dict[str, Any], cell: dict[str, Any], study_root: Path) -> dict[str, Any]:  # noqa: C901
    """Analyze one cell's steady-state measurement window."""
    cell_dir = study_root / cell["relative_output_dir"]
    runtime_failure = _runtime_failure(cell_dir)
    if runtime_failure is not None:
        return _invalid_cell_row(cell, runtime_failure, "failed")
    plan_path = study_root / "study.json"
    if plan_path.is_file():
        expected_plan_sha256 = _sha256_file(plan_path)
        for metadata_path in (cell_dir / "status.json", cell_dir / "metadata.json"):
            payload = _read_json_if_present(metadata_path)
            recorded_plan_sha256 = payload.get("plan_sha256")
            if recorded_plan_sha256 is not None and recorded_plan_sha256 != expected_plan_sha256:
                return _invalid_cell_row(cell, f"{metadata_path.name} belongs to a different study plan")
    data_path = cell_dir / "sim_data.npz"
    if not data_path.is_file():
        return _invalid_cell_row(cell, "sim_data.npz is missing", "missing")
    try:
        with np.load(data_path, allow_pickle=False) as data:
            required = (
                "time_steps",
                "desired_lin_vel",
                "true_lin_vel",
                "base_positions",
                "joint_torques",
                "joint_powers",
                "joint_effort_limits",
                "foot_ground_reaction_forces_w",
                "episode_done",
            )
            missing = [key for key in required if key not in data]
            if missing:
                return _invalid_cell_row(cell, f"missing arrays: {', '.join(missing)}")
            times = np.asarray(data["time_steps"], dtype=float)
            commands = np.asarray(data["desired_lin_vel"], dtype=float)
            velocities = np.asarray(data["true_lin_vel"], dtype=float)
            positions = np.asarray(data["base_positions"], dtype=float)
            torques = np.asarray(data["joint_torques"], dtype=float)
            powers = np.asarray(data["joint_powers"], dtype=float)
            joint_effort_limits = np.asarray(data["joint_effort_limits"], dtype=float)
            ground_reaction_forces = np.asarray(data["foot_ground_reaction_forces_w"], dtype=float)
            episode_done = np.asarray(data["episode_done"], dtype=bool)
            period_s = _period_s(data, cell_dir)
            foot_thetas = np.asarray(data["foot_thetas"], dtype=float) if "foot_thetas" in data else None
            configured_foot_thetas = (
                np.asarray(data["configured_foot_thetas"], dtype=float) if "configured_foot_thetas" in data else None
            )
            common_gait_phases = (
                np.asarray(data["common_gait_phases"], dtype=float) if "common_gait_phases" in data else None
            )
            gait_period_trace = np.asarray(data["gait_periods"], dtype=float) if "gait_periods" in data else None
            archive_plan_sha256 = str(np.asarray(data["plan_sha256"]).item()) if "plan_sha256" in data else None
            archive_cell_id = str(np.asarray(data["cell_id"]).item()) if "cell_id" in data else None
    except (OSError, ValueError) as exc:
        return _invalid_cell_row(cell, f"cannot load sim_data.npz: {exc}")

    if plan_path.is_file() and archive_plan_sha256 is not None and archive_plan_sha256 != expected_plan_sha256:
        return _invalid_cell_row(cell, "sim_data.npz belongs to a different study plan")
    if archive_cell_id is not None and archive_cell_id != str(cell["id"]):
        return _invalid_cell_row(cell, f"sim_data.npz cell_id {archive_cell_id!r} does not match plan")

    expected_samples = int(cell["total_steps"])
    arrays = (times, commands, velocities, positions, torques, powers, ground_reaction_forces, episode_done)
    if any(array.shape[0] != expected_samples for array in arrays):
        counts = sorted({int(array.shape[0]) for array in arrays})
        status = "short" if min(counts) < expected_samples else "invalid"
        return _invalid_cell_row(
            cell,
            f"rollout sample count {counts} does not exactly match expected {expected_samples}",
            status,
        )
    if commands.shape[1:] != (3,) or velocities.shape[1:] != (3,) or positions.ndim != 2 or positions.shape[1] < 2:
        return _invalid_cell_row(cell, "unexpected command, velocity, or position shape")
    if torques.shape[1:] != (12,) or powers.shape[1:] != (12,):
        return _invalid_cell_row(cell, "expected 12 leg-major joints")
    if joint_effort_limits.shape == (expected_samples, 12):
        if not np.allclose(joint_effort_limits, joint_effort_limits[0], rtol=0.0, atol=1.0e-6):
            return _invalid_cell_row(cell, "joint_effort_limits changes within the fixed rollout")
        joint_effort_limits = joint_effort_limits[0]
    if (
        joint_effort_limits.shape != (12,)
        or not np.all(np.isfinite(joint_effort_limits))
        or not np.all(joint_effort_limits > 0.0)
    ):
        return _invalid_cell_row(cell, "expected 12 finite positive recorded joint effort limits")
    if ground_reaction_forces.shape[1:] != (4, 3):
        return _invalid_cell_row(cell, "expected ground reaction forces with shape (T, 4, 3)")
    if any(not np.all(np.isfinite(array[:expected_samples])) for array in arrays[:-1]):
        return _invalid_cell_row(cell, "non-finite values in required arrays")
    if bool(np.any(episode_done[:expected_samples])):
        first_done = int(np.flatnonzero(episode_done[:expected_samples])[0])
        return _invalid_cell_row(cell, f"episode ended at sample {first_done}", "failed")

    expected_velocity = float(cell["velocity_mps"])
    command_tolerance = 1.0e-5
    if not np.allclose(commands[:, 0], expected_velocity, rtol=0.0, atol=command_tolerance):
        observed_range = (float(np.min(commands[:, 0])), float(np.max(commands[:, 0])))
        return _invalid_cell_row(
            cell,
            f"x command is not fixed at {expected_velocity:g} m/s; observed range {observed_range}",
        )
    if not np.allclose(commands[:, 1:], 0.0, rtol=0.0, atol=command_tolerance):
        return _invalid_cell_row(cell, "lateral or yaw command is nonzero")
    expected_phases = np.asarray(cell["phases"], dtype=float)
    for label, recorded_phases in (
        ("foot_thetas", foot_thetas),
        ("configured_foot_thetas", configured_foot_thetas),
    ):
        if recorded_phases is None:
            continue
        if recorded_phases.shape == (4,):
            recorded_phases = recorded_phases.reshape(1, 4)
        if recorded_phases.ndim != 2 or recorded_phases.shape[1] != 4:
            return _invalid_cell_row(cell, f"unexpected {label} shape {recorded_phases.shape}")
        if label == "foot_thetas" and recorded_phases.shape[0] != expected_samples:
            return _invalid_cell_row(
                cell,
                f"foot_thetas sample count {recorded_phases.shape[0]} does not match expected {expected_samples}",
            )
        circular_error = np.remainder(recorded_phases - expected_phases + 0.5, 1.0) - 0.5
        if not np.all(np.isfinite(circular_error)) or float(np.max(np.abs(circular_error))) > 1.0e-5:
            return _invalid_cell_row(cell, f"{label} does not match planned gait row {cell['gait_index']}")
    if gait_period_trace is not None:
        gait_period_trace = gait_period_trace.reshape(-1)
        if (
            gait_period_trace.shape != (expected_samples,)
            or not np.all(np.isfinite(gait_period_trace))
            or not np.all(gait_period_trace > 0.0)
            or not np.allclose(gait_period_trace, gait_period_trace[0], rtol=0.0, atol=1.0e-6)
        ):
            return _invalid_cell_row(cell, "gait_periods is not a fixed positive per-sample trace")

    step_dt = float(study["step_dt"])
    time_deltas = np.diff(times[:expected_samples])
    if not np.all(time_deltas > 0.0) or not np.allclose(time_deltas, step_dt, rtol=0.0, atol=1.0e-6):
        observed_range = (float(np.min(time_deltas)), float(np.max(time_deltas)))
        return _invalid_cell_row(
            cell,
            f"time_steps are not strictly uniform at plan step_dt={step_dt:g}; observed delta range {observed_range}",
        )
    start = int(cell["settle_steps"])
    stop = start + int(cell["measure_steps"])
    complete_cycles = None
    phase_boundary_trimmed = False
    if common_gait_phases is not None:
        phases = np.asarray(common_gait_phases, dtype=float).reshape(-1)
        if phases.shape != (expected_samples,) or not np.all(np.isfinite(phases)):
            return _invalid_cell_row(cell, "common_gait_phases is present but invalid")
        if np.all(np.diff(phases) >= 0.0):
            first_boundary = math.ceil(float(phases[start]) - 1.0e-8)
            last_boundary = math.floor(float(phases[stop - 1]) + 1.0e-8)
            if last_boundary - first_boundary >= 1:
                first_index = int(np.searchsorted(phases, first_boundary, side="left"))
                last_index = int(np.searchsorted(phases, last_boundary, side="left"))
                if start <= first_index < last_index <= stop:
                    start = first_index
                    stop = last_index
                    complete_cycles = last_boundary - first_boundary
                    phase_boundary_trimmed = True
    if not phase_boundary_trimmed and period_s is not None:
        complete_cycles = math.floor((stop - start) * step_dt / period_s + 1.0e-9)
        if complete_cycles >= 1:
            cycle_samples = round(complete_cycles * period_s / step_dt)
            stop = min(stop, start + cycle_samples)
        else:
            complete_cycles = None
    if stop - start < 2:
        return _invalid_cell_row(cell, "measurement window has fewer than two samples")
    selected = slice(start, stop)
    selected_torques = torques[selected].reshape(-1, 4, 3)
    selected_powers = powers[selected].reshape(-1, 4, 3)
    effort_limits = joint_effort_limits.reshape(4, 3)
    metric_values = {
        "torque_squared": np.square(selected_torques).sum(axis=2),
        "normalized_torque_utilization": np.square(selected_torques / effort_limits).sum(axis=2),
        "absolute_work": np.abs(selected_powers).sum(axis=2),
        "vertical_grf_impulse": np.maximum(ground_reaction_forces[selected, :, 2], 0.0),
    }

    command_x = commands[selected, 0]
    velocity_x = velocities[selected, 0]
    command_sign = 1.0 if float(cell["velocity_mps"]) > 0.0 else -1.0
    position_displacement_x = float(positions[stop - 1, 0] - positions[start - 1, 0])
    directed_progress = command_sign * position_displacement_x
    integrated_body_x_progress = float(np.sum(command_sign * velocity_x) * step_dt)
    planar_tracking_errors = velocities[selected, :2] - commands[selected, :2]
    tracking_rmse = float(np.sqrt(np.mean(np.square(planar_tracking_errors).sum(axis=1))))
    yaw_tracking_rmse = float(np.sqrt(np.mean(np.square(velocities[selected, 2] - commands[selected, 2]))))
    tracking_success_threshold = 0.05 + 0.25 * abs(expected_velocity)
    planar_tracking_success = tracking_rmse <= tracking_success_threshold
    yaw_tracking_success = yaw_tracking_rmse <= YAW_TRACKING_SUCCESS_THRESHOLD_RADPS
    row = _invalid_cell_row(cell, "", "valid")
    row.update(
        {
            "reason": "",
            "sample_count": stop - start,
            "measurement_start_s": start * step_dt,
            "measurement_stop_s": stop * step_dt,
            "measurement_duration_s": (stop - start) * step_dt,
            "complete_cycles": complete_cycles,
            "phase_boundary_trimmed": phase_boundary_trimmed,
            "gait_period_s": period_s,
            "effort_limit_source": "recorded_joint_effort_limits",
            "effort_limit_min_nm": float(np.min(joint_effort_limits)),
            "effort_limit_max_nm": float(np.max(joint_effort_limits)),
            "command_x_mean_mps": float(np.mean(command_x)),
            "measured_x_mean_mps": float(np.mean(velocity_x)),
            "tracking_rmse_mps": tracking_rmse,
            "yaw_tracking_rmse_radps": yaw_tracking_rmse,
            "tracking_success_threshold_mps": tracking_success_threshold,
            "yaw_tracking_success_threshold_radps": YAW_TRACKING_SUCCESS_THRESHOLD_RADPS,
            "planar_tracking_success": planar_tracking_success,
            "yaw_tracking_success": yaw_tracking_success,
            "tracking_success": planar_tracking_success and yaw_tracking_success,
            "signed_directed_progress_m": directed_progress,
            "integrated_body_x_progress_m": integrated_body_x_progress,
            "position_vs_integrated_progress_difference_m": directed_progress - integrated_body_x_progress,
            "directed_progress_definition": (
                "commanded-sign world-x displacement across the same sample intervals as effort integration"
            ),
            "positive_directed_progress": directed_progress > 0.0,
        }
    )
    for metric, values in metric_values.items():
        summary = _metric_summary(values, step_dt)
        duration_s = float(row["measurement_duration_s"])
        prefix = f"{metric}_"
        for name, value in summary.items():
            row[prefix + name] = value
        total = float(summary["total_integral"])
        row[prefix + "total_per_s"] = total / duration_s
        row[prefix + "total_per_directed_m"] = total / directed_progress if directed_progress > 0.0 else None
    if directed_progress <= 0.0:
        row["status"] = "nonpositive_progress"
        row["reason"] = "commanded-direction progress is nonpositive; per-distance metrics are N/A"
    return row


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(sum(finite) / len(finite))


def aggregate_results(
    study: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate cells without allowing signs or gait-family sizes to bias balance."""
    expected_by_group: dict[tuple[str, float | None], int] = defaultdict(int)
    valid_by_group: dict[tuple[str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for cell, row in zip(study["cells"], rows, strict=True):
        keys = ((cell["family"], float(cell["velocity_mps"])), (cell["family"], None))
        for key in keys:
            expected_by_group[key] += 1
            if row["status"] == "valid":
                valid_by_group[key].append(row)

    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        velocities: list[float | None] = [float(value) for value in study["velocities_mps"]] + [None]
        for velocity in velocities:
            key = (family, velocity)
            expected = expected_by_group.get(key, 0)
            valid = valid_by_group.get(key, [])
            tracking_qualified = [row for row in valid if row.get("tracking_success") is True]
            complete = expected > 0 and len(valid) == expected
            tracking_complete = complete and len(tracking_qualified) == expected
            result: dict[str, Any] = {
                "scope": "family" if velocity is None else "family_velocity",
                "family": family,
                "velocity_mps": velocity,
                "expected_cells": expected,
                "valid_cells": len(valid),
                "coverage_fraction": len(valid) / expected if expected else 0.0,
                "complete": complete,
                "tracking_success_cells": len(tracking_qualified),
                "tracking_success_fraction": len(tracking_qualified) / len(valid) if valid else None,
                "tracking_qualified_complete": tracking_complete,
            }
            for metric in ALL_METRICS:
                signed = _mean(row.get(f"{metric}_signed_imbalance_percent") for row in valid)
                absolute = _mean(row.get(f"{metric}_abs_imbalance_percent") for row in valid)
                result[f"{metric}_observed_mean_signed_imbalance_percent"] = signed
                result[f"{metric}_observed_mean_abs_imbalance_percent"] = absolute
                result[f"{metric}_mean_signed_imbalance_percent"] = signed if complete else None
                result[f"{metric}_mean_abs_imbalance_percent"] = absolute if complete else None
                tracking_signed = _mean(row.get(f"{metric}_signed_imbalance_percent") for row in tracking_qualified)
                tracking_absolute = _mean(row.get(f"{metric}_abs_imbalance_percent") for row in tracking_qualified)
                result[f"{metric}_tracking_qualified_observed_mean_signed_imbalance_percent"] = tracking_signed
                result[f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"] = tracking_absolute
                result[f"{metric}_tracking_qualified_mean_signed_imbalance_percent"] = (
                    tracking_signed if tracking_complete else None
                )
                result[f"{metric}_tracking_qualified_mean_abs_imbalance_percent"] = (
                    tracking_absolute if tracking_complete else None
                )
                for quantity in ("total_integral", "total_per_s", "total_per_directed_m"):
                    observed_quantity = _mean(row.get(f"{metric}_{quantity}") for row in valid)
                    tracking_quantity = _mean(row.get(f"{metric}_{quantity}") for row in tracking_qualified)
                    result[f"{metric}_observed_mean_{quantity}"] = observed_quantity
                    result[f"{metric}_mean_{quantity}"] = observed_quantity if complete else None
                    result[f"{metric}_tracking_qualified_observed_mean_{quantity}"] = tracking_quantity
                    result[f"{metric}_tracking_qualified_mean_{quantity}"] = (
                        tracking_quantity if tracking_complete else None
                    )
            family_rows.append(result)

    family_summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    metrics: dict[str, Any] = {}
    for metric in ALL_METRICS:
        family_abs = {
            family: family_summary[family].get(f"{metric}_mean_abs_imbalance_percent") for family in FAMILY_ORDER
        }
        observed_family_abs = {
            family: family_summary[family].get(f"{metric}_observed_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        family_signed = {
            family: family_summary[family].get(f"{metric}_mean_signed_imbalance_percent") for family in FAMILY_ORDER
        }
        tracking_family_abs = {
            family: family_summary[family].get(f"{metric}_tracking_qualified_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        observed_tracking_family_abs = {
            family: family_summary[family].get(f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent")
            for family in FAMILY_ORDER
        }
        complete = all(value is not None for value in family_abs.values())
        tracking_complete = all(value is not None for value in tracking_family_abs.values())
        metric_result = {
            "label": METRIC_LABELS[metric],
            "integral_unit": METRIC_UNITS[metric],
            "family_mean_abs_imbalance_percent": family_abs,
            "family_mean_signed_imbalance_percent": family_signed,
            "family_balanced_mean_abs_imbalance_percent": _mean(family_abs.values()) if complete else None,
            "observed_family_balanced_mean_abs_imbalance_percent": _mean(observed_family_abs.values()),
            "complete": complete,
            "tracking_qualified_family_mean_abs_imbalance_percent": tracking_family_abs,
            "tracking_qualified_family_balanced_mean_abs_imbalance_percent": (
                _mean(tracking_family_abs.values()) if tracking_complete else None
            ),
            "tracking_qualified_observed_family_balanced_mean_abs_imbalance_percent": _mean(
                observed_tracking_family_abs.values()
            ),
            "tracking_qualified_complete": tracking_complete,
        }
        for quantity in ("total_integral", "total_per_s", "total_per_directed_m"):
            family_quantity = {
                family: family_summary[family].get(f"{metric}_mean_{quantity}") for family in FAMILY_ORDER
            }
            observed_family_quantity = {
                family: family_summary[family].get(f"{metric}_observed_mean_{quantity}") for family in FAMILY_ORDER
            }
            tracking_family_quantity = {
                family: family_summary[family].get(f"{metric}_tracking_qualified_mean_{quantity}")
                for family in FAMILY_ORDER
            }
            observed_tracking_family_quantity = {
                family: family_summary[family].get(f"{metric}_tracking_qualified_observed_mean_{quantity}")
                for family in FAMILY_ORDER
            }
            metric_result[f"family_mean_{quantity}"] = family_quantity
            metric_result[f"family_balanced_mean_{quantity}"] = _mean(family_quantity.values()) if complete else None
            metric_result[f"observed_family_balanced_mean_{quantity}"] = _mean(observed_family_quantity.values())
            metric_result[f"tracking_qualified_family_mean_{quantity}"] = tracking_family_quantity
            metric_result[f"tracking_qualified_family_balanced_mean_{quantity}"] = (
                _mean(tracking_family_quantity.values()) if tracking_complete else None
            )
            metric_result[f"tracking_qualified_observed_family_balanced_mean_{quantity}"] = _mean(
                observed_tracking_family_quantity.values()
            )
        metrics[metric] = metric_result
    valid_count = sum(row["status"] == "valid" for row in rows)
    tracking_success_count = sum(row.get("tracking_success") is True for row in rows if row["status"] == "valid")
    status_counts = {
        status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})
    }
    overall = {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "coverage": {
            "expected_cells": len(rows),
            "valid_cells": valid_count,
            "failed_or_missing_cells": len(rows) - valid_count,
            "coverage_fraction": valid_count / len(rows) if rows else 0.0,
            "complete": valid_count == len(rows),
            "tracking_success_cells": tracking_success_count,
            "tracking_success_fraction_of_valid": tracking_success_count / valid_count if valid_count else None,
            "status_counts": status_counts,
        },
        "aggregation": study["aggregation"],
        "tracking_qualification": {
            "planar": "tracking_rmse_mps <= 0.05 + 0.25 * abs(velocity_mps)",
            "yaw": f"yaw_tracking_rmse_radps <= {YAW_TRACKING_SUCCESS_THRESHOLD_RADPS:g}",
        },
        "metrics": metrics,
    }
    return family_rows, overall


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_family_figure(path: Path, family: str, family_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in family_rows if row["family"] == family and row["scope"] == "family_velocity"]
    velocities = [float(row["velocity_mps"]) for row in selected]
    x_positions = np.arange(len(velocities), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        values = [row[f"{metric}_tracking_qualified_observed_mean_abs_imbalance_percent"] for row in selected]
        heights = [np.nan if value is None else value for value in values]
        axis.bar(x_positions + (metric_index - 1) * width, heights, width, label=METRIC_LABELS[metric])
    axis.set_xticks(x_positions, [f"{velocity:+g}" for velocity in velocities])
    axis.set_xlabel("Commanded x velocity [m/s]")
    axis.set_ylabel("Tracking-qualified mean absolute front/hind imbalance [%]")
    axis.set_title(f"{family.replace('_', ' ').title()} tracking-qualified leg usage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _save_overall_figure(path: Path, overall: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [*FAMILY_ORDER, "overall"]
    x_positions = np.arange(len(labels), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        summary = overall["metrics"][metric]
        values = [summary["tracking_qualified_family_mean_abs_imbalance_percent"][family] for family in FAMILY_ORDER]
        values.append(summary["tracking_qualified_family_balanced_mean_abs_imbalance_percent"])
        heights = [np.nan if value is None else value for value in values]
        axis.bar(x_positions + (metric_index - 1) * width, heights, width, label=METRIC_LABELS[metric])
    axis.set_xticks(x_positions, [label.replace("_", " ").title() for label in labels])
    axis.set_ylabel("Tracking-qualified mean absolute front/hind imbalance [%]")
    axis.set_title("Tracking-qualified family-balanced leg usage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _save_coverage_figure(path: Path, family_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    coverage = [100.0 * summary[family]["coverage_fraction"] for family in FAMILY_ORDER]
    tracking_coverage = [
        100.0 * summary[family]["tracking_success_cells"] / summary[family]["expected_cells"]
        if summary[family]["expected_cells"]
        else 0.0
        for family in FAMILY_ORDER
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    x_positions = np.arange(len(FAMILY_ORDER), dtype=float)
    width = 0.38
    valid_bars = axis.bar(x_positions - width / 2, coverage, width, label="Valid rollout")
    tracking_bars = axis.bar(x_positions + width / 2, tracking_coverage, width, label="Tracking qualified")
    axis.bar_label(valid_bars, labels=[f"{value:.0f}%" for value in coverage], fontsize="small")
    axis.bar_label(tracking_bars, labels=[f"{value:.0f}%" for value in tracking_coverage], fontsize="small")
    axis.set_xticks(x_positions, [family.replace("_", " ").title() for family in FAMILY_ORDER])
    axis.set_ylim(0.0, 110.0)
    axis.set_ylabel("Valid cells [%]")
    axis.set_title("Leg-usage grid coverage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, format="svg")
    plt.close(figure)


def _write_report(
    path: Path,
    study: dict[str, Any],
    family_rows: Sequence[dict[str, Any]],
    overall: dict[str, Any],
) -> None:
    coverage = overall["coverage"]
    summary = {row["family"]: row for row in family_rows if row["scope"] == "family"}
    lines = [
        "# Leg usage grid analysis",
        "",
        f"- Robot: `{study['robot']}`",
        f"- Checkpoint: `{study['checkpoint']['path']}` (SHA-256 `{study['checkpoint']['sha256']}`)",
        f"- Grid: {len(study['gaits'])} gait rows x {len(study['velocities_mps'])} velocities",
        f"- Window: {study['settle_s']:g} s settling, then {study['measure_s']:g} s measurement",
        f"- Coverage: {coverage['valid_cells']}/{coverage['expected_cells']} valid cells "
        f"({100.0 * coverage['coverage_fraction']:.1f}%)",
        f"- Tracking quality: {coverage['tracking_success_cells']}/{coverage['valid_cells']} valid cells pass both "
        "`planar RMSE <= 0.05 + 0.25 * abs(vx)` [m/s] and `yaw RMSE <= 0.05` [rad/s]",
        f"- Analysis provenance: `analysis_provenance.json` "
        f"(record `{overall['analysis_provenance']['record_sha256']}`)",
        "",
        "Primary balance is the mean of per-cell absolute front/hind imbalance. Signed imbalance is retained in "
        "`cell_metrics.csv` and `family_metrics.csv`, but opposite signs never cancel in the primary score. The "
        "overall "
        "score gives each of trot, bound, half-bound, and gallop equal weight.",
        "",
        "## Coverage by family",
        "",
        "| Family | Valid | Expected | Coverage | Tracking pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILY_ORDER:
        row = summary[family]
        lines.append(
            f"| {family.replace('_', ' ')} | {row['valid_cells']} | {row['expected_cells']} | "
            f"{100.0 * row['coverage_fraction']:.1f}% | {row['tracking_success_cells']}/{row['valid_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Family-balanced primary results",
            "",
            "| Metric | All-valid mean absolute imbalance | Tracking-qualified mean | Valid complete | "
            "Tracking complete |",
            "|---|---:|---:|:---:|:---:|",
        ]
    )
    for metric in PRIMARY_METRICS:
        result = overall["metrics"][metric]
        value = result["family_balanced_mean_abs_imbalance_percent"]
        tracking_value = result["tracking_qualified_family_balanced_mean_abs_imbalance_percent"]
        formatted = "N/A" if value is None else f"{value:.3f}%"
        tracking_formatted = "N/A" if tracking_value is None else f"{tracking_value:.3f}%"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {formatted} | {tracking_formatted} | "
            f"{'yes' if result['complete'] else 'no'} | "
            f"{'yes' if result['tracking_qualified_complete'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Tracking-qualified usage rates",
            "",
            "Complete-cycle trimming can produce slightly different measurement durations, so rates are compared "
            "instead of raw totals.",
            "",
            "| Metric | Family-balanced total / s | Family-balanced total / directed m |",
            "|---|---:|---:|",
        ]
    )
    for metric in PRIMARY_METRICS:
        result = overall["metrics"][metric]
        per_s = result["tracking_qualified_family_balanced_mean_total_per_s"]
        per_m = result["tracking_qualified_family_balanced_mean_total_per_directed_m"]
        formatted_per_s = "N/A" if per_s is None else f"{per_s:.6g}"
        formatted_per_m = "N/A" if per_m is None else f"{per_m:.6g}"
        lines.append(f"| {METRIC_LABELS[metric]} | {formatted_per_s} | {formatted_per_m} |")
    lines.extend(
        [
            "",
            "A cell with missing/short data, an episode termination, invalid arrays, or nonpositive "
            "commanded-direction progress remains explicit in `coverage.csv`. Cost-per-distance fields are N/A for "
            "nonpositive progress; the "
            "cell's time-normalized balance metrics remain available when its rollout is otherwise valid.",
            "Poor command tracking is retained rather than filtered and is flagged using "
            "`tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and `yaw_tracking_rmse_radps <= 0.05`. "
            "The category and overall SVG figures use the "
            "tracking-qualified aggregates; the CSV/JSON tables retain both views.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_study(study_path: Path) -> dict[str, Any]:
    """Analyze all available grid cells and regenerate tables, figures, and report."""
    study_path = study_path.resolve()
    study = load_study_manifest(study_path)
    study_root = study_path.parent
    rows = [analyze_cell(study, cell, study_root) for cell in study["cells"]]
    family_rows, overall = aggregate_results(study, rows)
    metrics_dir = study_root / "metrics"
    figures_dir = study_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_provenance = {
        "schema_version": 1,
        "analysis_method_version": METHOD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
        "git_commit": _git_value(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
        "git_branch": _git_value(Path(__file__).resolve().parents[2], "branch", "--show-current"),
        "input_study_path": str(study_path),
        "input_study_sha256": _sha256_file(study_path),
        "input_study_identity_sha256": study["study_identity_sha256"],
    }
    analysis_provenance["record_sha256"] = _canonical_sha256(analysis_provenance)
    overall["analysis_provenance"] = {
        "path": "metrics/analysis_provenance.json",
        "record_sha256": analysis_provenance["record_sha256"],
        "analyzer_sha256": analysis_provenance["analyzer_sha256"],
        "input_study_sha256": analysis_provenance["input_study_sha256"],
    }
    _write_csv(metrics_dir / "cell_metrics.csv", rows)
    _write_csv(metrics_dir / "family_metrics.csv", family_rows)
    coverage_rows = [
        {
            "cell_id": row["cell_id"],
            "gait_index": row["gait_index"],
            "family": row["family"],
            "velocity_mps": row["velocity_mps"],
            "seed": row["seed"],
            "status": row["status"],
            "reason": row["reason"],
            "positive_directed_progress": row.get("positive_directed_progress"),
            "tracking_rmse_mps": row.get("tracking_rmse_mps"),
            "yaw_tracking_rmse_radps": row.get("yaw_tracking_rmse_radps"),
            "tracking_success_threshold_mps": row.get("tracking_success_threshold_mps"),
            "yaw_tracking_success_threshold_radps": row.get("yaw_tracking_success_threshold_radps"),
            "planar_tracking_success": row.get("planar_tracking_success"),
            "yaw_tracking_success": row.get("yaw_tracking_success"),
            "tracking_success": row.get("tracking_success"),
            "relative_output_dir": row["relative_output_dir"],
        }
        for row in rows
    ]
    _write_csv(metrics_dir / "coverage.csv", coverage_rows)
    _write_json_atomic(metrics_dir / "analysis_provenance.json", analysis_provenance)
    _write_json_atomic(metrics_dir / "overall_metrics.json", overall)
    for family in FAMILY_ORDER:
        _save_family_figure(figures_dir / f"{family}.svg", family, family_rows)
    _save_overall_figure(figures_dir / "overall.svg", overall)
    _save_coverage_figure(figures_dir / "coverage.svg", family_rows)
    _write_report(metrics_dir / "REPORT.md", study, family_rows, overall)
    return overall


if __name__ == "__main__":
    from symm_cli import main

    raise SystemExit(main(["analyze_leg_usage", *sys.argv[1:]]))
